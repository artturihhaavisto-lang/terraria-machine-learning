#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;
using Terraria.ModLoader;
using BossMLMod.Helpers;
using BossMLMod.Networking;

namespace BossMLMod;

/// <summary>
/// Core ModSystem: orchestrates TCP communication, episode lifecycle, and mod commands.
///
/// Tick flow when ML mode is active:
///   1. BossMLPlayer.PreUpdate() applies PendingAction (set last tick)
///   2. All game updates run (player, NPCs, projectiles)
///   3. PostUpdateEverything() extracts state, sends to Python, waits for action
///
/// Episode lifecycle:
///   - Episode starts when ML mode is enabled (or on auto-reset after fight end)
///   - Episode ends when boss dies, player dies, or timeout
///   - On end: send done=true state, optionally auto-reset after delay
/// </summary>
public class BossMLSystem : ModSystem
{
    public static BossMLSystem? Instance { get; private set; }

    // --- TCP ---
    public TCPServer? Server { get; private set; }

    // --- ML Control ---
    public bool MLModeActive { get; set; }
    public ActionPacket PendingAction { get; set; } = new();

    // --- Episode State ---
    public EpisodeState Episode { get; private set; } = new();

    // --- Auto-reset ---
    private int _resetCountdown = -1;

    // --- Auto-enable ML mode (headless fleets) ---
    private int _autoEnableCountdown = -1;

    // --- God mode ---
    public bool GodMode { get; set; }

    // --- Auto-loadout: restore saved inventory on episode reset ---
    public bool AutoLoadout { get; set; }

    // --- Frame skip ---
    private int _frameSkipCounter;

    // --- Tracks whether we already sent the final done=true state to Python ---
    private bool _doneSent;
    private bool _reloadConfig;
    private bool _sendLoadout;

    // --- Debug/stats ---
    private int _totalExchanges;
    private int _failedExchanges;
    private int _totalEpisodes;
    private int _totalKills;
    private int _totalDeaths;
    private int _totalDespawns;
    private long _lastExchangeTick;
    private bool _wasConnected;
    public bool DebugHUD { get; set; }

    public override void Load()
    {
        Instance = this;
    }

    public override void Unload()
    {
        Server?.Dispose();
        Server = null;
        Instance = null;
    }

    public override void OnWorldLoad()
    {
        LoadPersistedBossType();
        var config = BossMLConfig.Instance;
        Server?.Dispose();
        Server = new TCPServer(config.Port);
        Server.Start();
        Episode = new EpisodeState();
        MLModeActive = false;
        _resetCountdown = -1;
        _doneSent = false;
        _autoEnableCountdown = config.AutoEnableML ? 180 : -1;
    }

    public override void OnWorldUnload()
    {
        MLModeActive = false;
        Server?.Dispose();
        Server = null;
    }

    public override void PostUpdateEverything()
    {
        // --- Connection state change notifications ---
        if (Server != null)
        {
            bool connected = Server.IsConnected;
            if (connected && !_wasConnected)
                Main.NewText("[BML] Python agent CONNECTED", 100, 255, 100);
            else if (!connected && _wasConnected)
                Main.NewText("[BML] Python agent DISCONNECTED", 255, 100, 100);
            _wasConnected = connected;
        }

        // Auto-enable ML mode once the Python agent connects (headless fleets)
        if (!MLModeActive && _autoEnableCountdown > 0 && Server != null && Server.IsConnected
            && Main.LocalPlayer != null && Main.LocalPlayer.active && !Main.LocalPlayer.dead)
        {
            if (--_autoEnableCountdown == 0)
            {
                MLModeActive = true;
                _doneSent = false;
                ResetEpisode();
                Main.NewText("[BML] ML mode AUTO-ENABLED (agent connected)", 100, 255, 100);
            }
        }

        if (!MLModeActive || Server == null || !Server.IsConnected) return;

        // --- Debug HUD: show stats every 5 seconds ---
        if (DebugHUD && Main.GameUpdateCount % 300 == 0)
        {
            long ticksSinceExchange = Main.GameUpdateCount - _lastExchangeTick;
            var dbgAct = PendingAction;
            Main.NewText(
                $"[BML DBG] Ep:{Episode.EpisodeNumber} Step:{Episode.StepCount} " +
                $"Exch:{_totalExchanges} Fail:{_failedExchanges} " +
                $"LastAct:{ticksSinceExchange}t ago " +
                $"M={dbgAct.Move} J={dbgAct.Jump} U={dbgAct.UseItem} A={dbgAct.AimSector}",
                200, 200, 200);
        }

        var config = BossMLConfig.Instance;
        var player = Main.LocalPlayer;

        // --- God mode: prevent player death ---
        if (GodMode && player.statLife < player.statLifeMax2)
            player.statLife = player.statLifeMax2;

        // --- Episode lifecycle ---
        Episode.StepCount++;

        // Track boss HP while boss is alive — must happen BEFORE terminal check
        // so LastBossHP is current when we evaluate bossGone this frame
        if (Episode.FightStarted && !Episode.IsDone)
        {
            var activeBosses = BossHelper.GetActiveBosses();
            if (activeBosses.Count > 0)
                Episode.LastBossHP = activeBosses[0].life;
        }

        // Check terminal conditions
        bool playerDied = player.dead;
        bool bossGone = Episode.FightStarted && !BossHelper.AnyBossActive();
        bool bossKilled = false;
        bool bossDespawned = false;

        if (bossGone)
        {
            // Scan all NPC slots (including just-deactivated ones) for a boss with life <= 0.
            // When a boss dies its NPC slot becomes inactive in the same frame HP hits 0,
            // so AnyBossActive() returns false before we can read the live HP value.
            bool foundDeadBoss = false;
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                var npc = Main.npc[i];
                if (npc.type > 0 && npc.boss && npc.life <= 0)
                {
                    foundDeadBoss = true;
                    break;
                }
            }
            if (foundDeadBoss || Episode.LastBossHP <= 0)
                bossKilled = true;
            else
                bossDespawned = true;
        }

        // Player death takes priority — if the player died, the boss disappearing
        // is a side effect (bosses despawn when the player dies), not a "kite despawn".
        // Only flag BossDespawned when the boss left on its own (player too far, etc.)
        if (playerDied)
            bossDespawned = false;

        if (bossKilled || bossDespawned || playerDied)
        {
            Episode.IsDone = true;
            if (bossKilled) Episode.BossKilled = true;
            if (bossDespawned) Episode.BossDespawned = true;
            if (playerDied) Episode.PlayerDied = true;
        }

        // Detect fight start (first boss appearing)
        if (!Episode.FightStarted && BossHelper.AnyBossActive())
            Episode.FightStarted = true;

        // --- If episode is done and we already sent the done state, just tick the countdown ---
        if (Episode.IsDone && _doneSent)
        {
            TickResetCountdown(config);
            return;
        }

        // --- Frame skip: only exchange with Python every N ticks ---
        _frameSkipCounter++;
        if (_frameSkipCounter < config.FrameSkip && !Episode.IsDone)
            return;
        _frameSkipCounter = 0;

        // --- Build and send state ---
        var state = StatePacket.Build(Episode);
        if (_reloadConfig)
        {
            state.ReloadConfig = true;
            _reloadConfig = false;
        }
        if (_sendLoadout)
        {
            var loadout = new LoadoutData();
            for (int i = 0; i < 50 && i < player.inventory.Length; i++)
            {
                var item = player.inventory[i];
                if (item != null && item.type != ItemID.None)
                    loadout.Inventory.Add(new ItemSlot { Slot = i, Type = item.type, Stack = item.stack, Prefix = (byte)item.prefix });
            }
            for (int i = 0; i < player.armor.Length; i++)
            {
                var item = player.armor[i];
                if (item != null && item.type != ItemID.None)
                    loadout.Armor.Add(new ItemSlot { Slot = i, Type = item.type, Stack = 1, Prefix = (byte)item.prefix });
            }
            state.Loadout = loadout;
            _sendLoadout = false;
        }

        bool gotAction = Server.Exchange(state, out var action, config.ActionTimeoutMs);
        _totalExchanges++;
        if (gotAction)
        {
            PendingAction = action;
            _lastExchangeTick = Main.GameUpdateCount;

            // If Python requested loadout data, include it in the next state
            if (action.RequestLoadout)
                _sendLoadout = true;

            // Check for boss type change command from curriculum system
            if (action.SetBossType > 0)
            {
                config.BossNpcType = action.SetBossType;
                Mod.Logger.Info($"Boss type changed to {action.SetBossType} via agent command.");
            }

            // Restore loadout sent from Python (e.g., on model resume)
            if (action.SetLoadout != null)
            {
                RestoreLoadoutFromData(action.SetLoadout);
                Mod.Logger.Info("Loadout restored from Python agent.");
                Main.NewText("[BML] Loadout restored from model metadata", 100, 255, 100);
            }

            // Display message from Python in chat
            if (!string.IsNullOrEmpty(action.Message))
            {
                Main.NewText($"[BML] {action.Message}", 255, 200, 50);
            }
        }
        else
        {
            _failedExchanges++;
        }

        // --- Start auto-reset countdown on episode end ---
        if (Episode.IsDone && !_doneSent)
        {
            _doneSent = true;
            _totalEpisodes++;
            if (Episode.BossKilled) _totalKills++;
            if (Episode.PlayerDied) _totalDeaths++;
            if (Episode.BossDespawned) _totalDespawns++;

            // In-game episode end summary
            string outcome = Episode.BossKilled ? "BOSS KILLED" :
                             Episode.BossDespawned ? "BOSS DESPAWNED" :
                             Episode.PlayerDied ? "PLAYER DIED" : "UNKNOWN";
            byte r = Episode.BossKilled ? (byte)100 : (byte)255;
            byte g = Episode.BossKilled ? (byte)255 : (byte)100;
            Main.NewText($"[BML] Ep {Episode.EpisodeNumber} → {outcome} " +
                $"({Episode.StepCount} steps, {Episode.StepCount / 60f:F1}s) " +
                $"| Total: {_totalKills}W {_totalDeaths}D {_totalDespawns}DS", r, g, 100);

            if (config.AutoReset && _resetCountdown < 0)
            {
                // Despawn → reset immediately (no point waiting, boss is already gone)
                // Kill/Death → use configured delay for the player to see the outcome
                _resetCountdown = Episode.BossDespawned ? 30 : config.AutoResetDelayTicks;
                Mod.Logger.Info($"Episode {Episode.EpisodeNumber} ended. " +
                    $"BossKilled={Episode.BossKilled}, Despawned={Episode.BossDespawned}, PlayerDied={Episode.PlayerDied}, " +
                    $"Steps={Episode.StepCount}. Auto-resetting in {_resetCountdown / 60f:F1}s.");
            }
        }
    }

    private void TickResetCountdown(BossMLConfig config)
    {
        if (_resetCountdown > 0)
        {
            _resetCountdown--;
        }
        else if (_resetCountdown == 0)
        {
            _resetCountdown = -1;
            ResetEpisode();
        }
    }

    /// <summary>
    /// Reset for a new training episode:
    ///   1. Heal the player to full
    ///   2. Teleport to arena center
    ///   3. Clear hostile NPCs and projectiles
    ///   4. Spawn the configured boss
    ///   5. Increment episode counter
    /// </summary>
    public void ResetEpisode()
    {
        var config = BossMLConfig.Instance;
        var player = Main.LocalPlayer;

        // Force time of day if configured (night-only bosses on headless instances)
        switch ((config.ForceTimeOfDay ?? "none").ToLowerInvariant())
        {
            case "day":      SetTime(0, true);      break;
            case "noon":     SetTime(27000, true);  break;
            case "night":    SetTime(0, false);     break;
            case "midnight": SetTime(16200, false); break;
        }

        // Revive if dead — directly reset death state
        if (player.dead)
        {
            player.dead = false;
            player.respawnTimer = 0;
            player.ghost = false;
        }

        // Full heal
        player.statLife = player.statLifeMax2;
        player.statMana = player.statManaMax2;

        // Restore saved loadout if enabled
        if (AutoLoadout)
            RestoreLoadout(config.BossNpcType);

        // Clear debuffs (indices where Main.debuff[type] is true)
        for (int i = 0; i < Player.MaxBuffs; i++)
        {
            int btype = player.buffType[i];
            if (btype > 0 && btype < Main.debuff.Length && Main.debuff[btype])
            {
                player.buffType[i] = 0;
                player.buffTime[i] = 0;
            }
        }

        // Teleport to arena center, or world spawn as fallback
        float teleX, teleY;
        if (config.ArenaCenterX != 0 && config.ArenaCenterY != 0)
        {
            teleX = config.ArenaCenterX;
            teleY = config.ArenaCenterY;
        }
        else
        {
            teleX = Main.spawnTileX * 16f;
            teleY = Main.spawnTileY * 16f;
        }
        player.position = new Vector2(teleX - player.width / 2f, teleY - player.height);
        player.velocity = Vector2.Zero;
        player.fallStart = (int)(player.position.Y / 16f);

        // Kill all existing boss NPCs and their segments
        for (int i = 0; i < Main.maxNPCs; i++)
        {
            var npc = Main.npc[i];
            if (npc.active && (npc.boss || npc.realLife >= 0))
            {
                npc.active = false;
                npc.life = 0;
            }
        }

        // Clear hostile projectiles
        for (int i = 0; i < Main.maxProjectiles; i++)
        {
            if (Main.projectile[i].active && Main.projectile[i].hostile)
            {
                Main.projectile[i].active = false;
            }
        }

        // New episode state
        int nextEpisode = Episode.EpisodeNumber + 1;
        Episode = new EpisodeState { EpisodeNumber = nextEpisode };
        _doneSent = false;

        // Spawn the boss
        SpawnBoss(config.BossNpcType);

        // Reset action
        PendingAction = new ActionPacket();
        _frameSkipCounter = 0;

        Mod.Logger.Info($"Episode {nextEpisode} started. Boss type: {config.BossNpcType}");
    }

    // ── Loadout save/restore ────────────────────────────────────

    private string GetBossTypePath()
        => Path.Combine(Main.SavePath, "BossMLMod", "boss_type.txt");

    private void SaveBossType(int bossType)
    {
        try
        {
            string dir = Path.Combine(Main.SavePath, "BossMLMod");
            Directory.CreateDirectory(dir);
            File.WriteAllText(GetBossTypePath(), bossType.ToString());
        }
        catch (Exception ex)
        {
            Mod.Logger.Warn($"Could not persist boss type: {ex.Message}");
        }
    }

    private void LoadPersistedBossType()
    {
        try
        {
            string path = GetBossTypePath();
            if (File.Exists(path) && int.TryParse(File.ReadAllText(path).Trim(), out int saved))
            {
                BossMLConfig.Instance.BossNpcType = saved;
                Mod.Logger.Info($"Loaded persisted boss type: {saved}");
            }
        }
        catch (Exception ex)
        {
            Mod.Logger.Warn($"Could not load persisted boss type: {ex.Message}");
        }
    }

    private string GetLoadoutPath(int bossType)
    {
        string dir = Path.Combine(Main.SavePath, "BossMLMod", "loadouts");
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, $"boss_{bossType}.json");
    }

    private void SaveLoadout(int bossType)
    {
        var player = Main.LocalPlayer;
        var data = new LoadoutData();

        // Inventory (first 50 slots)
        for (int i = 0; i < 50 && i < player.inventory.Length; i++)
        {
            var item = player.inventory[i];
            if (item != null && item.type != ItemID.None)
                data.Inventory.Add(new ItemSlot { Slot = i, Type = item.type, Stack = item.stack, Prefix = (byte)item.prefix });
        }

        // Armor + accessories (0-2 armor, 3-9 accessories)
        for (int i = 0; i < player.armor.Length; i++)
        {
            var item = player.armor[i];
            if (item != null && item.type != ItemID.None)
                data.Armor.Add(new ItemSlot { Slot = i, Type = item.type, Stack = 1, Prefix = (byte)item.prefix });
        }

        string json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(GetLoadoutPath(bossType), json);
    }

    private bool RestoreLoadout(int bossType)
    {
        string path = GetLoadoutPath(bossType);
        if (!File.Exists(path)) return false;

        try
        {
            string json = File.ReadAllText(path);
            var data = JsonSerializer.Deserialize<LoadoutData>(json);
            if (data == null) return false;

            var player = Main.LocalPlayer;

            // Clear inventory
            for (int i = 0; i < 50 && i < player.inventory.Length; i++)
            {
                player.inventory[i] = new Item();
            }

            // Restore inventory
            foreach (var slot in data.Inventory)
            {
                if (slot.Slot >= 0 && slot.Slot < player.inventory.Length)
                {
                    player.inventory[slot.Slot] = new Item();
                    player.inventory[slot.Slot].SetDefaults(slot.Type);
                    player.inventory[slot.Slot].stack = slot.Stack;
                    player.inventory[slot.Slot].prefix = slot.Prefix;
                }
            }

            // Clear armor/accessories
            for (int i = 0; i < player.armor.Length; i++)
            {
                player.armor[i] = new Item();
            }

            // Restore armor/accessories
            foreach (var slot in data.Armor)
            {
                if (slot.Slot >= 0 && slot.Slot < player.armor.Length)
                {
                    player.armor[slot.Slot] = new Item();
                    player.armor[slot.Slot].SetDefaults(slot.Type);
                    player.armor[slot.Slot].prefix = slot.Prefix;
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            Mod.Logger.Warn($"Failed to restore loadout: {ex.Message}");
            return false;
        }
    }

    private void RestoreLoadoutFromData(LoadoutData data)
    {
        var player = Main.LocalPlayer;

        // Clear inventory
        for (int i = 0; i < 50 && i < player.inventory.Length; i++)
            player.inventory[i] = new Item();

        // Restore inventory
        foreach (var slot in data.Inventory)
        {
            if (slot.Slot >= 0 && slot.Slot < player.inventory.Length)
            {
                player.inventory[slot.Slot] = new Item();
                player.inventory[slot.Slot].SetDefaults(slot.Type);
                player.inventory[slot.Slot].stack = slot.Stack;
                player.inventory[slot.Slot].prefix = slot.Prefix;
            }
        }

        // Clear armor/accessories
        for (int i = 0; i < player.armor.Length; i++)
            player.armor[i] = new Item();

        // Restore armor/accessories
        foreach (var slot in data.Armor)
        {
            if (slot.Slot >= 0 && slot.Slot < player.armor.Length)
            {
                player.armor[slot.Slot] = new Item();
                player.armor[slot.Slot].SetDefaults(slot.Type);
                player.armor[slot.Slot].prefix = slot.Prefix;
            }
        }
    }

    private void SpawnBoss(int npcType)
    {
        var player = Main.LocalPlayer;

        float spawnX = player.Center.X;
        float spawnY = player.Center.Y - 800f;

        NPC.NewNPC(
            NPC.GetBossSpawnSource(Main.myPlayer),
            (int)spawnX,
            (int)spawnY,
            npcType
        );
    }

    /// <summary>
    /// Clears the entire world and places evenly-spaced wooden platforms spanning
    /// the full width. Sets spawn and arena center to the world's middle.
    /// </summary>
    private void BuildArena(int platformCount)
    {
        int worldW = Main.maxTilesX;
        int worldH = Main.maxTilesY;

        // Hard boundaries: 200 tiles from top/bottom of world
        int roofY = 200;
        int floorY = worldH - 200;

        // Side margins
        int left = 40;
        int right = worldW - 40;

        // Playable area is between roof and floor
        int arenaTop = roofY + 3;    // just below the roof slab
        int arenaBottom = floorY - 3; // just above the floor slab
        int usableHeight = arenaBottom - arenaTop;

        // 1. Clear all tiles and walls in the entire world area
        for (int x = 0; x < worldW; x++)
        {
            for (int y = 0; y < worldH; y++)
            {
                var tile = Main.tile[x, y];
                tile.HasTile = false;
                tile.WallType = 0;
                tile.LiquidAmount = 0;
            }
        }

        // 2. Place hard roof (stone slab, 3 tiles thick)
        for (int x = 0; x < worldW; x++)
        {
            for (int dy = 0; dy < 3; dy++)
            {
                WorldGen.PlaceTile(x, roofY + dy, TileID.StoneSlab, forced: true);
            }
        }

        // 3. Place hard floor (stone slab, 3 tiles thick)
        for (int x = 0; x < worldW; x++)
        {
            for (int dy = 0; dy < 3; dy++)
            {
                WorldGen.PlaceTile(x, floorY + dy, TileID.StoneSlab, forced: true);
            }
        }

        // 4. Place platforms evenly across the playable area, with torches
        int torchSpacing = 15;
        for (int i = 0; i < platformCount; i++)
        {
            int y = arenaTop + (int)((i + 0.5f) / platformCount * usableHeight);
            for (int x = left; x < right; x++)
            {
                WorldGen.PlaceTile(x, y, TileID.Platforms, forced: true, style: 0);
            }
        }


        // 5. Set spawn point to world center (middle platform)
        int centerX = worldW / 2;
        int centerPlatformIndex = platformCount / 2;
        int centerY = arenaTop + (int)((centerPlatformIndex + 0.5f) / platformCount * usableHeight);
        Main.spawnTileX = centerX;
        Main.spawnTileY = centerY;

        // 6. Set arena center (for episode resets)
        BossMLConfig.Instance.ArenaCenterX = centerX * 16;
        BossMLConfig.Instance.ArenaCenterY = centerY * 16;

        // 7. Teleport player to the new spawn
        var player = Main.LocalPlayer;
        player.position = new Vector2(centerX * 16 - player.width / 2f, centerY * 16 - player.height);
        player.velocity = Vector2.Zero;
        player.fallStart = centerY;

        // 8. Force tile frame update for rendering
        for (int x = 0; x < worldW; x++)
        {
            for (int y = roofY; y <= floorY + 2; y++)
            {
                WorldGen.TileFrame(x, y);
            }
        }
    }

    public static void SetTime(double time, bool dayTime)
    {
        Main.dayTime = dayTime;
        Main.time = time;
    }

    // ==================== MOD COMMANDS ====================

    public static class Commands
    {
        public static void HandleCommand(string input)
        {
            var sys = Instance;
            if (sys == null) return;

            var parts = input.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) return;

            string sub = parts[0].ToLowerInvariant();

            switch (sub)
            {
                case "on":
                    sys.MLModeActive = true;
                    sys._doneSent = false;
                    sys.ResetEpisode();
                    ChatMessage("ML mode ENABLED. Boss spawned, agent controls the player.");
                    break;

                case "off":
                    sys.MLModeActive = false;
                    ChatMessage("ML mode DISABLED. Human controls restored.");
                    break;

                case "reset":
                    sys.ResetEpisode();
                    ChatMessage("Episode reset.");
                    break;

                case "god":
                    sys.GodMode = !sys.GodMode;
                    ChatMessage($"God mode: {(sys.GodMode ? "ON" : "OFF")}");
                    break;

                case "boss":
                    if (parts.Length >= 2 && int.TryParse(parts[1], out int bossId))
                    {
                        BossMLConfig.Instance.BossNpcType = bossId;
                        Instance?.SaveBossType(bossId);
                        ChatMessage($"Boss type set to {bossId} (saved).");
                    }
                    else
                    {
                        ChatMessage("Usage: /bml boss <npc_type_id>  (e.g., 50=King Slime, 4=Eye of Cthulhu)");
                    }
                    break;

                case "arena":
                    var p = Main.LocalPlayer;
                    BossMLConfig.Instance.ArenaCenterX = (int)p.Center.X;
                    BossMLConfig.Instance.ArenaCenterY = (int)p.Center.Y;
                    ChatMessage($"Arena center set to ({p.Center.X:F0}, {p.Center.Y:F0}).");
                    break;

                case "time":
                    if (parts.Length >= 2)
                    {
                        switch (parts[1].ToLowerInvariant())
                        {
                            case "noon":
                                SetTime(27000, true);
                                ChatMessage("Time set to noon.");
                                break;
                            case "midnight":
                                SetTime(16200, false);
                                ChatMessage("Time set to midnight.");
                                break;
                            case "night":
                                SetTime(0, false);
                                ChatMessage("Time set to 7:30 PM.");
                                break;
                            case "day":
                                SetTime(0, true);
                                ChatMessage("Time set to 4:30 AM.");
                                break;
                            default:
                                ChatMessage("Usage: /bml time <noon|midnight|night|day>");
                                break;
                        }
                    }
                    break;

                case "spawn":
                    sys.SpawnBoss(BossMLConfig.Instance.BossNpcType);
                    ChatMessage($"Spawned boss type {BossMLConfig.Instance.BossNpcType}.");
                    break;

                case "status":
                    ChatMessage($"ML={sys.MLModeActive} God={sys.GodMode} Debug={sys.DebugHUD} AutoLoadout={sys.AutoLoadout}");
                    ChatMessage($"TCP={sys.Server?.IsConnected ?? false} " +
                        $"Exchanges={sys._totalExchanges} Failed={sys._failedExchanges}");
                    ChatMessage($"Ep={sys.Episode.EpisodeNumber} Steps={sys.Episode.StepCount} " +
                        $"Done={sys.Episode.IsDone} Fight={sys.Episode.FightStarted}");
                    ChatMessage($"Lifetime: {sys._totalKills}W {sys._totalDeaths}D {sys._totalDespawns}DS " +
                        $"({sys._totalEpisodes} episodes)");
                    if (sys._totalExchanges > 0)
                    {
                        float failRate = 100f * sys._failedExchanges / sys._totalExchanges;
                        long ticksSince = Main.GameUpdateCount - sys._lastExchangeTick;
                        ChatMessage($"Fail rate: {failRate:F1}% | Last exchange: {ticksSince} ticks ago");
                    }
                    break;

                case "debug":
                    sys.DebugHUD = !sys.DebugHUD;
                    ChatMessage($"Debug HUD: {(sys.DebugHUD ? "ON (every 5s)" : "OFF")}");
                    break;

                case "buildarena":
                    int platCount = BossMLConfig.Instance.ArenaPlatforms;
                    if (parts.Length >= 2 && int.TryParse(parts[1], out int pc))
                        platCount = Math.Clamp(pc, 1, 200);
                    sys.BuildArena(platCount);
                    ChatMessage($"Arena built with {platCount} platforms. Spawn set to world center.");
                    break;

                case "reload":
                    sys._reloadConfig = true;
                    ChatMessage("Config reload requested (will apply on next exchange).");
                    break;

                case "saveloadout":
                    sys.SaveLoadout(BossMLConfig.Instance.BossNpcType);
                    ChatMessage($"Loadout saved for boss {BossMLConfig.Instance.BossNpcType}.");
                    break;

                case "loadloadout":
                    if (sys.RestoreLoadout(BossMLConfig.Instance.BossNpcType))
                        ChatMessage($"Loadout restored for boss {BossMLConfig.Instance.BossNpcType}.");
                    else
                        ChatMessage($"No saved loadout for boss {BossMLConfig.Instance.BossNpcType}.");
                    break;

                case "autoloadout":
                    sys.AutoLoadout = !sys.AutoLoadout;
                    ChatMessage($"Auto-loadout: {(sys.AutoLoadout ? "ON" : "OFF")}");
                    break;

                default:
                    ChatMessage("Commands: on, off, reset, reload, god, debug, boss <id>, arena, buildarena [n], time, spawn, status, saveloadout, loadloadout, autoloadout");
                    break;
            }
        }

        private static void ChatMessage(string msg)
        {
            Main.NewText("[BossML] " + msg, 100, 255, 100);
        }
    }
}

/// <summary>
/// Chat command handler. Players type "/bml ..." in chat.
/// </summary>
public class BossMLCommand : ModCommand
{
    public override string Command => "bml";
    public override CommandType Type => CommandType.Chat;
    public override string Usage => "/bml <on|off|reset|god|boss|arena|time|spawn|status>";
    public override string Description => "Control the Boss ML training framework.";

    public override void Action(CommandCaller caller, string input, string[] args)
    {
        // tModLoader may pass subcommands in args rather than input
        string cmd = args.Length > 0 ? string.Join(" ", args) : input;
        BossMLSystem.Commands.HandleCommand(cmd);
    }
}

/// <summary>
/// Tracks per-episode state: HP deltas, step count, terminal conditions.
/// </summary>
public class EpisodeState
{
    public int EpisodeNumber { get; set; }
    public int StepCount { get; set; }
    public bool IsDone { get; set; }
    public bool BossKilled { get; set; }
    public bool BossDespawned { get; set; }
    public bool PlayerDied { get; set; }
    public bool FightStarted { get; set; }
    public int LastBossHP { get; set; } = -1;

    private int _prevPlayerHP = -1;
    private int _prevBossHP = -1;

    public RewardComponents ComputeRewardComponents(Player player, List<NPC> bosses)
    {
        var rc = new RewardComponents();

        // Player HP delta (negative = took damage)
        if (_prevPlayerHP >= 0)
            rc.PlayerHpDelta = player.statLife - _prevPlayerHP;
        _prevPlayerHP = player.statLife;

        // Boss HP delta (negative = boss took damage)
        if (bosses.Count > 0)
        {
            var boss = bosses[0];
            if (_prevBossHP >= 0)
                rc.BossHpDelta = boss.life - _prevBossHP;
            _prevBossHP = boss.life;

            rc.DistanceToBoss = Vector2.Distance(player.Center, boss.Center);
        }

        rc.BossKilled = BossKilled;
        rc.BossDespawned = BossDespawned;
        rc.PlayerDied = PlayerDied;
        rc.TimeElapsed = StepCount;

        return rc;
    }
}

/// <summary>
/// Saved player loadout: inventory items, armor, and accessories.
/// Serialized to JSON per boss type.
/// </summary>
public class LoadoutData
{
    [JsonPropertyName("inventory")]
    public List<ItemSlot> Inventory { get; set; } = new();

    [JsonPropertyName("armor")]
    public List<ItemSlot> Armor { get; set; } = new();
}

public class ItemSlot
{
    [JsonPropertyName("slot")]
    public int Slot { get; set; }

    [JsonPropertyName("type")]
    public int Type { get; set; }

    [JsonPropertyName("stack")]
    public int Stack { get; set; }

    [JsonPropertyName("prefix")]
    public byte Prefix { get; set; }
}
