#nullable enable
using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using Terraria;
using BossMLMod;
using BossMLMod.Helpers;

namespace BossMLMod.Networking;

/// <summary>
/// Complete game state observation sent to Python each tick.
/// Serialized as JSON over TCP (newline-delimited).
/// </summary>
public class StatePacket
{
    [JsonPropertyName("tick")]
    public long Tick { get; set; }

    [JsonPropertyName("episode")]
    public int Episode { get; set; }

    [JsonPropertyName("step")]
    public int Step { get; set; }

    [JsonPropertyName("done")]
    public bool Done { get; set; }

    [JsonPropertyName("player")]
    public PlayerObs Player { get; set; } = new();

    [JsonPropertyName("bosses")]
    public List<BossObs> Bosses { get; set; } = new();

    [JsonPropertyName("projectiles")]
    public List<ProjectileObs> Projectiles { get; set; } = new();

    [JsonPropertyName("environment")]
    public EnvironmentObs Environment { get; set; } = new();

    [JsonPropertyName("reward_components")]
    public RewardComponents RewardComponents { get; set; } = new();

    [JsonPropertyName("reload_config")]
    public bool ReloadConfig { get; set; }

    [JsonPropertyName("boss_type")]
    public int BossType { get; set; }

    [JsonPropertyName("loadout")]
    public LoadoutData? Loadout { get; set; }

    /// <summary>
    /// Build a complete state packet from the current game state.
    /// Must be called on the game thread (after all updates are done).
    /// </summary>
    public static StatePacket Build(EpisodeState ep)
    {
        var player = Main.LocalPlayer;
        var mlPlayer = player.GetModPlayer<BossMLPlayer>();
        var config = BossMLConfig.Instance;

        var packet = new StatePacket
        {
            Tick = Main.GameUpdateCount,
            Episode = ep.EpisodeNumber,
            Step = ep.StepCount,
            Done = ep.IsDone,
        };

        // --- Player observation ---
        var ps = mlPlayer.ExtractState();
        packet.Player = new PlayerObs
        {
            PosX = ps.PosX,
            PosY = ps.PosY,
            VelX = ps.VelX,
            VelY = ps.VelY,
            HP = ps.HP,
            MaxHP = ps.MaxHP,
            Mana = ps.Mana,
            MaxMana = ps.MaxMana,
            Defense = ps.Defense,
            Direction = ps.Direction,
            SelectedItem = ps.SelectedItem,
            ItemType = ps.ItemType,
            ItemDamage = ps.ItemDamage,
            ItemUseTime = ps.ItemUseTime,
            ItemCooldown = ps.ItemCooldown,
            Grounded = ps.Grounded,
            Grappled = ps.Grappled,
            WingsAvailable = ps.WingsAvailable,
            FlightTime = ps.FlightTime,
            MaxFlightTime = ps.MaxFlightTime,
            PotionSickness = ps.PotionSickness,
            DashCooldown = ps.DashCooldown,
            Buffs = new List<int[]>(),
        };
        if (ps.Buffs != null)
        {
            foreach (var b in ps.Buffs)
                packet.Player.Buffs.Add(new[] { b.Type, b.Duration });
        }

        // --- Boss observations ---
        var bossNpcs = BossHelper.GetActiveBosses();
        foreach (var boss in bossNpcs)
        {
            var bossObs = new BossObs
            {
                WhoAmI = boss.whoAmI,
                NpcType = boss.type,
                PosX = boss.Center.X,
                PosY = boss.Center.Y,
                VelX = boss.velocity.X,
                VelY = boss.velocity.Y,
                HP = boss.life,
                MaxHP = boss.lifeMax,
                AI0 = boss.ai[0],
                AI1 = boss.ai[1],
                AI2 = boss.ai[2],
                AI3 = boss.ai[3],
                Active = boss.active,
                Immune = boss.immortal || boss.dontTakeDamage,
                Segments = new List<SegmentObs>(),
            };

            // Multi-segment boss parts
            var segments = BossHelper.GetBossSegments(boss, config.MaxSegments, player);
            foreach (var seg in segments)
            {
                bossObs.Segments.Add(new SegmentObs
                {
                    WhoAmI = seg.whoAmI,
                    PosX = seg.Center.X,
                    PosY = seg.Center.Y,
                    HP = seg.life,
                    MaxHP = seg.lifeMax,
                });
            }

            packet.Bosses.Add(bossObs);
        }

        // --- Hostile projectile observations ---
        var hostileProjs = ProjectileHelper.GetNearbyHostile(
            player.Center,
            config.ProjectileScanRadius,
            config.MaxProjectiles
        );
        foreach (var proj in hostileProjs)
        {
            packet.Projectiles.Add(new ProjectileObs
            {
                Type = proj.type,
                PosX = proj.Center.X,
                PosY = proj.Center.Y,
                VelX = proj.velocity.X,
                VelY = proj.velocity.Y,
                Damage = proj.damage,
                Width = proj.width,
                Height = proj.height,
            });
        }

        // --- Environment ---
        packet.Environment = new EnvironmentObs
        {
            Time = (float)Main.time,
            DayTime = Main.dayTime,
        };

        // --- Reward components (computed by episode tracker) ---
        packet.RewardComponents = ep.ComputeRewardComponents(player, bossNpcs);

        // --- Boss type and loadout (for model metadata) ---
        packet.BossType = config.BossNpcType;

        return packet;
    }
}

// --- Observation sub-structures ---

public class PlayerObs
{
    [JsonPropertyName("pos_x")] public float PosX { get; set; }
    [JsonPropertyName("pos_y")] public float PosY { get; set; }
    [JsonPropertyName("vel_x")] public float VelX { get; set; }
    [JsonPropertyName("vel_y")] public float VelY { get; set; }
    [JsonPropertyName("hp")] public int HP { get; set; }
    [JsonPropertyName("max_hp")] public int MaxHP { get; set; }
    [JsonPropertyName("mana")] public int Mana { get; set; }
    [JsonPropertyName("max_mana")] public int MaxMana { get; set; }
    [JsonPropertyName("defense")] public int Defense { get; set; }
    [JsonPropertyName("direction")] public int Direction { get; set; }
    [JsonPropertyName("selected_item")] public int SelectedItem { get; set; }
    [JsonPropertyName("item_type")] public int ItemType { get; set; }
    [JsonPropertyName("item_damage")] public int ItemDamage { get; set; }
    [JsonPropertyName("item_use_time")] public int ItemUseTime { get; set; }
    [JsonPropertyName("item_cooldown")] public int ItemCooldown { get; set; }
    [JsonPropertyName("grounded")] public bool Grounded { get; set; }
    [JsonPropertyName("grappled")] public bool Grappled { get; set; }
    [JsonPropertyName("wings_available")] public bool WingsAvailable { get; set; }
    [JsonPropertyName("flight_time")] public int FlightTime { get; set; }
    [JsonPropertyName("max_flight_time")] public int MaxFlightTime { get; set; }
    [JsonPropertyName("potion_sickness")] public int PotionSickness { get; set; }
    [JsonPropertyName("dash_cooldown")] public int DashCooldown { get; set; }
    [JsonPropertyName("buffs")] public List<int[]> Buffs { get; set; } = new();
}

public class BossObs
{
    [JsonPropertyName("whoami")] public int WhoAmI { get; set; }
    [JsonPropertyName("npc_type")] public int NpcType { get; set; }
    [JsonPropertyName("pos_x")] public float PosX { get; set; }
    [JsonPropertyName("pos_y")] public float PosY { get; set; }
    [JsonPropertyName("vel_x")] public float VelX { get; set; }
    [JsonPropertyName("vel_y")] public float VelY { get; set; }
    [JsonPropertyName("hp")] public int HP { get; set; }
    [JsonPropertyName("max_hp")] public int MaxHP { get; set; }
    [JsonPropertyName("ai0")] public float AI0 { get; set; }
    [JsonPropertyName("ai1")] public float AI1 { get; set; }
    [JsonPropertyName("ai2")] public float AI2 { get; set; }
    [JsonPropertyName("ai3")] public float AI3 { get; set; }
    [JsonPropertyName("active")] public bool Active { get; set; }
    [JsonPropertyName("immune")] public bool Immune { get; set; }
    [JsonPropertyName("segments")] public List<SegmentObs> Segments { get; set; } = new();
}

public class SegmentObs
{
    [JsonPropertyName("whoami")] public int WhoAmI { get; set; }
    [JsonPropertyName("pos_x")] public float PosX { get; set; }
    [JsonPropertyName("pos_y")] public float PosY { get; set; }
    [JsonPropertyName("hp")] public int HP { get; set; }
    [JsonPropertyName("max_hp")] public int MaxHP { get; set; }
}

public class ProjectileObs
{
    [JsonPropertyName("type")] public int Type { get; set; }
    [JsonPropertyName("pos_x")] public float PosX { get; set; }
    [JsonPropertyName("pos_y")] public float PosY { get; set; }
    [JsonPropertyName("vel_x")] public float VelX { get; set; }
    [JsonPropertyName("vel_y")] public float VelY { get; set; }
    [JsonPropertyName("damage")] public int Damage { get; set; }
    [JsonPropertyName("width")] public int Width { get; set; }
    [JsonPropertyName("height")] public int Height { get; set; }
}

public class EnvironmentObs
{
    [JsonPropertyName("time")] public float Time { get; set; }
    [JsonPropertyName("day_time")] public bool DayTime { get; set; }
}

public class RewardComponents
{
    [JsonPropertyName("boss_hp_delta")] public float BossHpDelta { get; set; }
    [JsonPropertyName("player_hp_delta")] public float PlayerHpDelta { get; set; }
    [JsonPropertyName("boss_killed")] public bool BossKilled { get; set; }
    [JsonPropertyName("boss_despawned")] public bool BossDespawned { get; set; }
    [JsonPropertyName("player_died")] public bool PlayerDied { get; set; }
    [JsonPropertyName("distance_to_boss")] public float DistanceToBoss { get; set; }
    [JsonPropertyName("time_elapsed")] public int TimeElapsed { get; set; }
}
