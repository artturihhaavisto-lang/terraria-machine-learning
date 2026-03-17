#nullable enable
using System;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.GameInput;
using Terraria.ModLoader;

namespace BossMLMod;

/// <summary>
/// Handles ML input injection and player state observation.
///
/// Input injection strategy (three layers, because tModLoader hook ordering
/// varies across builds and we need at least one to stick):
///
///   1. PreUpdate — runs BEFORE vanilla reads keyboard input.
///      We overwrite PlayerInput.Triggers.Current.KeyStatus so that when
///      vanilla does `controlLeft = Triggers.Current.Left`, it reads OUR values.
///      We also set direct state here (aim, selectedItem) which we know works.
///
///   2. SetControls — runs AFTER vanilla reads keyboard input (in builds
///      that support it). We directly override Player.control* flags + release flags.
///
///   3. Diagnostic — periodic chat output showing received action values
///      so we can verify the TCP exchange is working.
/// </summary>
public class BossMLPlayer : ModPlayer
{
    public override void PreUpdate()
    {
        var sys = BossMLSystem.Instance;
        if (sys == null || !sys.MLModeActive) return;

        var action = sys.PendingAction;
        if (action == null) return;

        // --- Layer 1: Inject into the input trigger system ---
        // PlayerInput.UpdateInput() has already run (in Main.DoUpdate, before Player.Update),
        // so Triggers.Current contains the current keyboard state.
        // By overriding KeyStatus HERE, the vanilla code that reads
        //   controlLeft = PlayerInput.Triggers.Current.Left
        // will pick up our ML-injected values instead of keyboard state.
        if (Player.whoAmI == Main.myPlayer)
        {
            var keys = PlayerInput.Triggers.Current.KeyStatus;
            if (keys != null)
            {
                keys["Left"] = action.Move == 0;
                keys["Right"] = action.Move == 2;
                keys["Up"] = action.Jump == 1 && Player.velocity.Y != 0f;
                keys["Down"] = false;
                keys["Jump"] = action.Jump == 1;
                keys["MouseLeft"] = action.UseItem == 1;
                keys["MouseRight"] = false;
                keys["Grapple"] = action.Hook == 1;
                keys["QuickHeal"] = action.QuickHeal == 1;
                keys["QuickMana"] = false;
                keys["Throw"] = false;
                keys["QuickMount"] = false;
                keys["SmartSelect"] = false;
                keys["SmartCursor"] = false;
            }
        }

        // --- Dash: fake a double-tap by priming the timer before vanilla checks ---
        // Terraria's Player.Update() checks: if (control{Dir} && release{Dir} && doubleTapCardinalTimer[dir] > 0)
        // We prime the timer here in PreUpdate so that when vanilla runs the check
        // (after SetControls sets control+release), it triggers KeyDoubleTap → dash.
        // Cardinal indices: 0=down, 1=up, 2=left, 3=right
        if (action.Dash == 1 && action.Move != 1)
        {
            int cardinalDir = action.Move == 0 ? 2 : 3; // left=2, right=3
            Player.doubleTapCardinalTimer[cardinalDir] = 15;
        }

        // --- Direct state (not control flags, so nothing overwrites these) ---
        Player.selectedItem = 0;
        ApplyAim(action.AimSector);
    }

    /// <summary>
    /// Fallback: runs after vanilla input reading (in tModLoader builds that call it).
    /// Directly overrides control flags and release flags.
    /// </summary>
    public override void SetControls()
    {
        var sys = BossMLSystem.Instance;
        if (sys == null || !sys.MLModeActive) return;

        var action = sys.PendingAction;
        if (action == null) return;

        // Clear all controls
        Player.controlLeft = false;
        Player.controlRight = false;
        Player.controlUp = false;
        Player.controlDown = false;
        Player.controlJump = false;
        Player.controlUseItem = false;
        Player.controlUseTile = false;
        Player.controlHook = false;
        Player.controlMount = false;
        Player.controlQuickHeal = false;
        Player.controlQuickMana = false;
        Player.controlTorch = false;
        Player.controlSmart = false;
        Player.controlThrow = false;

        // Apply ML actions
        Player.controlLeft = action.Move == 0;
        Player.controlRight = action.Move == 2;
        Player.controlJump = action.Jump == 1;
        Player.controlUp = action.Jump == 1 && Player.velocity.Y != 0f;
        Player.controlUseItem = action.UseItem == 1;
        Player.controlHook = action.Hook == 1;
        Player.controlQuickHeal = action.QuickHeal == 1;

        // Force release flags so every press is treated as fresh
        Player.releaseJump = true;
        Player.releaseUseItem = true;
        Player.releaseHook = true;
        Player.releaseQuickHeal = true;

        // Dash: set release flag for move direction so vanilla double-tap check fires
        if (action.Dash == 1 && action.Move != 1)
        {
            if (action.Move == 0)
                Player.releaseLeft = true;
            else
                Player.releaseRight = true;
        }
    }

    private void ApplyAim(int sector)
    {
        var config = BossMLConfig.Instance;
        float aimDist = config.AimDistance;

        float angleRad = sector * (MathF.PI * 2f / 16f);

        float dx = MathF.Cos(angleRad) * aimDist;
        float dy = MathF.Sin(angleRad) * aimDist;

        float targetWorldX = Player.Center.X + dx;
        float targetWorldY = Player.Center.Y + dy;

        Main.mouseX = (int)(targetWorldX - Main.screenPosition.X);
        Main.mouseY = (int)(targetWorldY - Main.screenPosition.Y);

        Player.direction = dx >= 0f ? 1 : -1;
        Player.itemRotation = MathF.Atan2(dy * Player.direction, dx * Player.direction);
    }

    /// <summary>
    /// Extract full player state for the observation packet.
    /// </summary>
    public PlayerState ExtractState()
    {
        var state = new PlayerState
        {
            PosX = Player.Center.X,
            PosY = Player.Center.Y,
            VelX = Player.velocity.X,
            VelY = Player.velocity.Y,
            HP = Player.statLife,
            MaxHP = Player.statLifeMax2,
            Mana = Player.statMana,
            MaxMana = Player.statManaMax2,
            Defense = Player.statDefense,
            Direction = Player.direction,
            SelectedItem = Player.selectedItem,
            Grounded = Player.velocity.Y == 0f && !Player.mount.Active,
            Grappled = Player.grappling[0] >= 0,
            WingsAvailable = Player.wingTimeMax > 0,
            FlightTime = (int)Player.wingTime,
            MaxFlightTime = (int)Player.wingTimeMax,
            PotionSickness = GetBuffDuration(21),
            DashCooldown = Player.dashDelay,
        };

        if (Player.selectedItem >= 0 && Player.selectedItem < Player.inventory.Length)
        {
            var item = Player.inventory[Player.selectedItem];
            state.ItemType = item.type;
            state.ItemDamage = item.damage;
            state.ItemUseTime = item.useTime;
            state.ItemCooldown = Player.itemAnimation;
        }

        state.Buffs = new BuffInfo[Player.MaxBuffs];
        for (int i = 0; i < Player.MaxBuffs; i++)
        {
            state.Buffs[i] = new BuffInfo
            {
                Type = Player.buffType[i],
                Duration = Player.buffTime[i]
            };
        }

        return state;
    }

    private int GetBuffDuration(int buffType)
    {
        for (int i = 0; i < Player.MaxBuffs; i++)
        {
            if (Player.buffType[i] == buffType)
                return Player.buffTime[i];
        }
        return 0;
    }
}

// --- Data structures for player state observation ---

public class PlayerState
{
    public float PosX { get; set; }
    public float PosY { get; set; }
    public float VelX { get; set; }
    public float VelY { get; set; }
    public int HP { get; set; }
    public int MaxHP { get; set; }
    public int Mana { get; set; }
    public int MaxMana { get; set; }
    public int Defense { get; set; }
    public int Direction { get; set; }
    public int SelectedItem { get; set; }
    public int ItemType { get; set; }
    public int ItemDamage { get; set; }
    public int ItemUseTime { get; set; }
    public int ItemCooldown { get; set; }
    public bool Grounded { get; set; }
    public bool Grappled { get; set; }
    public bool WingsAvailable { get; set; }
    public int FlightTime { get; set; }
    public int MaxFlightTime { get; set; }
    public int PotionSickness { get; set; }
    public int DashCooldown { get; set; }
    public BuffInfo[] Buffs { get; set; } = Array.Empty<BuffInfo>();
}

public class BuffInfo
{
    public int Type { get; set; }
    public int Duration { get; set; }
}
