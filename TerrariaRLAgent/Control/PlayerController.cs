using System;
using Microsoft.Xna.Framework;
using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Control
{
    /// <summary>
    /// Translates an <see cref="ActionData"/> struct from the Python ML backend into
    /// Terraria player inputs.  Must be called from the game (main) thread inside
    /// <c>Player.PreUpdate()</c> after <c>SuppressRealInput()</c>.
    /// </summary>
    public static class PlayerController
    {
        // How many pixels from the player centre the synthetic cursor is placed
        // when aiming. Large enough that it covers the entire screen.
        private const float AimRadius = 600f;

        // Dash state – we simulate a left/right double-tap by toggling the dash
        // direction flag for one tick.
        private static int s_dashRequestTick = -1;
        private static int s_dashDir         = 0;

        /// <summary>
        /// Applies all fields of <paramref name="action"/> to <paramref name="player"/>.
        /// </summary>
        public static void ApplyAction(Player player, ActionData action)
        {
            ApplyMovement(player, action);
            ApplyJump(player, action);
            ApplyUseItem(player, action);
            ApplyAim(player, action);
            ApplyDash(player, action);
            ApplyGrapple(player, action);
            ApplyHeal(player, action);
        }

        // -----------------------------------------------------------------------
        // Individual action handlers
        // -----------------------------------------------------------------------

        private static void ApplyMovement(Player player, ActionData action)
        {
            // movement[0]=left, movement[1]=right, movement[2]=up, movement[3]=down
            int[] m = action.Movement;
            if (m == null || m.Length < 4) return;

            player.controlLeft  = m[0] != 0;
            player.controlRight = m[1] != 0;
            player.controlUp    = m[2] != 0;
            player.controlDown  = m[3] != 0;
        }

        private static void ApplyJump(Player player, ActionData action)
        {
            player.controlJump = action.Jump != 0;
        }

        private static void ApplyUseItem(Player player, ActionData action)
        {
            player.controlUseItem = action.UseItem != 0;
        }

        private static void ApplyAim(Player player, ActionData action)
        {
            // Convert aim_angle (radians, 0=right, CW positive in screen space)
            // to a screen-space pixel coordinate.
            // Terraria treats Y-axis as downward, so angle=0 → right, angle=PI/2 → down.
            float angle = action.AimAngle;

            // Player centre in world coords
            Vector2 playerCenter = player.Center;

            // Direction vector in world space
            Vector2 worldAim = new(
                (float)Math.Cos(angle) * AimRadius,
                (float)Math.Sin(angle) * AimRadius
            );

            // Convert world offset to screen pixel offset using Main.screenPosition
            Vector2 screenPos = playerCenter + worldAim - Main.screenPosition;

            Main.mouseX = (int)MathHelper.Clamp(screenPos.X, 0, Main.screenWidth  - 1);
            Main.mouseY = (int)MathHelper.Clamp(screenPos.Y, 0, Main.screenHeight - 1);

            // Update player direction from aim angle
            player.direction = (Math.Cos(angle) >= 0) ? 1 : -1;
        }

        private static void ApplyDash(Player player, ActionData action)
        {
            if (action.Dash == 0) return;

            // Terraria dashing works by setting controlLeft/controlRight twice in
            // the same tick with dashDelay == 0. We simulate a right-dash by default
            // and let the aim direction determine left/right.
            int dir = player.direction; // 1=right, -1=left

            // Only trigger once per request; avoid spam
            int currentTick = (int)Main.GameUpdateCount;
            if (currentTick == s_dashRequestTick) return;

            s_dashRequestTick = currentTick;
            s_dashDir         = dir;

            if (dir > 0)
                player.controlRight = true;
            else
                player.controlLeft  = true;

            // Reset dash delay so Terraria registers the double-tap
            player.dashDelay = 0;
        }

        private static void ApplyGrapple(Player player, ActionData action)
        {
            player.controlHook = action.Grapple != 0;
        }

        private static void ApplyHeal(Player player, ActionData action)
        {
            if (action.Heal != 0)
                player.QuickHeal();
        }
    }
}
