#nullable enable
using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Terraria;
using Terraria.GameContent;
using Terraria.ModLoader;
using BossMLMod.Helpers;
using BossMLMod.Networking;

namespace BossMLMod;

/// <summary>
/// Draws analytical overlays when ML mode + debug HUD are active:
///   - Aim direction laser (where the agent is pointing)
///   - Boss tracking line (player → boss, color-coded by distance)
///   - Distance threshold ring (shows the "safe zone" boundary)
///   - Action indicators (movement arrows, jump/attack icons)
///   - Boss HP bar overlay
///   - Agent trail (recent position history, fading tracer)
/// </summary>
public class BossMLOverlay : ModSystem
{
    // Position history for trail effect
    private readonly Queue<Vector2> _trail = new();
    private const int TrailLength = 60; // ~1 second of positions

    // Damage flash timer
    private int _damageFlash;
    private int _prevPlayerHP = -1;

    public override void PostDrawTiles()
    {
        var sys = BossMLSystem.Instance;
        if (sys == null || !sys.MLModeActive || !sys.DebugHUD) return;

        var sb = Main.spriteBatch;
        var player = Main.LocalPlayer;
        var action = sys.PendingAction;
        if (action == null) return;

        // We need to use screen-space drawing (subtract screenPosition)
        sb.Begin(SpriteSortMode.Deferred, BlendState.AlphaBlend, SamplerState.PointClamp,
            DepthStencilState.None, RasterizerState.CullNone, null, Main.GameViewMatrix.ZoomMatrix);

        try
        {
            Vector2 playerScreen = player.Center - Main.screenPosition;

            // --- Damage flash detection ---
            if (_prevPlayerHP >= 0 && player.statLife < _prevPlayerHP)
                _damageFlash = 15;
            _prevPlayerHP = player.statLife;
            if (_damageFlash > 0) _damageFlash--;

            // --- Trail (fading position history) ---
            _trail.Enqueue(player.Center);
            while (_trail.Count > TrailLength)
                _trail.Dequeue();
            DrawTrail(sb);

            // --- Boss tracking line ---
            DrawBossLine(sb, player, playerScreen);

            // --- Distance threshold ring ---
            var config = BossMLConfig.Instance;
            DrawCircle(sb, playerScreen, 600f, new Color(255, 255, 100, 30), 1f);

            // --- Aim direction laser ---
            DrawAimLaser(sb, player, playerScreen, action);

            // --- Action indicators ---
            DrawActionIndicators(sb, playerScreen, action);

            // --- Damage flash border ---
            if (_damageFlash > 0)
                DrawDamageFlash(sb);
        }
        finally
        {
            sb.End();
        }
    }

    private void DrawTrail(SpriteBatch sb)
    {
        var texture = TextureAssets.MagicPixel.Value;
        int i = 0;
        int count = _trail.Count;
        foreach (var pos in _trail)
        {
            float alpha = (float)i / count;
            float size = 2f + alpha * 3f;
            var color = Color.Lerp(new Color(209, 120, 50, 0), new Color(209, 160, 80, 180), alpha);
            Vector2 screen = pos - Main.screenPosition;
            sb.Draw(texture, screen, new Rectangle(0, 0, 1, 1), color,
                0f, new Vector2(0.5f), size, SpriteEffects.None, 0f);
            i++;
        }
    }

    private void DrawBossLine(SpriteBatch sb, Player player, Vector2 playerScreen)
    {
        var bosses = BossHelper.GetActiveBosses();
        if (bosses.Count == 0) return;

        var boss = bosses[0];
        Vector2 bossScreen = boss.Center - Main.screenPosition;
        float dist = Vector2.Distance(player.Center, boss.Center);

        // Color: green (close) → yellow (mid) → red (far/despawn risk)
        Color lineColor;
        if (dist < 400f)
            lineColor = new Color(100, 255, 100, 150);
        else if (dist < 800f)
            lineColor = Color.Lerp(new Color(100, 255, 100, 150), new Color(255, 255, 100, 150),
                (dist - 400f) / 400f);
        else
            lineColor = Color.Lerp(new Color(255, 255, 100, 150), new Color(255, 60, 60, 200),
                Math.Min((dist - 800f) / 800f, 1f));

        DrawLine(sb, playerScreen, bossScreen, lineColor, 2f);

        // Distance text
        DrawText(sb, $"{dist:F0}px", (playerScreen + bossScreen) / 2f + new Vector2(0, -12), lineColor);

        // Boss HP bar above boss
        DrawBossHPBar(sb, boss, bossScreen);
    }

    private void DrawBossHPBar(SpriteBatch sb, NPC boss, Vector2 bossScreen)
    {
        var texture = TextureAssets.MagicPixel.Value;
        float barWidth = 80f;
        float barHeight = 6f;
        float hpFrac = Math.Clamp((float)boss.life / Math.Max(boss.lifeMax, 1), 0f, 1f);

        Vector2 barPos = bossScreen + new Vector2(-barWidth / 2f, -50f);

        // Background
        sb.Draw(texture, barPos, new Rectangle(0, 0, 1, 1), new Color(0, 0, 0, 160),
            0f, Vector2.Zero, new Vector2(barWidth, barHeight), SpriteEffects.None, 0f);

        // HP fill
        Color hpColor = Color.Lerp(new Color(255, 60, 60), new Color(100, 255, 100), hpFrac);
        sb.Draw(texture, barPos + new Vector2(1, 1), new Rectangle(0, 0, 1, 1), hpColor,
            0f, Vector2.Zero, new Vector2((barWidth - 2) * hpFrac, barHeight - 2), SpriteEffects.None, 0f);

        // HP text
        DrawText(sb, $"{boss.life}/{boss.lifeMax}", barPos + new Vector2(barWidth / 2f, -14f),
            new Color(255, 255, 255, 200));
    }

    private void DrawAimLaser(SpriteBatch sb, Player player, Vector2 playerScreen, ActionPacket action)
    {
        float aimDist = BossMLConfig.Instance.AimDistance;
        float angleRad = action.AimSector * (MathF.PI * 2f / 16f);
        float dx = MathF.Cos(angleRad) * aimDist;
        float dy = MathF.Sin(angleRad) * aimDist;
        Vector2 aimEnd = playerScreen + new Vector2(dx, dy);

        // Pulsing alpha for the laser
        float pulse = 0.5f + 0.5f * MathF.Sin(Main.GameUpdateCount * 0.1f);
        Color laserColor = action.UseItem == 1
            ? new Color(255, 80, 80, (int)(120 + 80 * pulse))   // Red when attacking
            : new Color(80, 180, 255, (int)(60 + 40 * pulse));   // Blue when idle

        // Draw segmented laser
        int segments = 20;
        for (int i = 0; i < segments; i++)
        {
            float t0 = (float)i / segments;
            float t1 = (float)(i + 1) / segments;
            // Skip every other segment for dashed effect
            if (i % 2 == 1) continue;
            Vector2 p0 = Vector2.Lerp(playerScreen, aimEnd, t0);
            Vector2 p1 = Vector2.Lerp(playerScreen, aimEnd, t1);
            float width = action.UseItem == 1 ? 3f : 1.5f;
            DrawLine(sb, p0, p1, laserColor, width);
        }

        // Crosshair at aim point
        float crossSize = action.UseItem == 1 ? 8f : 5f;
        Vector2 aimTarget = playerScreen + new Vector2(dx * 0.3f, dy * 0.3f);
        DrawLine(sb, aimTarget + new Vector2(-crossSize, 0), aimTarget + new Vector2(crossSize, 0), laserColor, 1.5f);
        DrawLine(sb, aimTarget + new Vector2(0, -crossSize), aimTarget + new Vector2(0, crossSize), laserColor, 1.5f);
    }

    private void DrawActionIndicators(SpriteBatch sb, Vector2 playerScreen, ActionPacket action)
    {
        // Movement arrow
        float arrowY = playerScreen.Y - 50f;
        if (action.Move == 0) // left
            DrawText(sb, "◄", new Vector2(playerScreen.X - 20f, arrowY), new Color(100, 200, 255, 200));
        else if (action.Move == 2) // right
            DrawText(sb, "►", new Vector2(playerScreen.X + 12f, arrowY), new Color(100, 200, 255, 200));

        // Jump indicator
        if (action.Jump == 1)
            DrawText(sb, "▲", new Vector2(playerScreen.X - 4f, arrowY - 14f), new Color(100, 255, 150, 200));

        // Attack indicator
        if (action.UseItem == 1)
            DrawText(sb, "⚔", new Vector2(playerScreen.X + 22f, playerScreen.Y - 30f), new Color(255, 100, 100, 220));

        // Heal indicator
        if (action.QuickHeal == 1)
            DrawText(sb, "♥", new Vector2(playerScreen.X - 28f, playerScreen.Y - 30f), new Color(255, 100, 200, 220));

        // Hook indicator
        if (action.Hook == 1)
            DrawText(sb, "⚓", new Vector2(playerScreen.X + 22f, playerScreen.Y - 14f), new Color(200, 200, 100, 220));
    }

    private void DrawDamageFlash(SpriteBatch sb)
    {
        var texture = TextureAssets.MagicPixel.Value;
        float alpha = _damageFlash / 15f;
        int a = (int)(40 * alpha);
        // Red border flash
        int w = Main.screenWidth;
        int h = Main.screenHeight;
        int border = 4;
        var color = new Color(255, 0, 0, a);
        sb.Draw(texture, new Rectangle(0, 0, w, border), color);           // top
        sb.Draw(texture, new Rectangle(0, h - border, w, border), color);  // bottom
        sb.Draw(texture, new Rectangle(0, 0, border, h), color);           // left
        sb.Draw(texture, new Rectangle(w - border, 0, border, h), color);  // right
    }

    // ── Drawing primitives ────────────────────────────────────

    private static void DrawLine(SpriteBatch sb, Vector2 start, Vector2 end, Color color, float thickness)
    {
        var texture = TextureAssets.MagicPixel.Value;
        Vector2 diff = end - start;
        float length = diff.Length();
        if (length < 1f) return;
        float angle = MathF.Atan2(diff.Y, diff.X);
        sb.Draw(texture, start, new Rectangle(0, 0, 1, 1), color,
            angle, new Vector2(0f, 0.5f), new Vector2(length, thickness), SpriteEffects.None, 0f);
    }

    private static void DrawCircle(SpriteBatch sb, Vector2 center, float radius, Color color, float thickness)
    {
        var texture = TextureAssets.MagicPixel.Value;
        int segments = 48;
        for (int i = 0; i < segments; i++)
        {
            float a0 = i * MathF.PI * 2f / segments;
            float a1 = (i + 1) * MathF.PI * 2f / segments;
            Vector2 p0 = center + new Vector2(MathF.Cos(a0), MathF.Sin(a0)) * radius;
            Vector2 p1 = center + new Vector2(MathF.Cos(a1), MathF.Sin(a1)) * radius;
            DrawLine(sb, p0, p1, color, thickness);
        }
    }

    private static void DrawText(SpriteBatch sb, string text, Vector2 pos, Color color)
    {
        var font = FontAssets.MouseText.Value;
        Vector2 size = font.MeasureString(text);
        Utils.DrawBorderString(sb, text, pos - size / 2f, color);
    }
}
