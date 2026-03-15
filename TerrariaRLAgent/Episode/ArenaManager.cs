using System;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;
using Terraria.ModLoader;
using TerrariaRLAgent.Config;

namespace TerrariaRLAgent.Episode
{
    /// <summary>
    /// Manages the RL training arena:
    ///   • Teleports the player to the arena centre.
    ///   • Generates a large open arena with horizontal platforms at regular vertical intervals.
    ///   • Clears all tiles inside the arena bounding box.
    /// </summary>
    public static class ArenaManager
    {
        // Tile type used for platforms (wood platform = 19)
        private const ushort PlatformTileType = TileID.WoodPlatform;

        // -----------------------------------------------------------------------
        // Public API
        // -----------------------------------------------------------------------

        /// <summary>
        /// Teleports the player to the arena centre (world pixels).
        /// The arena centre is taken from <see cref="RLConfig"/>; if it is (0,0) we
        /// default to the map centre near surface.
        /// </summary>
        public static void TeleportPlayerToArena(Player player)
        {
            Vector2 centre = GetArenaCentreWorld();
            // Place player so his feet land on the centre tile
            player.position = new Vector2(
                centre.X - player.width  / 2f,
                centre.Y - player.height
            );
            // Kill velocity so the player doesn't drift
            player.velocity = Vector2.Zero;
        }

        /// <summary>
        /// Builds the arena:
        ///   1. Carves out a rectangular void (removes all tiles).
        ///   2. Places a solid floor.
        ///   3. Places horizontal platforms every <see cref="RLConfig.PlatformInterval"/> tiles.
        /// </summary>
        public static void GenerateArena()
        {
            var cfg = ModContent.GetInstance<RLConfig>();
            if (cfg == null) return;

            int centerTileX = GetCenterTileX(cfg);
            int centerTileY = GetCenterTileY(cfg);
            int halfW = cfg.ArenaWidth  / 2;
            int halfH = cfg.ArenaHeight / 2;

            int left   = Math.Max(0, centerTileX - halfW);
            int right  = Math.Min(Main.maxTilesX - 1, centerTileX + halfW);
            int top    = Math.Max(0, centerTileY - halfH);
            int bottom = Math.Min(Main.maxTilesY - 1, centerTileY + halfH);

            // Step 1: carve interior void
            ClearRect(left, top, right, bottom);

            // Step 2: solid floor at bottom row
            PlaceSolidRow(left, right, bottom, TileID.Stone);

            // Step 3: platforms at regular vertical intervals (top to bottom)
            int interval = Math.Max(4, cfg.PlatformInterval);
            for (int tileY = bottom - interval; tileY > top; tileY -= interval)
            {
                PlacePlatformRow(left, right, tileY);
            }

            ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                .Info($"[ArenaManager] Arena generated: ({left},{top}) -> ({right},{bottom}), " +
                      $"platform interval {interval} tiles.");
        }

        // -----------------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------------

        private static Vector2 GetArenaCentreWorld()
        {
            var cfg = ModContent.GetInstance<RLConfig>();
            int tx = GetCenterTileX(cfg);
            int ty = GetCenterTileY(cfg);
            return new Vector2(tx * 16f, ty * 16f);
        }

        private static int GetCenterTileX(RLConfig? cfg)
        {
            int tx = cfg?.ArenaCenterX ?? 0;
            if (tx <= 0) tx = Main.maxTilesX / 2;
            return tx;
        }

        private static int GetCenterTileY(RLConfig? cfg)
        {
            int ty = cfg?.ArenaCenterY ?? 0;
            if (ty <= 0)
            {
                // Default: surface layer (roughly 1/4 down)
                ty = (int)(Main.worldSurface * 0.5);
            }
            return ty;
        }

        /// <summary>Removes all tiles and walls inside the rectangle (inclusive).</summary>
        private static void ClearRect(int left, int top, int right, int bottom)
        {
            for (int x = left; x <= right; x++)
            {
                for (int y = top; y <= bottom; y++)
                {
                    Tile tile = Main.tile[x, y];
                    if (tile == null) continue;
                    tile.ClearTile();
                    tile.WallType = 0;
                }
            }

            // Notify the renderer that tiles have changed in this region.
            // SendTileSquare's size parameter is capped at 255; send multiple
            // squares to cover the full arena.
            int rectW = right  - left + 1;
            int rectH = bottom - top  + 1;
            int chunkSize = 200;
            for (int cx = left; cx <= right; cx += chunkSize)
            {
                for (int cy = top; cy <= bottom; cy += chunkSize)
                {
                    int cw = System.Math.Min(chunkSize, right  - cx + 1);
                    int ch = System.Math.Min(chunkSize, bottom - cy + 1);
                    NetMessage.SendTileSquare(-1, cx + cw / 2, cy + ch / 2, cw, ch);
                }
            }
        }

        /// <summary>Places a solid stone floor row at tileY.</summary>
        private static void PlaceSolidRow(int left, int right, int tileY, int tileType)
        {
            for (int x = left; x <= right; x++)
            {
                WorldGen.PlaceTile(x, tileY, tileType, mute: true, forced: true);
            }
        }

        /// <summary>Places a platform row at tileY.</summary>
        private static void PlacePlatformRow(int left, int right, int tileY)
        {
            for (int x = left; x <= right; x++)
            {
                Tile tile = Main.tile[x, tileY];
                if (tile == null) continue;

                tile.HasTile  = true;
                tile.TileType = PlatformTileType;

                // Slope: none (flat platform)
                tile.Slope    = Terraria.Enums.SlopeType.Solid;
                tile.IsHalfBlock = false;
            }
        }
    }
}
