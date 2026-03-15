using System;
using Microsoft.Xna.Framework;
using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Builds a 40x30 binary tile grid centered on the player.
    /// Each cell represents one 16-pixel tile: 1 = solid/active, 0 = air.
    /// The grid is rebuilt only every RebuildInterval ticks for performance.
    /// </summary>
    public class TileObserver
    {
        public const int GridWidth         = 40;
        public const int GridHeight        = 30;
        public const int RebuildInterval   = 10;  // ticks between full rebuilds

        private TileGridData _cached  = new();
        private int          _ticksSinceRebuild = RebuildInterval; // Force rebuild first call

        public TileGridData Observe(Player player, int currentTick)
        {
            _ticksSinceRebuild++;
            if (_ticksSinceRebuild < RebuildInterval)
                return _cached;

            _ticksSinceRebuild = 0;
            _cached = BuildGrid(player);
            return _cached;
        }

        private static TileGridData BuildGrid(Player player)
        {
            // Center tile coords
            int centerTileX = (int)(player.Center.X / 16f);
            int centerTileY = (int)(player.Center.Y / 16f);

            int halfW = GridWidth  / 2;
            int halfH = GridHeight / 2;

            int originX = centerTileX - halfW;
            int originY = centerTileY - halfH;

            byte[] grid = new byte[GridWidth * GridHeight];

            for (int row = 0; row < GridHeight; row++)
            {
                for (int col = 0; col < GridWidth; col++)
                {
                    int tileX = originX + col;
                    int tileY = originY + row;

                    bool solid = IsSolid(tileX, tileY);
                    grid[row * GridWidth + col] = solid ? (byte)1 : (byte)0;
                }
            }

            return new TileGridData
            {
                Width       = GridWidth,
                Height      = GridHeight,
                OriginTileX = originX,
                OriginTileY = originY,
                Grid        = grid
            };
        }

        private static bool IsSolid(int tileX, int tileY)
        {
            // Bounds check
            if (tileX < 0 || tileY < 0 || tileX >= Main.maxTilesX || tileY >= Main.maxTilesY)
                return true; // treat out-of-bounds as solid (wall)

            Tile tile = Main.tile[tileX, tileY];
            if (tile == null) return false;

            // HasTile: tile exists (not air) and is active
            return tile.HasTile && Main.tileSolid[tile.TileType];
        }
    }
}
