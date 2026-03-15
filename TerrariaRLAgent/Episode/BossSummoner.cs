using System;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;
using Terraria.ModLoader;

namespace TerrariaRLAgent.Episode
{
    /// <summary>
    /// Handles spawning the target boss NPC at the start of each episode.
    /// Supports both vanilla and modded boss types.
    /// </summary>
    public static class BossSummoner
    {
        /// <summary>
        /// Summons the boss defined by <paramref name="npcType"/> near the player.
        /// Returns true if at least one NPC was successfully spawned.
        /// </summary>
        public static bool SummonBoss(Player player, int npcType)
        {
            if (npcType <= 0) return false;

            try
            {
                if (IsVanillaBossWithSpecialSpawn(npcType))
                    return SummonVanillaSpecial(player, npcType);

                // Generic path: NPC.SpawnOnPlayer returns void in tML 1.4.4.
                NPC.SpawnOnPlayer(player.whoAmI, npcType);
                return true;
            }
            catch (Exception ex)
            {
                ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                    .Error($"[BossSummoner] Failed to spawn NPC type {npcType}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Spawns the boss at an explicit world-space position instead of on the player.
        /// </summary>
        public static bool SummonBossAt(int npcType, Vector2 worldPosition)
        {
            if (npcType <= 0) return false;
            try
            {
                // NPC.NewNPC first arg is IEntitySource; use the local player as source.
                int index = NPC.NewNPC(
                    Main.player[Main.myPlayer].GetSource_FromAI(),
                    (int)worldPosition.X,
                    (int)worldPosition.Y,
                    npcType
                );
                return index >= 0 && index < Main.maxNPCs;
            }
            catch (Exception ex)
            {
                ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                    .Error($"[BossSummoner] SummonBossAt failed for type {npcType}: {ex.Message}");
                return false;
            }
        }

        // -----------------------------------------------------------------------
        // Vanilla boss special-case logic
        // -----------------------------------------------------------------------

        private static bool IsVanillaBossWithSpecialSpawn(int type)
        {
            return type == NPCID.EyeofCthulhu     ||
                   type == NPCID.KingSlime         ||
                   type == NPCID.EaterofWorldsHead ||
                   type == NPCID.BrainofCthulhu    ||
                   type == NPCID.QueenBee           ||
                   type == NPCID.SkeletronHead      ||
                   type == NPCID.WallofFlesh        ||
                   type == NPCID.Retinazer          ||
                   type == NPCID.Spazmatism         ||
                   type == NPCID.SkeletronPrime      ||
                   type == NPCID.TheDestroyer        ||
                   type == NPCID.Plantera           ||
                   type == NPCID.Golem              ||
                   type == NPCID.DukeFishron         ||
                   type == NPCID.HallowBoss         ||  // Empress of Light
                   type == NPCID.CultistBoss        ||
                   type == NPCID.MoonLordCore;
        }

        private static bool SummonVanillaSpecial(Player player, int type)
        {
            // Wall of Flesh must be placed explicitly in the underworld.
            if (type == NPCID.WallofFlesh)
            {
                int wofX = (int)player.Center.X;
                int wofY = Main.maxTilesY * 16 - 200;
                int index = NPC.NewNPC(
                    player.GetSource_FromAI(),
                    wofX, wofY, NPCID.WallofFlesh
                );
                return index >= 0;
            }

            // All other vanilla bosses: SpawnOnPlayer handles them (returns void in 1.4.4).
            NPC.SpawnOnPlayer(player.whoAmI, type);
            return true;
        }
    }
}
