using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Collects up to 30 non-boss hostile NPCs nearest to the local player.
    /// </summary>
    public static class EnemyObserver
    {
        private const int MaxEnemies = 30;

        public static List<EnemyData> Observe(Player player)
        {
            Vector2 playerCenter = player.Center;
            var candidates = new List<(float dist, int index)>(64);

            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active)  continue;
                if (npc.friendly) continue;
                if (npc.boss)     continue;
                if (npc.life <= 0) continue;
                // Exclude critters and town NPCs (they have no damage)
                if (npc.damage == 0) continue;

                float dist = Vector2.Distance(playerCenter, npc.Center);
                candidates.Add((dist, i));
            }

            candidates.Sort((a, b) => a.dist.CompareTo(b.dist));

            var result = new List<EnemyData>(Math.Min(candidates.Count, MaxEnemies));
            int limit  = Math.Min(candidates.Count, MaxEnemies);

            for (int k = 0; k < limit; k++)
            {
                var (dist, i) = candidates[k];
                NPC npc = Main.npc[i];

                result.Add(new EnemyData
                {
                    NpcIndex = i,
                    TypeId   = npc.type,
                    Position = new Vec2(npc.position.X, npc.position.Y),
                    Velocity = new Vec2(npc.velocity.X, npc.velocity.Y),
                    Hp       = npc.life,
                    MaxHp    = npc.lifeMax,
                    Hitbox   = new HitboxData
                    {
                        X      = npc.Hitbox.X,
                        Y      = npc.Hitbox.Y,
                        Width  = npc.Hitbox.Width,
                        Height = npc.Hitbox.Height
                    },
                    Damage   = npc.damage,
                    Distance = dist
                });
            }

            return result;
        }
    }
}
