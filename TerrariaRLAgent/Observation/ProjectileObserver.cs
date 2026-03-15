using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Collects up to 50 hostile projectiles nearest to the local player.
    /// </summary>
    public static class ProjectileObserver
    {
        private const int MaxProjectiles = 50;

        public static List<ProjectileData> Observe(Player player)
        {
            Vector2 playerCenter = player.Center;
            var candidates = new List<(float dist, int index)>(64);

            for (int i = 0; i < Main.maxProjectiles; i++)
            {
                Projectile proj = Main.projectile[i];
                if (!proj.active)  continue;
                if (!proj.hostile) continue;

                float dist = Vector2.Distance(playerCenter, proj.Center);
                candidates.Add((dist, i));
            }

            // Sort ascending by distance, cap at MaxProjectiles
            candidates.Sort((a, b) => a.dist.CompareTo(b.dist));

            var result = new List<ProjectileData>(Math.Min(candidates.Count, MaxProjectiles));
            int limit  = Math.Min(candidates.Count, MaxProjectiles);

            for (int k = 0; k < limit; k++)
            {
                var (dist, i) = candidates[k];
                Projectile proj = Main.projectile[i];

                result.Add(new ProjectileData
                {
                    ProjIndex = i,
                    TypeId    = proj.type,
                    Position  = new Vec2(proj.position.X, proj.position.Y),
                    Velocity  = new Vec2(proj.velocity.X, proj.velocity.Y),
                    Hitbox    = new HitboxData
                    {
                        X      = proj.Hitbox.X,
                        Y      = proj.Hitbox.Y,
                        Width  = proj.Hitbox.Width,
                        Height = proj.Hitbox.Height
                    },
                    Damage    = proj.damage,
                    TimeLeft  = proj.timeLeft,
                    Penetrate = proj.penetrate,
                    Distance  = dist
                });
            }

            return result;
        }
    }
}
