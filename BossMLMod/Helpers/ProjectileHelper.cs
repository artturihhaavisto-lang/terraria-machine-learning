using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Terraria;

namespace BossMLMod.Helpers;

/// <summary>
/// Scans for hostile projectiles near the player and returns them sorted by threat level.
///
/// Threat heuristic: damage / (distance² + 1)
///   - High-damage projectiles close to the player rank highest
///   - The +1 prevents division by zero at point-blank range
///   - This naturally prioritizes immediate threats over distant ones
///
/// Why cap at 20 projectiles (configurable):
///   - Many boss fights produce 50-100+ projectiles (e.g., Plantera's seeds, Moon Lord's beams)
///   - Including all projectiles would create a variable-size observation (bad for fixed-size NN input)
///   - The 20 most threatening projectiles capture >95% of relevant dodging information
///   - Less threatening projectiles (far away, low damage) contribute minimal signal
/// </summary>
public static class ProjectileHelper
{
    /// <summary>
    /// Get the N most threatening hostile projectiles near the player.
    /// </summary>
    /// <param name="playerCenter">Player's center position in world coords</param>
    /// <param name="scanRadius">Maximum distance in world pixels to scan</param>
    /// <param name="maxCount">Maximum number of projectiles to return</param>
    /// <returns>List of hostile Projectiles sorted by threat (highest first)</returns>
    public static List<Projectile> GetNearbyHostile(Vector2 playerCenter, float scanRadius, int maxCount)
    {
        float scanRadiusSq = scanRadius * scanRadius;
        var candidates = new List<(Projectile proj, float threat)>();

        for (int i = 0; i < Main.maxProjectiles; i++)
        {
            var proj = Main.projectile[i];
            if (!proj.active) continue;
            if (!proj.hostile) continue;     // Only enemy projectiles
            if (proj.friendly) continue;     // Skip player-friendly projectiles
            if (proj.damage <= 0) continue;  // Skip zero-damage visual effects

            float dx = proj.Center.X - playerCenter.X;
            float dy = proj.Center.Y - playerCenter.Y;
            float distSq = dx * dx + dy * dy;

            if (distSq > scanRadiusSq) continue;

            // Threat = damage / (distance² + 1)
            float threat = proj.damage / (distSq + 1f);
            candidates.Add((proj, threat));
        }

        // Sort by threat descending
        candidates.Sort((a, b) => b.threat.CompareTo(a.threat));

        var result = new List<Projectile>();
        int count = Math.Min(candidates.Count, maxCount);
        for (int i = 0; i < count; i++)
            result.Add(candidates[i].proj);

        return result;
    }

    /// <summary>
    /// Count the player's own active projectiles (friendly, non-hostile).
    /// Useful for the observation to know how many attacks are in flight.
    /// </summary>
    public static int CountPlayerProjectiles(int playerIndex)
    {
        int count = 0;
        for (int i = 0; i < Main.maxProjectiles; i++)
        {
            var proj = Main.projectile[i];
            if (proj.active && proj.friendly && proj.owner == playerIndex)
                count++;
        }
        return count;
    }
}
