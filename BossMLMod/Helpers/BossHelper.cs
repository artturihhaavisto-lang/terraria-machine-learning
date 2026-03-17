#nullable enable
using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Terraria;
using Terraria.ID;

namespace BossMLMod.Helpers;

/// <summary>
/// Utilities for detecting active bosses and their body segments.
///
/// Boss identification strategy:
///   1. Scan Main.npc[] for NPCs with npc.boss == true
///   2. For multi-part bosses (Eater of Worlds, Skeletron, etc.),
///      collect segments by checking npc.realLife or known segment type IDs
///
/// Known multi-segment boss type IDs:
///   Eater of Worlds: 13 (Head), 14 (Body), 15 (Tail)
///   Skeletron:       35 (Head, boss=true), 36 (Hand)
///   Wall of Flesh:   113 (Eye, boss=true), 114 (Mouth)
///   The Twins:       125 (Retinazer, boss=true), 126 (Spazmatism, boss=true)
///   The Destroyer:   134 (Head, boss=true), 135 (Body), 136 (Tail)
///   Skeletron Prime: 127 (Head, boss=true), 128 (Saw), 129 (Vice), 130 (Cannon), 131 (Laser)
///   Golem:           245 (Body, boss=true), 246 (Head), 247 (Fist L), 248 (Fist R)
///   Moon Lord:       396 (Core, boss=true), 397 (Head), 398 (Hand)
///   Brain of Cthulhu: 266 (Brain, boss=true), 267 (Creeper)
///   Plantera:        262 (boss=true), 263/264 (Hooks/Tentacles)
/// </summary>
public static class BossHelper
{
    // Maps a boss "master" NPC type to all its segment/part type IDs
    private static readonly Dictionary<int, HashSet<int>> BossSegmentTypes = new()
    {
        // Eater of Worlds: head is boss, body/tail are segments
        { NPCID.EaterofWorldsHead, new HashSet<int> { NPCID.EaterofWorldsBody, NPCID.EaterofWorldsTail } },
        // Skeletron: head is boss, hands are segments
        { NPCID.SkeletronHead, new HashSet<int> { NPCID.SkeletronHand } },
        // Wall of Flesh: eye is boss, mouth is segment
        { NPCID.WallofFlesh, new HashSet<int> { NPCID.WallofFleshEye } },
        // The Destroyer
        { NPCID.TheDestroyer, new HashSet<int> { NPCID.TheDestroyerBody, NPCID.TheDestroyerTail } },
        // Skeletron Prime
        { NPCID.SkeletronPrime, new HashSet<int> { NPCID.PrimeSaw, NPCID.PrimeVice, NPCID.PrimeCannon, NPCID.PrimeLaser } },
        // Golem
        { NPCID.Golem, new HashSet<int> { NPCID.GolemHead, NPCID.GolemFistLeft, NPCID.GolemFistRight } },
        // Brain of Cthulhu
        { NPCID.BrainofCthulhu, new HashSet<int> { NPCID.Creeper } },
        // Plantera
        { NPCID.Plantera, new HashSet<int> { NPCID.PlanterasHook, NPCID.PlanterasTentacle } },
    };

    // Reverse lookup: segment type → master type
    private static readonly Dictionary<int, int> SegmentToMaster;

    static BossHelper()
    {
        SegmentToMaster = new Dictionary<int, int>();
        foreach (var (masterType, segTypes) in BossSegmentTypes)
        {
            foreach (int segType in segTypes)
                SegmentToMaster[segType] = masterType;
        }
    }

    /// <summary>
    /// Get all active boss NPCs. Returns only "master" NPCs (the one with boss=true).
    /// For multi-part bosses like Eater of Worlds, returns the head segment.
    /// </summary>
    public static List<NPC> GetActiveBosses()
    {
        var bosses = new List<NPC>();
        var seen = new HashSet<int>(); // track whoAmI to avoid duplicates

        for (int i = 0; i < Main.maxNPCs; i++)
        {
            var npc = Main.npc[i];
            if (!npc.active) continue;

            // Standard boss flag check
            if (npc.boss && !seen.Contains(npc.whoAmI))
            {
                bosses.Add(npc);
                seen.Add(npc.whoAmI);
                continue;
            }

            // For worm bosses (Eater of Worlds, Destroyer): the head has boss=true,
            // but we also catch segments via realLife pointing to the head.
            if (npc.realLife >= 0 && !seen.Contains(npc.realLife))
            {
                var master = Main.npc[npc.realLife];
                if (master.active && master.boss)
                {
                    bosses.Add(master);
                    seen.Add(master.whoAmI);
                }
            }
        }

        return bosses;
    }

    /// <summary>
    /// Get body segments for a multi-segment boss, sorted by distance to player.
    /// Returns up to maxSegments NPCs.
    /// </summary>
    public static List<NPC> GetBossSegments(NPC master, int maxSegments, Player player)
    {
        // Check if this boss type has known segments
        if (!BossSegmentTypes.TryGetValue(master.type, out var segTypes))
            return new List<NPC>();

        var segments = new List<(NPC npc, float dist)>();

        for (int i = 0; i < Main.maxNPCs; i++)
        {
            var npc = Main.npc[i];
            if (!npc.active) continue;
            if (npc.whoAmI == master.whoAmI) continue; // Skip the master itself

            // Check if this NPC is a segment of the master boss.
            // Method 1: npc.realLife points to the master
            bool isSegment = npc.realLife == master.whoAmI;
            // Method 2: NPC type is in the known segment types for this boss
            if (!isSegment)
                isSegment = segTypes.Contains(npc.type);

            if (isSegment)
            {
                float dist = Vector2.Distance(npc.Center, player.Center);
                segments.Add((npc, dist));
            }
        }

        // Sort by distance to player (closest first — most relevant for dodging)
        segments.Sort((a, b) => a.dist.CompareTo(b.dist));

        var result = new List<NPC>();
        int count = Math.Min(segments.Count, maxSegments);
        for (int i = 0; i < count; i++)
            result.Add(segments[i].npc);

        return result;
    }

    /// <summary>
    /// Find the closest active boss to the player. Used for distance-to-boss reward.
    /// </summary>
    public static NPC? GetClosestBoss(Player player)
    {
        NPC? closest = null;
        float minDist = float.MaxValue;

        foreach (var boss in GetActiveBosses())
        {
            float dist = Vector2.Distance(boss.Center, player.Center);
            if (dist < minDist)
            {
                minDist = dist;
                closest = boss;
            }
        }

        return closest;
    }

    /// <summary>
    /// Check if any boss is currently active in the world.
    /// </summary>
    public static bool AnyBossActive()
    {
        for (int i = 0; i < Main.maxNPCs; i++)
        {
            if (Main.npc[i].active && Main.npc[i].boss)
                return true;
        }
        return false;
    }
}
