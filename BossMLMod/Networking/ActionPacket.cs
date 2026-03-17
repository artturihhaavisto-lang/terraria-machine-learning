#nullable enable
using System.Text.Json.Serialization;
using BossMLMod;

namespace BossMLMod.Networking;

/// <summary>
/// Deserialized action from the Python agent. Each field maps to a discrete action dimension.
///
/// Action space: MultiDiscrete([3, 2, 2, 16, 2, 2, 2])
///   move:       0=left, 1=none, 2=right
///   jump:       0=no, 1=yes
///   use_item:   0=no, 1=yes
///   aim_sector: 0-15 (22.5 degree increments from East, clockwise)
///   hook:       0=no, 1=yes
///   quick_heal: 0=no, 1=yes
///   dash:       0=no, 1=yes (double-tap in move direction; requires dash accessory)
///
/// Control field (used by curriculum system, not part of action space):
///   set_boss_type: if > 0, changes the boss NPC type for the next episode
/// </summary>
public class ActionPacket
{
    [JsonPropertyName("move")]
    public int Move { get; set; } = 1; // Default: no movement

    [JsonPropertyName("jump")]
    public int Jump { get; set; } = 0;

    [JsonPropertyName("use_item")]
    public int UseItem { get; set; } = 0;

    [JsonPropertyName("aim_sector")]
    public int AimSector { get; set; } = 0;

    [JsonPropertyName("hook")]
    public int Hook { get; set; } = 0;

    [JsonPropertyName("quick_heal")]
    public int QuickHeal { get; set; } = 0;

    [JsonPropertyName("dash")]
    public int Dash { get; set; } = 0;

    // Control field: non-zero triggers a boss type change (used by curriculum)
    [JsonPropertyName("set_boss_type")]
    public int SetBossType { get; set; } = 0;

    // Control field: if non-null, displayed as an in-game chat message
    [JsonPropertyName("message")]
    public string? Message { get; set; }

    // Control field: if true, the next state packet will include the full loadout
    [JsonPropertyName("request_loadout")]
    public bool RequestLoadout { get; set; }

    // Control field: if non-null, restores this loadout on the player
    [JsonPropertyName("set_loadout")]
    public LoadoutData? SetLoadout { get; set; }
}
