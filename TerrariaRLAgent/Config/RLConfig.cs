using System.ComponentModel;
using Terraria.ModLoader.Config;

namespace TerrariaRLAgent.Config
{
    // In tML 1.4.4+ [Label] and [Tooltip] with plain strings are obsolete.
    // Labels are auto-generated from property names; tooltips go in localization files.
    public class RLConfig : ModConfig
    {
        public override ConfigScope Mode => ConfigScope.ServerSide;

        // ── Boss ─────────────────────────────────────────────────────────────
        [Header("BossSettings")]
        [DefaultValue(4)]
        [Range(1, 9999)]
        public int BossNPCType { get; set; } = 4;   // 4 = Eye of Cthulhu

        // ── Arena ─────────────────────────────────────────────────────────────
        [Header("ArenaSettings")]
        [DefaultValue(0)]
        public int ArenaCenterX { get; set; } = 0;   // 0 = map centre

        [DefaultValue(0)]
        public int ArenaCenterY { get; set; } = 0;   // 0 = surface default

        [DefaultValue(200)]
        [Range(20, 2000)]
        public int ArenaWidth { get; set; } = 200;

        [DefaultValue(150)]
        [Range(20, 1000)]
        public int ArenaHeight { get; set; } = 150;

        [DefaultValue(10)]
        [Range(4, 50)]
        public int PlatformInterval { get; set; } = 10;

        // ── Networking ────────────────────────────────────────────────────────
        [Header("Networking")]
        [DefaultValue(7777)]
        [Range(1024, 65535)]
        public int Port { get; set; } = 7777;

        // ── Episode ───────────────────────────────────────────────────────────
        [Header("EpisodeSettings")]
        [DefaultValue(18000)]
        [Range(600, 216000)]
        public int EpisodeTimeout { get; set; } = 18000;   // ticks (60/sec)

        [DefaultValue(true)]
        public bool AutoPause { get; set; } = true;

        [DefaultValue("default")]
        public string LoadoutPreset { get; set; } = "default";
    }
}
