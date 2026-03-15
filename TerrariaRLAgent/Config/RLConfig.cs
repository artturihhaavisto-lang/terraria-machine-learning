using System.ComponentModel;
using Terraria.ModLoader.Config;

namespace TerrariaRLAgent.Config
{
    public class RLConfig : ModConfig
    {
        public override ConfigScope Mode => ConfigScope.ServerSide;

        [Header("Boss Settings")]
        [Label("Boss NPC Type ID")]
        [Tooltip("The NPC type ID of the boss to summon. Default 4 = Eye of Cthulhu.")]
        [DefaultValue(4)]
        [Range(1, 9999)]
        public int BossNPCType { get; set; } = 4;

        [Header("Arena Settings")]
        [Label("Arena Center X (tile coordinate)")]
        [DefaultValue(0)]
        public int ArenaCenterX { get; set; } = 0;

        [Label("Arena Center Y (tile coordinate)")]
        [DefaultValue(0)]
        public int ArenaCenterY { get; set; } = 0;

        [Label("Arena Width (tiles)")]
        [DefaultValue(200)]
        [Range(20, 2000)]
        public int ArenaWidth { get; set; } = 200;

        [Label("Arena Height (tiles)")]
        [DefaultValue(150)]
        [Range(20, 1000)]
        public int ArenaHeight { get; set; } = 150;

        [Label("Platform Interval (tiles between platforms vertically)")]
        [DefaultValue(10)]
        [Range(4, 50)]
        public int PlatformInterval { get; set; } = 10;

        [Header("Networking")]
        [Label("TCP Port")]
        [Tooltip("Port the mod listens on for the Python RL agent.")]
        [DefaultValue(7777)]
        [Range(1024, 65535)]
        public int Port { get; set; } = 7777;

        [Header("Episode Settings")]
        [Label("Episode Timeout (ticks, 60 ticks/sec)")]
        [DefaultValue(18000)]
        [Range(600, 216000)]
        public int EpisodeTimeout { get; set; } = 18000;

        [Label("Auto Pause when agent disconnects")]
        [DefaultValue(true)]
        public bool AutoPause { get; set; } = true;

        [Label("Loadout Preset")]
        [Tooltip("Name of the predefined loadout preset to apply at episode start.")]
        [DefaultValue("default")]
        public string LoadoutPreset { get; set; } = "default";
    }
}
