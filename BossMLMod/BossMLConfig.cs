using System.ComponentModel;
using Terraria.ModLoader;
using Terraria.ModLoader.Config;

namespace BossMLMod;

public class BossMLConfig : ModConfig
{
    public override ConfigScope Mode => ConfigScope.ClientSide;

    public static BossMLConfig Instance => ModContent.GetInstance<BossMLConfig>();

    [Header("Networking")]
    [Range(1024, 65535)]
    [DefaultValue(7777)]
    public int Port { get; set; } = 7777;

    [Range(0, 5000)]
    [DefaultValue(100)]
    public int ActionTimeoutMs { get; set; } = 100;

    [Header("Observations")]
    [Range(5, 50)]
    [DefaultValue(20)]
    public int MaxProjectiles { get; set; } = 20;

    [Range(5, 50)]
    [DefaultValue(10)]
    public int MaxSegments { get; set; } = 10;

    [Range(500, 5000)]
    [DefaultValue(1500)]
    public int ProjectileScanRadius { get; set; } = 1500;

    [Header("EpisodeManagement")]
    [DefaultValue(true)]
    public bool AutoReset { get; set; } = true;

    [Range(0, 600)]
    [DefaultValue(120)]
    public int AutoResetDelayTicks { get; set; } = 120;

    [DefaultValue(0)]
    public int ArenaCenterX { get; set; } = 0;

    [DefaultValue(0)]
    public int ArenaCenterY { get; set; } = 0;

    [DefaultValue(50)]
    public int BossNpcType { get; set; } = 50;

    [Header("Arena")]
    [Range(1, 200)]
    [DefaultValue(10)]
    public int ArenaPlatforms { get; set; } = 10;

    [Header("Training")]
    [Range(100, 3000)]
    [DefaultValue(1000)]
    public int AimDistance { get; set; } = 1000;

    [Range(1, 10)]
    [DefaultValue(4)]
    public int FrameSkip { get; set; } = 4;

    [Header("Automation")]
    [DefaultValue(false)]
    public bool AutoEnableML { get; set; } = false;

    [DefaultValue("none")]
    public string ForceTimeOfDay { get; set; } = "none";

}
