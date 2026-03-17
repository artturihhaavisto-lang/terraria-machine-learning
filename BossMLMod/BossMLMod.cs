#nullable enable
using Terraria.ModLoader;

namespace BossMLMod;

public class BossMLMod : Mod
{
    public static BossMLMod? Instance { get; private set; }

    public override void Load()
    {
        Instance = this;
        Logger.Info("BossMLMod loaded.");
    }

    public override void Unload()
    {
        Instance = null;
    }
}
