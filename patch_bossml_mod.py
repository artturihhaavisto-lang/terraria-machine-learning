#!/usr/bin/env python3
"""
patch_bossml_mod.py — add headless-automation features to BossMLMod's C# source.

Adds two ModConfig-driven features (both off by default, zero behavior change
until enabled):

  1. AutoEnableML   (bool)   — when true, ML mode arms itself ~3s after the
                               Python agent connects. Replaces typing '/bml on'
                               in chat — essential for headless instances.
  2. ForceTimeOfDay (string) — "none" | "day" | "noon" | "night" | "midnight".
                               Applied on every episode reset, so night-only
                               bosses (EoC, Skeletron, mech bosses) work
                               without '/bml time night'.

Both are plain ModConfig fields, so the fleet can set them per-instance via:
    fleet> automl on
    fleet> time night

Idempotent: safe to run multiple times. Usage:
    python3 patch_bossml_mod.py /path/to/BossMLMod
"""
import sys
import os

MARKER = "AutoEnableML"


def patch_file(path: str, edits: list[tuple[str, str, str]]) -> int:
    """Apply (anchor, insertion, mode) edits. mode: 'after' or 'before'."""
    with open(path) as f:
        src = f.read()
    applied = 0
    for anchor, insertion, mode in edits:
        if insertion.strip().splitlines()[0].strip() in src:
            continue  # already applied
        if anchor not in src:
            print(f"  ✘ anchor not found in {os.path.basename(path)}: {anchor[:60]!r}")
            continue
        if mode == "after":
            src = src.replace(anchor, anchor + insertion, 1)
        else:
            src = src.replace(anchor, insertion + anchor, 1)
        applied += 1
    with open(path, "w") as f:
        f.write(src)
    return applied


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    mod_dir = sys.argv[1]
    cfg_path = os.path.join(mod_dir, "BossMLConfig.cs")
    sys_path = os.path.join(mod_dir, "BossMLSystem.cs")
    for p in (cfg_path, sys_path):
        if not os.path.exists(p):
            sys.exit(f"✘ {p} not found — is this a BossMLMod source dir?")

    with open(cfg_path) as f:
        if MARKER in f.read():
            print("  ✔ mod already patched — nothing to do.")
            return

    n = 0

    # ── BossMLConfig.cs: two new config fields ───────────────────────────────
    n += patch_file(cfg_path, [(
        "    [Range(1, 10)]\n    [DefaultValue(4)]\n    public int FrameSkip { get; set; } = 4;",
        """

    [Header("Automation")]
    [DefaultValue(false)]
    public bool AutoEnableML { get; set; } = false;

    [DefaultValue("none")]
    public string ForceTimeOfDay { get; set; } = "none";
""",
        "after",
    )])

    # ── BossMLSystem.cs ──────────────────────────────────────────────────────
    n += patch_file(sys_path, [
        # field
        (
            "    private int _resetCountdown = -1;",
            "\n\n    // --- Auto-enable ML mode (headless fleets) ---\n"
            "    private int _autoEnableCountdown = -1;",
            "after",
        ),
        # arm the countdown on world load
        (
            "        MLModeActive = false;\n"
            "        _resetCountdown = -1;\n"
            "        _doneSent = false;",
            "\n        _autoEnableCountdown = config.AutoEnableML ? 180 : -1;",
            "after",
        ),
        # auto-enable once the agent is connected and the player exists
        (
            "        if (!MLModeActive || Server == null || !Server.IsConnected) return;",
            """        // Auto-enable ML mode once the Python agent connects (headless fleets)
        if (!MLModeActive && _autoEnableCountdown > 0 && Server != null && Server.IsConnected
            && Main.LocalPlayer != null && Main.LocalPlayer.active && !Main.LocalPlayer.dead)
        {
            if (--_autoEnableCountdown == 0)
            {
                MLModeActive = true;
                _doneSent = false;
                ResetEpisode();
                Main.NewText("[BML] ML mode AUTO-ENABLED (agent connected)", 100, 255, 100);
            }
        }

""",
            "before",
        ),
        # force time of day on every episode reset (anchor unique to ResetEpisode)
        (
            "        // Revive if dead — directly reset death state",
            """        // Force time of day if configured (night-only bosses on headless instances)
        switch ((config.ForceTimeOfDay ?? "none").ToLowerInvariant())
        {
            case "day":      SetTime(0, true);      break;
            case "noon":     SetTime(27000, true);  break;
            case "night":    SetTime(0, false);     break;
            case "midnight": SetTime(16200, false); break;
        }

""",
            "before",
        ),
    ])

    print(f"  ✔ applied {n} patch block(s) to BossMLMod "
          f"(AutoEnableML + ForceTimeOfDay).")
    print("  ⚠ Rebuild the mod once in tModLoader (Workshop → Develop Mods → Build)")


if __name__ == "__main__":
    main()
