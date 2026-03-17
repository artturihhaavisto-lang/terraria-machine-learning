"""
Action space definition and mapping for the Terraria boss-fighting agent.

Action space: MultiDiscrete([3, 2, 2, 16, 2, 2, 2])

Dimensions:
  0: move         — 0=left, 1=none, 2=right
  1: jump         — 0=no, 1=yes
  2: use_item     — 0=no, 1=yes (attack with weapon in slot 0)
  3: aim_sector   — 0-15 (22.5° per sector, 0=East, clockwise)
  4: hook         — 0=no, 1=fire/retract grappling hook
  5: quick_heal   — 0=no, 1=use healing potion
  6: dash         — 0=no, 1=yes (double-tap dash in move direction)

Hotbar is hardcoded: slot 0 = weapon, slot 1 = healing potion.
The agent heals via quick_heal (which uses the best potion in inventory).
"""

import numpy as np
from gymnasium import spaces


def make_action_space(cfg: dict) -> spaces.MultiDiscrete:
    """Create the MultiDiscrete action space from config."""
    return spaces.MultiDiscrete([
        cfg.get("move_size", 3),
        cfg.get("jump_size", 2),
        cfg.get("use_item_size", 2),
        cfg.get("aim_sectors", 16),
        cfg.get("hook_size", 2),
        cfg.get("quick_heal_size", 2),
        cfg.get("dash_size", 2),
    ])


# Action dimension names (for logging / debugging)
ACTION_NAMES = ["move", "jump", "use_item", "aim_sector", "hook", "quick_heal", "dash"]


def numpy_to_action_dict(action: np.ndarray) -> dict[str, int]:
    """
    Convert a numpy action array (from the policy) to the JSON dict
    expected by the game's ActionPacket.

    Args:
        action: np.ndarray of shape (7,) with integer values.

    Returns:
        Dict with string keys matching ActionPacket field names.
    """
    return {
        "move": int(action[0]),
        "jump": int(action[1]),
        "use_item": int(action[2]),
        "aim_sector": int(action[3]),
        "hook": int(action[4]),
        "quick_heal": int(action[5]),
        "dash": int(action[6]),
    }


def noop_action() -> dict[str, int]:
    """Return a no-op action (stand still, do nothing)."""
    return {
        "move": 1,       # none
        "jump": 0,
        "use_item": 0,
        "aim_sector": 0,
        "hook": 0,
        "quick_heal": 0,
        "dash": 0,
    }


def noop_numpy() -> np.ndarray:
    """Return a no-op action as a numpy array."""
    return np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
