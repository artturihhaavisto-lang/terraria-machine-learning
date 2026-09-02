#!/usr/bin/env python3
"""
Dummy BossMLMod server — speaks the mod's NDJSON protocol with simulated
(fake) game data, so the whole fleet + trainer stack can be tested without
Terraria. One instance per port, like the real mod.

Simulated fight: a 'King Slime' hops toward the player. Attacking (use_item)
while close deals damage; touching the boss hurts the player. Episode ends
with boss_killed / player_died, then auto-resets after 30 ticks (done=True
states in between, exactly like the real mod's episode lifecycle).
"""
import argparse, json, random, socket


def make_state(sim, done=False):
    return {
        "tick": sim["tick"], "done": done, "boss_type": 50,
        "episode": sim["episode"], "step": sim["step"],
        "player": {
            "pos_x": sim["px"], "pos_y": sim["py"], "vel_x": sim["pvx"], "vel_y": 0.0,
            "hp": max(sim["php"], 0), "max_hp": 500, "mana": 200, "max_mana": 200,
            "grounded": True, "defense": 20, "flight_time": 0, "max_flight_time": 100,
            "potion_sickness": 0, "wings_available": False, "grappled": False,
            "direction": 1 if sim["pvx"] >= 0 else -1, "dash_cooldown": 0,
            "buffs": [], "item_cooldown": 0, "item_use_time": 20,
            "item_damage": 40, "selected_item": 0,
        },
        "bosses": [{
            "hp": max(sim["bhp"], 0), "max_hp": 2000,
            "pos_x": sim["bx"], "pos_y": sim["by"], "vel_x": sim["bvx"], "vel_y": 0.0,
            "segments": [],
        }],
        "projectiles": [],
        "environment": {"day_time": True, "time": 27000.0,
                        "world_width": 67200.0, "world_height": 19200.0},
        "reward_components": {
            "boss_hp_delta": sim["boss_hp_delta"],
            "player_hp_delta": sim["player_hp_delta"],
            "distance_to_boss": abs(sim["bx"] - sim["px"]),
            "boss_killed": sim["boss_killed"],
            "player_died": sim["player_died"],
            "boss_despawned": False,
        },
    }


def fresh_sim(episode):
    return {
        "tick": 0, "step": 0, "episode": episode,
        "px": 33600.0, "py": 9600.0, "pvx": 0.0, "php": 500,
        "bx": 33600.0 + random.choice([-1, 1]) * random.uniform(600, 1200),
        "by": 9600.0, "bvx": 0.0, "bhp": 2000,
        "boss_hp_delta": 0.0, "player_hp_delta": 0.0,
        "boss_killed": False, "player_died": False,
    }


def tick(sim, act):
    sim["tick"] += 1; sim["step"] += 1
    sim["boss_hp_delta"] = 0.0; sim["player_hp_delta"] = 0.0
    # player movement: move 0=left 1=none 2=right
    sim["pvx"] = {0: -6.0, 1: 0.0, 2: 6.0}.get(act.get("move", 1), 0.0)
    sim["px"] += sim["pvx"] * 16
    # boss hops toward player, slightly faster than the player can run
    sim["bvx"] = 7.0 if sim["bx"] < sim["px"] else -7.0
    sim["bx"] += sim["bvx"] * 16 + random.uniform(-32, 32)
    dist = abs(sim["bx"] - sim["px"])
    # attack lands if using item within range
    if act.get("use_item", 0) and dist < 900:
        dmg = random.randint(25, 45)
        sim["bhp"] -= dmg
        sim["boss_hp_delta"] = float(dmg)
    # contact damage
    if dist < 250 and random.random() < 0.35:
        dmg = random.randint(20, 40)
        sim["php"] -= dmg
        sim["player_hp_delta"] = float(-dmg)
    sim["boss_killed"] = sim["bhp"] <= 0
    sim["player_died"] = sim["php"] <= 0
    return sim["boss_killed"] or sim["player_died"]


def serve(port):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(4)
    print(f"[dummy-mod] listening on {port}", flush=True)
    episode = 0
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rwb")
        episode += 1
        sim, reset_cooldown = fresh_sim(episode), 0
        try:
            while True:
                line = f.readline()
                if not line:
                    break
                act = json.loads(line)
                if reset_cooldown > 0:          # between episodes: report done
                    reset_cooldown -= 1
                    if reset_cooldown == 0:
                        episode += 1
                        sim = fresh_sim(episode)
                    out = make_state(sim, done=reset_cooldown > 0)
                else:
                    ep_done = tick(sim, act)
                    out = make_state(sim, done=ep_done)
                    if ep_done:
                        reset_cooldown = 30      # mimic AutoResetDelayTicks
                f.write((json.dumps(out) + "\n").encode()); f.flush()
        except (ConnectionError, OSError):
            pass
        finally:
            f.close(); conn.close()
            print(f"[dummy-mod:{port}] client disconnected", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    serve(args.port)
