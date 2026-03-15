"""
Flask + SocketIO dashboard server for the Terraria RL training agent.

Usage (standalone / testing):
    python -m terraria_rl.dashboard.app

Usage (embedded in training loop):
    from terraria_rl.dashboard.app import DashboardServer
    from terraria_rl.dashboard.metrics_store import MetricsStore

    store = MetricsStore()
    server = DashboardServer(store, host="0.0.0.0", port=5000)
    server.start()          # non-blocking – runs in background thread
    # ... training loop ...
    store.add_episode(...)
    store.update_live_state(...)
    server.push_episode_update(ep_metrics)
    server.push_tick_update()
    server.push_training_update(update_metrics)
"""

import os
import threading
import time
import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit

from terraria_rl.dashboard.metrics_store import MetricsStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(store: MetricsStore) -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)
    app.config["SECRET_KEY"] = os.environ.get("DASHBOARD_SECRET", "terraria-rl-dev-secret")

    # Attach the store on the app so blueprints / socketio callbacks can reach it
    app.metrics_store = store  # type: ignore[attr-defined]

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    app.socketio = socketio  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # HTTP routes
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        return jsonify(store.get_summary())

    @app.route("/api/metrics")
    def api_metrics():
        n = int(request.args.get("n", 100))
        return jsonify(
            {
                "episodes": store.get_recent_episodes(n),
                "updates": store.get_recent_updates(n),
                "log": store.get_recent_log(50),
            }
        )

    @app.route("/api/config")
    def api_config():
        cfg = getattr(app, "_training_config", {})
        return jsonify(cfg)

    @app.route("/api/control/pause", methods=["POST"])
    def api_pause():
        store.set_training_state("paused")
        _invoke_callback(app, "on_pause")
        return jsonify({"ok": True, "state": "paused"})

    @app.route("/api/control/resume", methods=["POST"])
    def api_resume():
        store.set_training_state("training")
        _invoke_callback(app, "on_resume")
        return jsonify({"ok": True, "state": "training"})

    @app.route("/api/control/save", methods=["POST"])
    def api_save():
        store.log_event("checkpoint", "Manual checkpoint save requested via dashboard")
        _invoke_callback(app, "on_save")
        return jsonify({"ok": True})

    @app.route("/api/control/eval", methods=["POST"])
    def api_eval():
        store.set_training_state("eval")
        _invoke_callback(app, "on_eval")
        return jsonify({"ok": True, "state": "eval"})

    # ------------------------------------------------------------------
    # SocketIO events
    # ------------------------------------------------------------------

    @socketio.on("connect")
    def on_connect():
        logger.debug("Dashboard client connected")
        # Send a full snapshot so the client can populate all charts immediately
        emit("full_snapshot", _build_full_snapshot(store))

    @socketio.on("disconnect")
    def on_disconnect():
        logger.debug("Dashboard client disconnected")

    @socketio.on("request_snapshot")
    def on_request_snapshot():
        emit("full_snapshot", _build_full_snapshot(store))

    return app


# ---------------------------------------------------------------------------
# Helper: build a full data snapshot for initial page load
# ---------------------------------------------------------------------------

def _build_full_snapshot(store: MetricsStore) -> Dict[str, Any]:
    episodes = store.get_recent_episodes(500)
    updates = store.get_recent_updates(500)
    rolling_wr = store.get_rolling_win_rate(100)
    # Trim to last 500
    rolling_wr = rolling_wr[-500:]
    return {
        "summary": store.get_summary(),
        "episodes": episodes,
        "updates": updates,
        "rolling_win_rates": rolling_wr,
        "live": store.get_live_state(),
        "log": store.get_recent_log(100),
    }


def _invoke_callback(app: Flask, name: str) -> None:
    cb = getattr(app, "_callbacks", {}).get(name)
    if cb is not None:
        try:
            cb()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dashboard callback %s raised: %s", name, exc)


# ---------------------------------------------------------------------------
# DashboardServer – high-level wrapper for embedding in a training loop
# ---------------------------------------------------------------------------

class DashboardServer:
    """
    Wraps the Flask/SocketIO app and provides push helpers so the training
    loop can emit real-time events to connected browser clients.

    Example::

        server = DashboardServer(store, host="0.0.0.0", port=5000)
        server.set_config({"lr": 3e-4, "n_steps": 2048, ...})
        server.on_save = lambda: trainer.save_checkpoint()
        server.start()

        # In training loop:
        server.push_episode_update(metrics_dict)
        server.push_tick_update()
        server.push_training_update(update_dict)
    """

    def __init__(
        self,
        store: MetricsStore,
        host: str = "127.0.0.1",
        port: int = 5000,
        debug: bool = False,
    ):
        self.store = store
        self.host = host
        self.port = port
        self.debug = debug

        self._app = create_app(store)
        self._socketio: SocketIO = self._app.socketio  # type: ignore[attr-defined]
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Callback hooks – assign callables before calling start()
        self.on_pause: Optional[callable] = None
        self.on_resume: Optional[callable] = None
        self.on_save: Optional[callable] = None
        self.on_eval: Optional[callable] = None

        # Wire callbacks into app
        self._app._callbacks = {  # type: ignore[attr-defined]
            "on_pause": lambda: self.on_pause() if self.on_pause else None,
            "on_resume": lambda: self.on_resume() if self.on_resume else None,
            "on_save": lambda: self.on_save() if self.on_save else None,
            "on_eval": lambda: self.on_eval() if self.on_eval else None,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the web server in a daemon background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True, name="dashboard-server")
        self._thread.start()
        logger.info("Dashboard server started at http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running = False
        # SocketIO/Werkzeug doesn't have a clean public stop API in threaded mode;
        # since the thread is daemonic it will die when the main process exits.

    # ------------------------------------------------------------------
    # Push helpers (called from training loop / any thread)
    # ------------------------------------------------------------------

    def push_episode_update(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Emit 'episode_update' to all connected clients."""
        if metrics is None:
            recent = self.store.get_recent_episodes(1)
            metrics = recent[0] if recent else {}
        summary = self.store.get_summary()
        self._emit("episode_update", {"metrics": metrics, "summary": summary})

    def push_tick_update(self) -> None:
        """Emit 'tick_update' with the current live state (call ~1 Hz)."""
        self._emit("tick_update", self.store.get_live_state())

    def push_training_update(self, update: Optional[Dict[str, Any]] = None) -> None:
        """Emit 'training_update' with the latest PPO update metrics."""
        if update is None:
            recent = self.store.get_recent_updates(1)
            update = recent[0] if recent else {}
        self._emit("training_update", update)

    def push_log_event(self, event_type: str, message: str) -> None:
        """Log an event and push it to the dashboard log panel."""
        self.store.log_event(event_type, message)
        self._emit("log_event", {"type": event_type, "message": message, "timestamp": time.time()})

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def set_config(self, config: Dict[str, Any]) -> None:
        """Store the training config so /api/config can serve it."""
        self._app._training_config = config  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _serve(self) -> None:
        self._socketio.run(
            self._app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True,
        )

    def _emit(self, event: str, data: Any) -> None:
        try:
            self._socketio.emit(event, data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SocketIO emit '%s' failed: %s", event, exc)


# ---------------------------------------------------------------------------
# Standalone entry-point (demo / development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    import random

    logging.basicConfig(level=logging.INFO)

    store = MetricsStore()
    server = DashboardServer(store, host="0.0.0.0", port=5000, debug=False)
    server.set_config(
        {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "env": "TerrariaKingSlime-v0",
        }
    )
    server.start()

    print(f"Dashboard running at http://0.0.0.0:5000  — generating fake metrics...")

    # Fake tile grid (40 wide x 30 tall): bottom 5 rows are solid
    grid = [[0] * 40 for _ in range(30)]
    for row in range(25, 30):
        for col in range(40):
            grid[row][col] = 1

    update_step = 0
    for ep in range(1, 10_000):
        length = random.randint(200, 1800)
        boss_hp = max(0.0, random.gauss(0.4, 0.3))
        result = "win" if boss_hp == 0.0 else ("loss" if random.random() < 0.6 else "timeout")
        reward = (1.0 - boss_hp) * 100 + random.gauss(0, 5)

        store.add_episode(
            episode=ep,
            reward=reward,
            length=length,
            result=result,
            boss_hp_remaining=boss_hp,
            damage_dealt=random.uniform(200, 800),
            damage_taken=random.uniform(50, 400),
        )
        server.push_episode_update()

        # Simulate ticks within an episode
        for tick in range(0, length, 20):
            px = 20 + math.sin(tick * 0.05) * 15
            py = 22 + math.cos(tick * 0.03) * 2
            bx = 20 + math.cos(tick * 0.04) * 10
            by = 20
            store.update_live_state(
                player_hp=max(1, 500 - tick * 0.15),
                player_hp_max=500,
                boss_hp=max(0, 14_000 - tick * 7.0),
                boss_hp_max=14_000,
                current_action=random.randint(0, 11),
                player_x=px,
                player_y=py,
                boss_x=bx,
                boss_y=by,
                projectiles=[
                    {"x": px + random.uniform(-3, 3), "y": py - 2}
                    for _ in range(random.randint(0, 3))
                ],
                tile_grid=grid,
                episode=ep,
                timestep=ep * 1000 + tick,
                episode_reward=reward * (tick / length),
                training_state="training",
            )
            server.push_tick_update()
            time.sleep(0.01)

        # PPO update every 2 episodes
        if ep % 2 == 0:
            update_step += 1
            store.add_training_update(
                update_step=update_step,
                policy_loss=random.gauss(0.05, 0.02),
                value_loss=random.gauss(0.3, 0.1),
                entropy=random.gauss(1.2, 0.1),
                kl_divergence=random.gauss(0.01, 0.005),
                learning_rate=3e-4,
                timestep=ep * 1000,
            )
            server.push_training_update()

        time.sleep(0.05)
