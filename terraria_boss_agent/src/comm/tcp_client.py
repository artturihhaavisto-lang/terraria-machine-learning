"""
TCP client for communicating with the BossMLMod TCP server.

Protocol: Newline-delimited JSON (NDJSON) over TCP localhost.
  - Game sends state JSON line → client reads it
  - Client sends action JSON line → game reads it

The client connects to the game's TCP server and maintains a persistent
connection for the duration of training. If the connection drops, it
automatically retries.
"""

import json
import socket
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TerrariaClient:
    """TCP client that exchanges JSON state/action packets with the game."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7777,
                 connect_timeout: float = 60.0, recv_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self._sock: socket.socket | None = None
        self._buffer = b""

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        """Connect to the game's TCP server. Retries every 2s until timeout."""
        start = time.monotonic()
        while time.monotonic() - start < self.connect_timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.recv_timeout)
                sock.connect((self.host, self.port))
                self._sock = sock
                self._buffer = b""
                logger.info(f"Connected to Terraria at {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.debug(f"Connection attempt failed: {e}. Retrying in 2s...")
                sock.close()
                time.sleep(2.0)

        raise ConnectionError(
            f"Could not connect to Terraria at {self.host}:{self.port} "
            f"within {self.connect_timeout}s. Is the game running with BossMLMod?"
        )

    def close(self) -> None:
        """Close the connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._buffer = b""
            logger.info("Disconnected from Terraria.")

    def recv_state(self) -> dict[str, Any]:
        """
        Read one JSON line from the server (game state).

        Blocks until a complete line is received or recv_timeout is hit.

        Returns:
            Parsed state dictionary.

        Raises:
            ConnectionError: If the connection is lost.
            TimeoutError: If no data is received within recv_timeout.
        """
        if self._sock is None:
            raise ConnectionError("Not connected.")

        while True:
            # Check if we already have a complete line in the buffer
            newline_idx = self._buffer.find(b"\n")
            if newline_idx >= 0:
                line = self._buffer[:newline_idx]
                self._buffer = self._buffer[newline_idx + 1:]
                return json.loads(line.decode("utf-8"))

            # Read more data
            try:
                data = self._sock.recv(65536)
            except socket.timeout:
                raise TimeoutError("Timed out waiting for state from game.")
            except OSError as e:
                self.close()
                raise ConnectionError(f"Connection lost: {e}")

            if not data:
                self.close()
                raise ConnectionError("Connection closed by game (EOF).")

            self._buffer += data

    def send_action(self, action: dict[str, int]) -> None:
        """
        Send an action as a JSON line to the server.

        Args:
            action: Dictionary with keys matching ActionPacket fields.

        Raises:
            ConnectionError: If the connection is lost.
        """
        if self._sock is None:
            raise ConnectionError("Not connected.")

        try:
            line = json.dumps(action, separators=(",", ":")) + "\n"
            self._sock.sendall(line.encode("utf-8"))
        except OSError as e:
            self.close()
            raise ConnectionError(f"Failed to send action: {e}")

    def exchange(self, action: dict[str, int]) -> dict[str, Any]:
        """
        Send action, then receive next state. This is the primary per-step call.

        The ordering is: send action → receive state. This matches the game's
        flow: PostUpdateEverything sends state, then blocks reading our action.
        So from the client's perspective, we first send the action the game is
        waiting for, then read the new state it sends after processing.

        For the very first step (no action yet), send a no-op action.

        Returns:
            Parsed state dictionary.
        """
        self.send_action(action)
        return self.recv_state()

    def initial_state(self) -> dict[str, Any]:
        """
        Receive the initial state without sending an action first.
        Used at the start of a connection when the game sends the first state
        before expecting an action.
        """
        return self.recv_state()
