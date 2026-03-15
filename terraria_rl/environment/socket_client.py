import socket
import json
import threading
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SocketClient:
    """Thread-safe TCP client for communicating with the tModLoader Terraria mod.

    Sends and receives newline-delimited JSON messages. Includes reconnection
    logic and configurable timeouts.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7777,
        timeout: float = 10.0,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 10,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._sock: Optional[socket.socket] = None
        self._recv_buffer = ""
        self._lock = threading.Lock()
        self._connected = False

    def connect(self) -> None:
        """Establish TCP connection to the Terraria mod server."""
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                with self._lock:
                    self._sock = sock
                    self._recv_buffer = ""
                    self._connected = True
                logger.info(f"Connected to Terraria mod at {self.host}:{self.port}")
                return
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                attempts += 1
                logger.warning(
                    f"Connection attempt {attempts}/{self.max_reconnect_attempts} failed: {e}. "
                    f"Retrying in {self.reconnect_delay}s..."
                )
                time.sleep(self.reconnect_delay)

        raise ConnectionError(
            f"Failed to connect to {self.host}:{self.port} after "
            f"{self.max_reconnect_attempts} attempts."
        )

    def disconnect(self) -> None:
        """Close the TCP connection."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            self._connected = False
            self._recv_buffer = ""
        logger.info("Disconnected from Terraria mod.")

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def send(self, data: Dict[str, Any]) -> None:
        """Serialize data as JSON and send over the socket (newline-delimited).

        Args:
            data: Dictionary to serialize and send.

        Raises:
            ConnectionError: If the socket is not connected or send fails.
        """
        message = json.dumps(data) + "\n"
        encoded = message.encode("utf-8")
        with self._lock:
            if not self._connected or self._sock is None:
                raise ConnectionError("Not connected to Terraria mod.")
            try:
                self._sock.sendall(encoded)
            except (BrokenPipeError, OSError) as e:
                self._connected = False
                raise ConnectionError(f"Send failed: {e}") from e

    def receive(self) -> Dict[str, Any]:
        """Receive a newline-delimited JSON message from the socket.

        Blocks until a complete message is available or timeout occurs.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ConnectionError: If the socket is disconnected or an error occurs.
            TimeoutError: If no complete message is received within timeout.
        """
        with self._lock:
            if not self._connected or self._sock is None:
                raise ConnectionError("Not connected to Terraria mod.")
            sock = self._sock

        while True:
            # Check if a complete message is already in the buffer (lock-free read).
            newline_idx = self._recv_buffer.find("\n")
            if newline_idx != -1:
                line = self._recv_buffer[:newline_idx]
                self._recv_buffer = self._recv_buffer[newline_idx + 1:]
                try:
                    return json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON message: {e!r}. Raw: {line!r}") from e

            # Read more data from the socket.
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    with self._lock:
                        self._connected = False
                    raise ConnectionError("Connection closed by remote host.")
                self._recv_buffer += chunk.decode("utf-8")
            except socket.timeout as e:
                raise TimeoutError("Timed out waiting for message from Terraria mod.") from e
            except OSError as e:
                with self._lock:
                    self._connected = False
                raise ConnectionError(f"Receive failed: {e}") from e

    def send_and_receive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON message and wait for a response.

        Args:
            data: Dictionary to send.

        Returns:
            Response dictionary.
        """
        self.send(data)
        return self.receive()

    def reconnect(self) -> None:
        """Disconnect and reconnect to the mod server."""
        logger.info("Attempting reconnection...")
        self.disconnect()
        time.sleep(self.reconnect_delay)
        self.connect()
