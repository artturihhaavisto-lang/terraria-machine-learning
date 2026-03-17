#nullable enable
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Terraria.ModLoader;

namespace BossMLMod.Networking;

/// <summary>
/// TCP server running on localhost. Accepts a single Python client connection.
///
/// Threading model:
///   - A background thread runs AcceptLoop to accept new connections.
///   - State/action exchange happens synchronously on the GAME THREAD
///     (called from ModSystem.PostUpdateEverything) to ensure perfect
///     tick synchronization for RL training.
///   - The game thread blocks up to ActionTimeoutMs waiting for the agent's response.
///
/// Protocol: Newline-delimited JSON (NDJSON).
///   Game → Python:  { ...state... }\n
///   Python → Game:  { ...action... }\n
/// </summary>
public class TCPServer : IDisposable
{
    private TcpListener? _listener;
    private TcpClient? _client;
    private StreamReader? _reader;
    private StreamWriter? _writer;
    private readonly object _connLock = new();
    private CancellationTokenSource _cts = new();
    private readonly int _port;

    public bool IsConnected
    {
        get
        {
            lock (_connLock)
            {
                return _client?.Connected == true;
            }
        }
    }

    public TCPServer(int port)
    {
        _port = port;
    }

    public void Start()
    {
        _cts = new CancellationTokenSource();
        _listener = new TcpListener(IPAddress.Loopback, _port);
        _listener.Start();
        Task.Factory.StartNew(AcceptLoop, TaskCreationOptions.LongRunning);
        ModContent.GetInstance<BossMLMod>()?.Logger.Info($"BossML TCP server listening on localhost:{_port}");
    }

    public void Stop()
    {
        _cts.Cancel();
        lock (_connLock)
        {
            _reader?.Dispose();
            _writer?.Dispose();
            _client?.Dispose();
            _client = null;
        }
        _listener?.Stop();
    }

    /// <summary>
    /// Background loop: accepts incoming connections. Only one client at a time.
    /// If a new client connects, the old one is dropped.
    /// </summary>
    private async Task AcceptLoop()
    {
        while (!_cts.IsCancellationRequested)
        {
            try
            {
                var newClient = await _listener!.AcceptTcpClientAsync();
                newClient.NoDelay = true; // Disable Nagle's algorithm for low latency
                newClient.ReceiveTimeout = 0; // We handle timeouts in Exchange()

                lock (_connLock)
                {
                    // Drop old connection
                    _reader?.Dispose();
                    _writer?.Dispose();
                    _client?.Dispose();

                    _client = newClient;
                    var stream = _client.GetStream();
                    // Use UTF8 WITHOUT BOM — BOM bytes would corrupt the first JSON line
                    var utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
                    _reader = new StreamReader(stream, utf8NoBom);
                    _writer = new StreamWriter(stream, utf8NoBom) { AutoFlush = true };
                }

                ModContent.GetInstance<BossMLMod>()?.Logger.Info("BossML: Python agent connected.");
            }
            catch (ObjectDisposedException)
            {
                break; // Server shutting down
            }
            catch (SocketException)
            {
                // Listener stopped
                break;
            }
            catch (Exception ex)
            {
                ModContent.GetInstance<BossMLMod>()?.Logger.Warn($"BossML accept error: {ex.Message}");
                await Task.Delay(500);
            }
        }
    }

    /// <summary>
    /// Synchronous state/action exchange. Called on the GAME THREAD.
    ///
    /// 1. Serialize and send the state packet as a JSON line.
    /// 2. Read one JSON line back (the action), with a timeout.
    ///
    /// If the read times out or fails, returns false and action is default.
    /// </summary>
    public bool Exchange(StatePacket state, out ActionPacket action, int timeoutMs)
    {
        action = new ActionPacket();

        StreamReader? reader;
        StreamWriter? writer;
        TcpClient? client;

        lock (_connLock)
        {
            reader = _reader;
            writer = _writer;
            client = _client;
        }

        if (client == null || !client.Connected || reader == null || writer == null)
            return false;

        try
        {
            // Send state as a single JSON line
            string stateJson = JsonSerializer.Serialize(state);
            writer.WriteLine(stateJson);

            // Read action with timeout
            // We use the underlying stream's ReadTimeout for this.
            var stream = client.GetStream();
            int oldTimeout = stream.ReadTimeout;
            stream.ReadTimeout = timeoutMs > 0 ? timeoutMs : Timeout.Infinite;

            try
            {
                string? line = reader.ReadLine();
                if (line == null)
                {
                    // Client disconnected (EOF)
                    HandleDisconnect("EOF received");
                    return false;
                }

                action = JsonSerializer.Deserialize<ActionPacket>(line) ?? new ActionPacket();
                return true;
            }
            finally
            {
                stream.ReadTimeout = oldTimeout;
            }
        }
        catch (IOException ex) when (ex.InnerException is SocketException { SocketErrorCode: SocketError.TimedOut })
        {
            // Timeout waiting for action — not an error, just use last action
            return false;
        }
        catch (Exception ex)
        {
            HandleDisconnect($"Exchange error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Send a raw JSON message (used for episode reset notifications, etc.).
    /// </summary>
    public void SendMessage(string json)
    {
        lock (_connLock)
        {
            if (_writer == null || _client == null || !_client.Connected) return;
            try
            {
                _writer.WriteLine(json);
            }
            catch
            {
                HandleDisconnect("SendMessage failed");
            }
        }
    }

    private void HandleDisconnect(string reason)
    {
        lock (_connLock)
        {
            ModContent.GetInstance<BossMLMod>()?.Logger.Info($"BossML: Python agent disconnected ({reason})");
            _reader?.Dispose();
            _writer?.Dispose();
            _client?.Dispose();
            _client = null;
            _reader = null;
            _writer = null;
        }
    }

    public void Dispose()
    {
        Stop();
        _cts.Dispose();
    }
}
