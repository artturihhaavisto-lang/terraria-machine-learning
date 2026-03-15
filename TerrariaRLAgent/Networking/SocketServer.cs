using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Terraria.ModLoader;
using TerrariaRLAgent.Control;

namespace TerrariaRLAgent.Networking
{
    /// <summary>
    /// Async TCP server that accepts a single Python ML client.
    /// Observations are sent from the game (main) thread via a concurrent queue;
    /// a background Task drains the queue and writes to the socket.
    /// Actions arrive on the background read Task and are deposited into ActionBuffer.
    /// All socket I/O is fully async so the game thread never blocks.
    /// </summary>
    public class SocketServer
    {
        private readonly int _port;
        private readonly ActionBuffer _actionBuffer;

        private TcpListener? _listener;
        private TcpClient? _client;
        private NetworkStream? _stream;
        private StreamReader? _reader;
        private StreamWriter? _writer;

        private CancellationTokenSource? _cts;
        private Task? _acceptTask;
        private Task? _readTask;
        private Task? _writeTask;

        // Observations from game thread go into this queue; the write task drains it.
        private readonly ConcurrentQueue<string> _sendQueue = new();
        // Signals the write task that new data is available.
        private readonly SemaphoreSlim _sendSignal = new(0, int.MaxValue);

        private volatile bool _isClientConnected;
        public bool IsClientConnected => _isClientConnected;

        // "ready" handshake flag – set by the read loop, consumed by EpisodeManager
        private volatile bool _pendingReady;
        public bool HasPendingReady => _pendingReady;
        public void ConsumePendingReady() => _pendingReady = false;

        public SocketServer(int port, ActionBuffer actionBuffer)
        {
            _port = port;
            _actionBuffer = actionBuffer;
        }

        /// <summary>Starts listening for incoming connections.</summary>
        public void Start()
        {
            _cts = new CancellationTokenSource();
            _listener = new TcpListener(IPAddress.Any, _port);
            _listener.Start();

            var token = _cts.Token;
            _acceptTask = Task.Run(() => AcceptLoopAsync(token), token);

            ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                .Info($"[SocketServer] Listening on port {_port}.");
        }

        /// <summary>Stops all background tasks and closes the socket.</summary>
        public void Stop()
        {
            _cts?.Cancel();
            DisconnectClient();
            try { _listener?.Stop(); } catch { /* ignore */ }
            _listener = null;

            // Wait briefly for tasks to finish; ignore exceptions.
            try { Task.WhenAll(
                    _acceptTask ?? Task.CompletedTask,
                    _readTask  ?? Task.CompletedTask,
                    _writeTask ?? Task.CompletedTask)
                .Wait(TimeSpan.FromSeconds(2)); }
            catch { /* ignore */ }

            _cts?.Dispose();
            _cts = null;
        }

        /// <summary>
        /// Called from the game (main) thread.
        /// Serializes the observation and pushes it onto the send queue; never blocks.
        /// </summary>
        public void EnqueueObservation(Observation.ObservationData observation)
        {
            if (!_isClientConnected) return;

            string json = MessageSerializer.SerializeObservation(observation);
            _sendQueue.Enqueue(json + "\n");
            _sendSignal.Release();
        }

        /// <summary>
        /// Sends a raw JSON line (e.g. episode_end messages) from the game thread.
        /// </summary>
        public void EnqueueRaw(string jsonLine)
        {
            if (!_isClientConnected) return;
            _sendQueue.Enqueue(jsonLine.TrimEnd('\n') + "\n");
            _sendSignal.Release();
        }

        // -----------------------------------------------------------------------
        // Background tasks
        // -----------------------------------------------------------------------

        private async Task AcceptLoopAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var client = await _listener!.AcceptTcpClientAsync(ct);
                    // Only allow one client at a time.
                    DisconnectClient();

                    _client = client;
                    _client.NoDelay = true;
                    _stream = _client.GetStream();
                    _reader = new StreamReader(_stream, Encoding.UTF8, leaveOpen: true);
                    _writer = new StreamWriter(_stream, Encoding.UTF8, leaveOpen: true)
                    {
                        AutoFlush = false,
                        NewLine = "\n"
                    };

                    _isClientConnected = true;

                    ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                        .Info("[SocketServer] Client connected.");

                    // Drain any stale queued messages from a previous episode.
                    while (_sendQueue.TryDequeue(out _)) { }
                    // Drain leftover semaphore permits.
                    while (_sendSignal.CurrentCount > 0)
                        _sendSignal.Wait(0);

                    _readTask  = Task.Run(() => ReadLoopAsync(ct), ct);
                    _writeTask = Task.Run(() => WriteLoopAsync(ct), ct);

                    // Wait until one of them finishes (i.e. the client disconnected).
                    await Task.WhenAny(_readTask, _writeTask);
                    DisconnectClient();

                    ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                        .Info("[SocketServer] Client disconnected. Waiting for new connection.");
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                        .Warn($"[SocketServer] Accept error: {ex.Message}");
                    await Task.Delay(500, ct).ConfigureAwait(false);
                }
            }
        }

        private async Task ReadLoopAsync(CancellationToken ct)
        {
            try
            {
                while (!ct.IsCancellationRequested && _reader != null)
                {
                    string? line = await _reader.ReadLineAsync(ct);
                    if (line == null) break; // stream closed

                    if (string.IsNullOrWhiteSpace(line)) continue;

                    // Peek message type first
                    string? msgType = MessageSerializer.PeekMessageType(line);

                    if (msgType == "ready")
                    {
                        _pendingReady = true;
                        continue;
                    }

                    var action = MessageSerializer.DeserializeAction(line);
                    if (action != null)
                        _actionBuffer.SetAction(action);
                }
            }
            catch (OperationCanceledException) { /* expected on shutdown */ }
            catch (Exception ex)
            {
                ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                    .Warn($"[SocketServer] Read error: {ex.Message}");
            }
        }

        private async Task WriteLoopAsync(CancellationToken ct)
        {
            try
            {
                while (!ct.IsCancellationRequested && _writer != null)
                {
                    await _sendSignal.WaitAsync(ct);

                    // Drain all queued messages in one flush
                    while (_sendQueue.TryDequeue(out string? line) && line != null)
                    {
                        await _writer.WriteAsync(line.AsMemory(), ct);
                    }
                    await _writer.FlushAsync(ct);
                }
            }
            catch (OperationCanceledException) { /* expected on shutdown */ }
            catch (Exception ex)
            {
                ModContent.GetInstance<TerrariaRLAgent>()?.Logger
                    .Warn($"[SocketServer] Write error: {ex.Message}");
            }
        }

        private void DisconnectClient()
        {
            _isClientConnected = false;
            try { _writer?.Dispose(); } catch { /* ignore */ }
            try { _reader?.Dispose(); } catch { /* ignore */ }
            try { _stream?.Dispose(); } catch { /* ignore */ }
            try { _client?.Close(); } catch { /* ignore */ }
            _writer = null;
            _reader = null;
            _stream = null;
            _client = null;
            // Release write loop so it can exit.
            _sendSignal.Release();
        }
    }
}
