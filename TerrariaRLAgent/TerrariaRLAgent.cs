using Terraria;
using Terraria.ModLoader;
using TerrariaRLAgent.Networking;
using TerrariaRLAgent.Episode;
using TerrariaRLAgent.Control;

namespace TerrariaRLAgent
{
    public class TerrariaRLAgent : Mod
    {
        public static TerrariaRLAgent Instance { get; private set; } = null!;

        public SocketServer? SocketServer { get; private set; }
        public EpisodeManager? EpisodeManager { get; private set; }
        public ActionBuffer ActionBuffer { get; private set; } = null!;

        public override void Load()
        {
            Instance = this;
            ActionBuffer = new ActionBuffer();
            EpisodeManager = new EpisodeManager(this);
            Logger.Info("[TerrariaRLAgent] Mod loaded. Initializing socket server...");
        }

        public void InitializeSocketServer(int port)
        {
            SocketServer?.Stop();
            SocketServer = new SocketServer(port, ActionBuffer);
            SocketServer.Start();
            Logger.Info($"[TerrariaRLAgent] Socket server started on port {port}.");
        }

        public override void Unload()
        {
            Logger.Info("[TerrariaRLAgent] Unloading mod, stopping socket server...");
            SocketServer?.Stop();
            SocketServer = null;
            EpisodeManager = null;
            Instance = null!;
        }
    }
}
