using Terraria;
using Terraria.ModLoader;
using TerrariaRLAgent.Config;
using TerrariaRLAgent.Episode;
using TerrariaRLAgent.Observation;

namespace TerrariaRLAgent
{
    public class RLModSystem : ModSystem
    {
        private ObservationBuilder? _observationBuilder;

        /// <summary>Called by EpisodeManager at the start of each new episode.</summary>
        public void ResetObservationBuilder() => _observationBuilder?.ResetEpisode();

        public override void OnWorldLoad()
        {
            var mod = TerrariaRLAgent.Instance;
            if (mod == null) return;

            // Initialize the socket server using config port now that the world is loaded
            var config = ModContent.GetInstance<RLConfig>();
            int port = config?.Port ?? 7777;
            mod.InitializeSocketServer(port);

            _observationBuilder = new ObservationBuilder();

            Mod.Logger.Info("[TerrariaRLAgent] World loaded. Socket server initialized.");
        }

        public override void OnWorldUnload()
        {
            _observationBuilder = null;
        }

        public override void PostUpdateNPCs()
        {
            var mod = TerrariaRLAgent.Instance;
            if (mod?.EpisodeManager == null) return;

            // Check if we're in FIGHTING state and the boss has died or despawned
            if (mod.EpisodeManager.CurrentState == EpisodeState.FIGHTING)
            {
                mod.EpisodeManager.CheckBossStatus();
            }
        }

        public override void PostUpdateWorld()
        {
            var mod = TerrariaRLAgent.Instance;
            if (mod?.EpisodeManager == null) return;

            // Tick the episode state machine every game tick
            mod.EpisodeManager.Tick();

            // If in FIGHTING state, build and send observation to the Python backend
            if (mod.EpisodeManager.CurrentState == EpisodeState.FIGHTING
                && _observationBuilder != null
                && mod.SocketServer != null
                && mod.SocketServer.IsClientConnected)
            {
                var player = Main.LocalPlayer;
                if (player != null && player.active)
                {
                    var observation = _observationBuilder.Build(player);

                    // Attach per-tick reward signals
                    var (dealt, taken) = mod.EpisodeManager.GetTickRewards();
                    ObservationBuilder.AttachRewardSignals(observation, dealt, taken);

                    // Non-blocking: enqueue observation to be sent by the socket thread
                    mod.SocketServer.EnqueueObservation(observation);
                }
            }
        }
    }
}
