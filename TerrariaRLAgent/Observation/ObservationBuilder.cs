using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Assembles a complete ObservationData snapshot by calling each sub-observer.
    /// The tile grid is cached and only rebuilt every TileObserver.RebuildInterval ticks.
    /// </summary>
    public class ObservationBuilder
    {
        private readonly TileObserver _tileObserver = new();
        private int _tick;
        private int _episodeTick;

        /// <summary>Increments the episode tick counter (call once per episode tick).</summary>
        public void TickEpisode() => _episodeTick++;

        /// <summary>Resets episode-local counters at the start of a new episode.</summary>
        public void ResetEpisode()
        {
            _episodeTick = 0;
        }

        public ObservationData Build(Player player)
        {
            _tick++;

            var obs = new ObservationData
            {
                Type        = "observation",
                Tick        = Main.GameUpdateCount,
                EpisodeTick = _episodeTick,
                Player      = PlayerObserver.Observe(player),
                Bosses      = BossObserver.Observe(),
                Projectiles = ProjectileObserver.Observe(player),
                Enemies     = EnemyObserver.Observe(player),
                TileGrid    = _tileObserver.Observe(player, _tick),
                // Reward signals are filled in by EpisodeManager before sending
                DamageDealt = 0f,
                DamageTaken = 0f
            };

            return obs;
        }

        /// <summary>
        /// Attach per-tick reward signals (called by EpisodeManager after Build).
        /// </summary>
        public static void AttachRewardSignals(ObservationData obs, float damageDealt, float damageTaken)
        {
            obs.DamageDealt = damageDealt;
            obs.DamageTaken = damageTaken;
        }
    }
}
