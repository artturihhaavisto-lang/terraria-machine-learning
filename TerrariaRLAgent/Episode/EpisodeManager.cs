using System;
using Terraria;
using Terraria.ModLoader;
using TerrariaRLAgent.Config;
using TerrariaRLAgent.Networking;
using TerrariaRLAgent.Observation;

namespace TerrariaRLAgent.Episode
{
    // =========================================================================
    // State enum
    // =========================================================================

    public enum EpisodeState
    {
        /// <summary>Mod loaded; no active episode. Waiting for a socket connection.</summary>
        IDLE,

        /// <summary>Client connected; waiting for a "ready" JSON message.</summary>
        WAITING_FOR_READY,

        /// <summary>Setting up arena, spawning boss. Lasts a few ticks.</summary>
        SUMMONING,

        /// <summary>Boss is alive; agent is controlling the player.</summary>
        FIGHTING,

        /// <summary>Boss defeated or player died or timeout; sending episode_end.</summary>
        EPISODE_END
    }

    // =========================================================================
    // EpisodeManager
    // =========================================================================

    /// <summary>
    /// Central state machine that drives the RL training loop.
    ///
    /// Tick flow:
    ///   IDLE              → WAITING_FOR_READY   (socket client connects)
    ///   WAITING_FOR_READY → SUMMONING            (client sends {type:"ready"})
    ///   SUMMONING         → FIGHTING             (after setup completes)
    ///   FIGHTING          → EPISODE_END          (boss dead / player dead / timeout)
    ///   EPISODE_END       → WAITING_FOR_READY    (after cleanup + episode_end sent)
    /// </summary>
    public class EpisodeManager
    {
        // ------------------------------------------------------------------ //
        // Constants
        // ------------------------------------------------------------------ //

        /// <summary>Ticks spent in SUMMONING state before transitioning to FIGHTING.</summary>
        private const int SummoningSetupTicks = 60;

        // ------------------------------------------------------------------ //
        // State
        // ------------------------------------------------------------------ //

        public EpisodeState CurrentState { get; private set; } = EpisodeState.IDLE;

        // Episode counters
        public long EpisodeTick            { get; private set; }
        public int  DamageDealtThisEpisode { get; private set; }
        public int  DamageTakenThisEpisode { get; private set; }

        // Per-tick reward deltas (populated by CheckBossStatus / RecordDamageTaken)
        private int _tickDamageDealt;
        private int _tickDamageTaken;

        // Previous boss HP sum, used to calculate damage dealt each tick
        private int _prevBossHpSum;

        // Summoning setup counter
        private int _summoningTicksElapsed;

        // EPISODE_END cooldown (ticks before returning to WAITING_FOR_READY)
        private int _episodeEndCooldown;
        private const int EpisodeEndCooldownTicks = 120;

        // Result string for the episode_end message
        private string _episodeResult = "timeout";
        private int    _bossHpRemaining;

        private readonly TerrariaRLAgent _mod;

        // ------------------------------------------------------------------ //
        // Constructor
        // ------------------------------------------------------------------ //

        public EpisodeManager(TerrariaRLAgent mod)
        {
            _mod = mod;
        }

        // ------------------------------------------------------------------ //
        // Public API
        // ------------------------------------------------------------------ //

        /// <summary>Called every game tick from RLModSystem.PostUpdateWorld().</summary>
        public void Tick()
        {
            switch (CurrentState)
            {
                case EpisodeState.IDLE:
                    TickIdle();
                    break;

                case EpisodeState.WAITING_FOR_READY:
                    TickWaitingForReady();
                    break;

                case EpisodeState.SUMMONING:
                    TickSummoning();
                    break;

                case EpisodeState.FIGHTING:
                    TickFighting();
                    break;

                case EpisodeState.EPISODE_END:
                    TickEpisodeEnd();
                    break;
            }

            // Reset per-tick deltas at the end of every tick
            _tickDamageDealt = 0;
            _tickDamageTaken = 0;
        }

        /// <summary>
        /// Called from RLModSystem.PostUpdateNPCs() when in FIGHTING state.
        /// Detects boss death/despawn and transitions to EPISODE_END.
        /// Also accumulates damage-dealt by comparing current vs previous boss HP.
        /// </summary>
        public void CheckBossStatus()
        {
            if (CurrentState != EpisodeState.FIGHTING) return;

            int bossHpSum  = 0;
            bool bossAlive = false;

            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active) continue;
                if (!npc.boss)   continue;

                bossAlive  = true;
                bossHpSum += Math.Max(0, npc.life);
            }

            if (_prevBossHpSum > 0 && bossHpSum < _prevBossHpSum)
            {
                int dealt         = _prevBossHpSum - bossHpSum;
                _tickDamageDealt += dealt;
                DamageDealtThisEpisode += dealt;
            }

            _prevBossHpSum    = bossHpSum;
            _bossHpRemaining  = bossHpSum;

            // Boss defeated
            if (!bossAlive && CurrentState == EpisodeState.FIGHTING)
            {
                _mod.Logger.Info("[EpisodeManager] Boss defeated! Transitioning to EPISODE_END.");
                BeginEpisodeEnd("win");
                return;
            }

            // Player died
            Player player = Main.LocalPlayer;
            if (player != null && !player.active)
            {
                _mod.Logger.Info("[EpisodeManager] Player died. Transitioning to EPISODE_END.");
                BeginEpisodeEnd("loss");
                return;
            }

            // Timeout
            var cfg = ModContent.GetInstance<RLConfig>();
            int timeout = cfg?.EpisodeTimeout ?? 18000;
            if (EpisodeTick >= timeout)
            {
                _mod.Logger.Info("[EpisodeManager] Episode timeout. Transitioning to EPISODE_END.");
                BeginEpisodeEnd("timeout");
            }
        }

        /// <summary>Called from RLModPlayer.OnHurt() to record damage taken.</summary>
        public void RecordDamageTaken(int damage)
        {
            _tickDamageTaken       += damage;
            DamageTakenThisEpisode += damage;
        }

        // ------------------------------------------------------------------ //
        // State tick handlers
        // ------------------------------------------------------------------ //

        private void TickIdle()
        {
            // Transition as soon as a client connects
            if (_mod.SocketServer?.IsClientConnected == true)
            {
                _mod.Logger.Info("[EpisodeManager] Client connected. Moving to WAITING_FOR_READY.");
                TransitionTo(EpisodeState.WAITING_FOR_READY);
            }
        }

        private void TickWaitingForReady()
        {
            // If client disconnected, go back to IDLE
            if (_mod.SocketServer?.IsClientConnected != true)
            {
                _mod.Logger.Info("[EpisodeManager] Client disconnected. Returning to IDLE.");
                TransitionTo(EpisodeState.IDLE);
                return;
            }

            // Check if we received a ready message via the ActionBuffer acting as a
            // command channel. A "ready" message is identified by a null/default action
            // with a special flag. In practice the Python side sends:
            //   {"type":"ready","boss_type":4}
            // The SocketServer now dispatches this through the ActionBuffer;
            // we peek the latest received raw type via the server's pending ready flag.
            if (_mod.SocketServer.HasPendingReady)
            {
                _mod.SocketServer.ConsumePendingReady();
                _mod.Logger.Info("[EpisodeManager] Ready message received. Beginning SUMMONING.");
                BeginSummoning();
            }
        }

        private void TickSummoning()
        {
            _summoningTicksElapsed++;

            if (_summoningTicksElapsed >= SummoningSetupTicks)
            {
                _mod.Logger.Info("[EpisodeManager] Setup complete. Transitioning to FIGHTING.");
                TransitionTo(EpisodeState.FIGHTING);
            }
        }

        private void TickFighting()
        {
            EpisodeTick++;

            // Send observation with reward signals this tick
            // (Building is done in RLModSystem; we attach reward signals here)
            // This is intentionally left for RLModSystem to handle so that
            // observations are built after all NPC/projectile updates.
        }

        private void TickEpisodeEnd()
        {
            _episodeEndCooldown--;

            if (_episodeEndCooldown <= 0)
            {
                CleanupAfterEpisode();

                if (_mod.SocketServer?.IsClientConnected == true)
                {
                    _mod.Logger.Info("[EpisodeManager] Episode cleanup done. Moving to WAITING_FOR_READY.");
                    TransitionTo(EpisodeState.WAITING_FOR_READY);
                }
                else
                {
                    _mod.Logger.Info("[EpisodeManager] Client gone. Moving to IDLE.");
                    TransitionTo(EpisodeState.IDLE);
                }
            }
        }

        // ------------------------------------------------------------------ //
        // Helpers
        // ------------------------------------------------------------------ //

        private void BeginSummoning()
        {
            TransitionTo(EpisodeState.SUMMONING);
            _summoningTicksElapsed = 0;

            Player player = Main.LocalPlayer;
            if (player == null) return;

            // --- Reset player state ---
            player.statLife        = player.statLifeMax2;
            player.statMana        = player.statManaMax2;
            player.immune          = false;
            player.immuneTime      = 0;

            // Clear all existing player buffs
            for (int i = 0; i < Player.MaxBuffs; i++)
                player.buffType[i] = 0;

            // Heal
            player.Heal(player.statLifeMax2);

            // --- Teleport to arena ---
            ArenaManager.TeleportPlayerToArena(player);

            // --- Clear hostile NPCs and projectiles in the arena ---
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active) continue;
                if (npc.friendly || npc.townNPC) continue;
                npc.active = false;
                npc.life   = 0;
            }

            for (int i = 0; i < Main.maxProjectiles; i++)
            {
                Projectile proj = Main.projectile[i];
                if (!proj.active) continue;
                if (!proj.hostile && proj.owner == Main.myPlayer) continue;
                proj.active = false;
            }

            // --- Summon boss ---
            var cfg = ModContent.GetInstance<RLConfig>();
            int bossType = cfg?.BossNPCType ?? 4;
            bool spawned = BossSummoner.SummonBoss(player, bossType);
            if (!spawned)
                _mod.Logger.Warn($"[EpisodeManager] Boss spawn may have failed for type {bossType}.");

            // Initialise damage tracking
            DamageDealtThisEpisode = 0;
            DamageTakenThisEpisode = 0;
            _prevBossHpSum         = 0;
            EpisodeTick            = 0;

            // Reset ActionBuffer so stale actions from the previous episode are discarded
            _mod.ActionBuffer.Clear();

            // Reset ObservationBuilder episode counter (accessed via RLModSystem)
            ModContent.GetInstance<RLModSystem>()?.ResetObservationBuilder();
        }

        private void BeginEpisodeEnd(string result)
        {
            _episodeResult      = result;
            _episodeEndCooldown = EpisodeEndCooldownTicks;
            TransitionTo(EpisodeState.EPISODE_END);

            // Send episode_end message immediately
            SendEpisodeEndMessage();
        }

        private void SendEpisodeEndMessage()
        {
            if (_mod.SocketServer == null) return;

            var endData = new EpisodeEndData
            {
                Type             = "episode_end",
                Result           = _episodeResult,
                TicksSurvived    = EpisodeTick,
                DamageDealtTotal = DamageDealtThisEpisode,
                DamageTakenTotal = DamageTakenThisEpisode,
                BossHpRemaining  = _bossHpRemaining
            };

            string json = MessageSerializer.SerializeEpisodeEnd(endData);
            _mod.SocketServer.EnqueueRaw(json);
        }

        private void CleanupAfterEpisode()
        {
            // Despawn remaining bosses
            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active) continue;
                if (!npc.boss)   continue;
                npc.active = false;
                npc.life   = 0;
            }

            // Clear hostile projectiles
            for (int i = 0; i < Main.maxProjectiles; i++)
            {
                if (Main.projectile[i].active && Main.projectile[i].hostile)
                    Main.projectile[i].active = false;
            }

            // Respawn the player if dead
            Player player = Main.LocalPlayer;
            if (player != null && player.dead)
            {
                player.Spawn(Terraria.DataStructures.PlayerSpawnContext.ReviveFromDeath);
            }
        }

        private void TransitionTo(EpisodeState next)
        {
            _mod.Logger.Info($"[EpisodeManager] {CurrentState} -> {next}");
            CurrentState = next;
        }

        /// <summary>
        /// Exposes per-tick damage signals so RLModSystem can attach them to
        /// the observation before sending.
        /// </summary>
        public (int damageDealt, int damageTaken) GetTickRewards()
            => (_tickDamageDealt, _tickDamageTaken);
    }
}
