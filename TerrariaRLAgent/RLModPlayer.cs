using Terraria;
using Terraria.ModLoader;
using TerrariaRLAgent.Control;
using TerrariaRLAgent.Episode;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent
{
    public class RLModPlayer : ModPlayer
    {
        // Damage tracking for reward calculation
        public int DamageTakenThisTick { get; private set; }
        public int TotalDamageTaken { get; private set; }

        private bool _wasHurtThisTick;
        private int _hurtAmount;

        public override void PreUpdate()
        {
            // Reset per-tick damage
            DamageTakenThisTick = 0;
            _wasHurtThisTick = false;

            var mod = TerrariaRLAgent.Instance;
            if (mod?.EpisodeManager == null)
                return;

            // Only control during FIGHTING state
            if (mod.EpisodeManager.CurrentState != EpisodeState.FIGHTING)
                return;

            // Only control the local player
            if (Player.whoAmI != Main.myPlayer)
                return;

            // Suppress all real input so the agent has full control
            SuppressRealInput();

            // Apply current action from the buffer
            var action = mod.ActionBuffer.GetAction();
            if (action != null)
            {
                PlayerController.ApplyAction(Player, action);
            }
        }

        public override void PostUpdate()
        {
            if (_wasHurtThisTick)
            {
                DamageTakenThisTick = _hurtAmount;
                TotalDamageTaken += _hurtAmount;
            }
        }

        /// <summary>
        /// Called when this player takes damage. Records damage for episode tracking.
        /// </summary>
        public override void OnHurt(Player.HurtInfo info)
        {
            _wasHurtThisTick = true;
            _hurtAmount = info.Damage;

            var mod = TerrariaRLAgent.Instance;
            mod?.EpisodeManager?.RecordDamageTaken(info.Damage);
        }

        /// <summary>
        /// Resets damage tracking for a new episode.
        /// </summary>
        public void ResetDamageTracking()
        {
            DamageTakenThisTick = 0;
            TotalDamageTaken = 0;
            _wasHurtThisTick = false;
            _hurtAmount = 0;
        }

        /// <summary>
        /// Blocks all player control flags so only the RL agent's synthetic inputs are
        /// processed.  Must be called before <see cref="PlayerController.ApplyAction"/>.
        /// In tModLoader 1.4 the control booleans on the Player object are the single
        /// source of truth after ProcessTriggers; clearing them here is sufficient.
        /// </summary>
        private void SuppressRealInput()
        {
            Player.controlLeft      = false;
            Player.controlRight     = false;
            Player.controlUp        = false;
            Player.controlDown      = false;
            Player.controlJump      = false;
            Player.controlUseItem   = false;
            Player.controlUseTile   = false;
            Player.controlThrow     = false;
            Player.controlHook      = false;
            Player.controlInv       = false;
            Player.controlSmart     = false;
            Player.controlQuickMana = false;
            Player.controlMount     = false;
        }
    }
}
