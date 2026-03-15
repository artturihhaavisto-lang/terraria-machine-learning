using System.Collections.Generic;
using Terraria;
using Terraria.ID;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Extracts a snapshot of the local player's state for the observation vector.
    /// </summary>
    public static class PlayerObserver
    {
        private static readonly HashSet<int> s_debuffIds = BuildDebuffSet();

        public static PlayerData Observe(Player player)
        {
            var data = new PlayerData
            {
                Position = new Vec2(player.position.X, player.position.Y),
                Velocity = new Vec2(player.velocity.X, player.velocity.Y),
                Hp       = player.statLife,
                MaxHp    = player.statLifeMax2,
                Mana     = player.statMana,
                MaxMana  = player.statManaMax2,
                Defense  = player.statDefense,
                // Grounded: vertical velocity is zero and there is a solid tile immediately below the player.
                Grounded = player.velocity.Y == 0f
                           && Collision.SolidCollision(
                               player.position + new Microsoft.Xna.Framework.Vector2(0, player.height),
                               player.width, 4),
                WingTimeRemaining = (int)player.wingTime,
                // HasDoubleJump: check wing flight or any active extra-jump option.
                // In tML 1.4.4 the per-jump booleans were replaced with the ExtraJump
                // system; use wingTime as the primary flight indicator.
                HasDoubleJump     = player.wingTime > 0,
                DashCooldown      = player.dashDelay,
                ImmuneTicks       = player.immuneTime,
                DirectionFacing   = player.direction,
            };

            // Potion sickness: buff index 21 = PotionSickness
            for (int i = 0; i < Player.MaxBuffs; i++)
            {
                int buffType = player.buffType[i];
                if (buffType == 0) continue;

                if (buffType == BuffID.PotionSickness)
                    data.PotionSicknessTicks = player.buffTime[i];

                if (s_debuffIds.Contains(buffType))
                    data.ActiveDebuffs.Add(buffType);
                else
                    data.ActiveBuffs.Add(buffType);
            }

            // Held item stats
            var heldItem = player.HeldItem;
            if (heldItem != null && !heldItem.IsAir)
            {
                data.HeldItemUseTime  = heldItem.useTime;
                data.HeldItemDamage   = player.GetWeaponDamage(heldItem);
                data.HeldItemType     = ClassifyItem(heldItem);
            }
            else
            {
                data.HeldItemType = "none";
            }

            return data;
        }

        private static string ClassifyItem(Item item)
        {
            // tML 1.4.4 replaced item.magic/melee/ranged/summon with the DamageClass system.
            if (item.CountsAsClass(DamageClass.Magic))  return "magic";
            if (item.CountsAsClass(DamageClass.Melee))  return "melee";
            if (item.CountsAsClass(DamageClass.Ranged)) return "ranged";
            if (item.CountsAsClass(DamageClass.Summon)) return "summon";
            return "other";
        }

        /// <summary>
        /// Build a HashSet of vanilla debuff IDs so we can separate buffs from debuffs.
        /// Uses Main.debuff[] which is a bool[] available in all tML 1.4.x versions.
        /// </summary>
        private static HashSet<int> BuildDebuffSet()
        {
            var set = new HashSet<int>();
            for (int i = 0; i < Main.debuff.Length; i++)
            {
                if (Main.debuff[i])
                    set.Add(i);
            }
            return set;
        }
    }
}
