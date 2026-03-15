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
                WingTimeRemaining = player.wingTime,
                HasDoubleJump     = player.jumpAgainBlizzard || player.jumpAgainCloud ||
                                    player.jumpAgainFart   || player.jumpAgainSandstorm ||
                                    player.jumpAgainUnicorn,
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
            if (item.magic)  return "magic";
            if (item.melee)  return "melee";
            if (item.ranged) return "ranged";
            if (item.summon) return "summon";
            return "other";
        }

        /// <summary>
        /// Build a HashSet of vanilla debuff IDs so we can separate buffs from debuffs.
        /// tModLoader exposes BuffID.Sets.IsADebuff[] for this purpose.
        /// </summary>
        private static HashSet<int> BuildDebuffSet()
        {
            var set = new HashSet<int>();
            // BuffID.Sets.IsADebuff is available in tML 1.4+
            for (int i = 0; i < BuffID.Sets.IsADebuff.Length; i++)
            {
                if (BuffID.Sets.IsADebuff[i])
                    set.Add(i);
            }
            return set;
        }
    }
}
