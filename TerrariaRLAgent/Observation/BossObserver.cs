using System.Collections.Generic;
using Terraria;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Observation
{
    /// <summary>
    /// Finds all active boss NPCs (including multi-part bosses) and returns
    /// structured data for each part.
    /// </summary>
    public static class BossObserver
    {
        public static List<BossData> Observe()
        {
            var result = new List<BossData>();

            for (int i = 0; i < Main.maxNPCs; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active) continue;

                // Include main boss parts (npc.boss == true)
                // Also include servant/part NPCs whose realLife index points to a boss.
                bool isBossPart = npc.boss;
                bool isSubPart  = false;

                if (!isBossPart && npc.realLife >= 0 && npc.realLife < Main.maxNPCs)
                {
                    NPC realLife = Main.npc[npc.realLife];
                    if (realLife.active && realLife.boss)
                        isSubPart = true;
                }

                if (!isBossPart && !isSubPart) continue;

                result.Add(ExtractBossData(npc, i, isBossPart));
            }

            return result;
        }

        private static BossData ExtractBossData(NPC npc, int index, bool isRealLife)
        {
            float maxHp  = npc.lifeMax > 0 ? npc.lifeMax : 1;
            float curHp  = npc.life;
            float hpPct  = curHp / maxHp;

            return new BossData
            {
                NpcIndex  = index,
                TypeId    = npc.type,
                Position  = new Vec2(npc.position.X, npc.position.Y),
                Velocity  = new Vec2(npc.velocity.X, npc.velocity.Y),
                Hp        = npc.life,
                MaxHp     = npc.lifeMax,
                HpPercent = hpPct,
                Hitbox    = new HitboxData
                {
                    X      = npc.Hitbox.X,
                    Y      = npc.Hitbox.Y,
                    Width  = npc.Hitbox.Width,
                    Height = npc.Hitbox.Height
                },
                Defense   = npc.defense,
                Damage    = npc.damage,
                AiStyle   = npc.aiStyle,
                AiSlots   = new float[] { npc.ai[0], npc.ai[1], npc.ai[2], npc.ai[3] },
                Active    = npc.active,
                RealLife  = npc.realLife
            };
        }
    }
}
