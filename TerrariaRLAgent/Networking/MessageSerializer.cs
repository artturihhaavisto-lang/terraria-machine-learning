using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TerrariaRLAgent.Networking
{
    // -------------------------------------------------------------------------
    // Data transfer objects
    // -------------------------------------------------------------------------

    public class Vec2
    {
        [JsonPropertyName("x")] public float X { get; set; }
        [JsonPropertyName("y")] public float Y { get; set; }

        public Vec2() { }
        public Vec2(float x, float y) { X = x; Y = y; }
    }

    public class HitboxData
    {
        [JsonPropertyName("x")]      public float X      { get; set; }
        [JsonPropertyName("y")]      public float Y      { get; set; }
        [JsonPropertyName("width")]  public float Width  { get; set; }
        [JsonPropertyName("height")] public float Height { get; set; }
    }

    public class PlayerData
    {
        [JsonPropertyName("position")]               public Vec2     Position             { get; set; } = new();
        [JsonPropertyName("velocity")]               public Vec2     Velocity             { get; set; } = new();
        [JsonPropertyName("hp")]                     public int      Hp                   { get; set; }
        [JsonPropertyName("max_hp")]                 public int      MaxHp                { get; set; }
        [JsonPropertyName("mana")]                   public int      Mana                 { get; set; }
        [JsonPropertyName("max_mana")]               public int      MaxMana              { get; set; }
        [JsonPropertyName("defense")]                public int      Defense              { get; set; }
        [JsonPropertyName("grounded")]               public bool     Grounded             { get; set; }
        [JsonPropertyName("wing_time_remaining")]    public int      WingTimeRemaining    { get; set; }
        [JsonPropertyName("has_double_jump")]        public bool     HasDoubleJump        { get; set; }
        [JsonPropertyName("dash_cooldown")]          public int      DashCooldown         { get; set; }
        [JsonPropertyName("potion_sickness_ticks")]  public int      PotionSicknessTicks  { get; set; }
        [JsonPropertyName("immune_ticks")]           public int      ImmuneTicks          { get; set; }
        [JsonPropertyName("active_buffs")]           public List<int> ActiveBuffs         { get; set; } = new();
        [JsonPropertyName("active_debuffs")]         public List<int> ActiveDebuffs       { get; set; } = new();
        [JsonPropertyName("held_item_use_time")]     public int      HeldItemUseTime      { get; set; }
        [JsonPropertyName("held_item_damage")]       public int      HeldItemDamage       { get; set; }
        [JsonPropertyName("held_item_type")]         public string   HeldItemType         { get; set; } = "none";
        [JsonPropertyName("direction_facing")]       public int      DirectionFacing      { get; set; }
    }

    public class BossData
    {
        [JsonPropertyName("npc_index")]   public int      NpcIndex  { get; set; }
        [JsonPropertyName("type_id")]     public int      TypeId    { get; set; }
        [JsonPropertyName("position")]    public Vec2     Position  { get; set; } = new();
        [JsonPropertyName("velocity")]    public Vec2     Velocity  { get; set; } = new();
        [JsonPropertyName("hp")]          public int      Hp        { get; set; }
        [JsonPropertyName("max_hp")]      public int      MaxHp     { get; set; }
        [JsonPropertyName("hp_percent")]  public float    HpPercent { get; set; }
        [JsonPropertyName("hitbox")]      public HitboxData Hitbox  { get; set; } = new();
        [JsonPropertyName("defense")]     public int      Defense   { get; set; }
        [JsonPropertyName("damage")]      public int      Damage    { get; set; }
        [JsonPropertyName("ai_style")]    public int      AiStyle   { get; set; }
        [JsonPropertyName("ai_slots")]    public float[]  AiSlots   { get; set; } = new float[4];
        [JsonPropertyName("active")]      public bool     Active    { get; set; }
        [JsonPropertyName("real_life")]   public int      RealLife  { get; set; }
    }

    public class ProjectileData
    {
        [JsonPropertyName("proj_index")]  public int      ProjIndex { get; set; }
        [JsonPropertyName("type_id")]     public int      TypeId    { get; set; }
        [JsonPropertyName("position")]    public Vec2     Position  { get; set; } = new();
        [JsonPropertyName("velocity")]    public Vec2     Velocity  { get; set; } = new();
        [JsonPropertyName("hitbox")]      public HitboxData Hitbox  { get; set; } = new();
        [JsonPropertyName("damage")]      public int      Damage    { get; set; }
        [JsonPropertyName("time_left")]   public int      TimeLeft  { get; set; }
        [JsonPropertyName("penetrate")]   public int      Penetrate { get; set; }
        [JsonPropertyName("distance")]    public float    Distance  { get; set; }
    }

    public class EnemyData
    {
        [JsonPropertyName("npc_index")]  public int      NpcIndex { get; set; }
        [JsonPropertyName("type_id")]    public int      TypeId   { get; set; }
        [JsonPropertyName("position")]   public Vec2     Position { get; set; } = new();
        [JsonPropertyName("velocity")]   public Vec2     Velocity { get; set; } = new();
        [JsonPropertyName("hp")]         public int      Hp       { get; set; }
        [JsonPropertyName("max_hp")]     public int      MaxHp    { get; set; }
        [JsonPropertyName("hitbox")]     public HitboxData Hitbox { get; set; } = new();
        [JsonPropertyName("damage")]     public int      Damage   { get; set; }
        [JsonPropertyName("distance")]   public float    Distance { get; set; }
    }

    public class TileGridData
    {
        [JsonPropertyName("width")]  public int     Width  { get; set; }
        [JsonPropertyName("height")] public int     Height { get; set; }
        /// <summary>Row-major flat array: grid[row * Width + col]</summary>
        [JsonPropertyName("grid")]   public byte[]  Grid   { get; set; } = System.Array.Empty<byte>();
        [JsonPropertyName("origin_tile_x")] public int OriginTileX { get; set; }
        [JsonPropertyName("origin_tile_y")] public int OriginTileY { get; set; }
    }

    public class ObservationData
    {
        [JsonPropertyName("type")]        public string            Type        { get; set; } = "observation";
        [JsonPropertyName("tick")]        public long              Tick        { get; set; }
        [JsonPropertyName("player")]      public PlayerData        Player      { get; set; } = new();
        [JsonPropertyName("bosses")]      public List<BossData>    Bosses      { get; set; } = new();
        [JsonPropertyName("projectiles")] public List<ProjectileData> Projectiles { get; set; } = new();
        [JsonPropertyName("enemies")]     public List<EnemyData>   Enemies     { get; set; } = new();
        [JsonPropertyName("tile_grid")]   public TileGridData      TileGrid    { get; set; } = new();
        [JsonPropertyName("episode_tick")] public long             EpisodeTick { get; set; }
        [JsonPropertyName("damage_dealt")] public float            DamageDealt { get; set; }
        [JsonPropertyName("damage_taken")] public float            DamageTaken { get; set; }
    }

    public class EpisodeEndData
    {
        [JsonPropertyName("type")]               public string Type              { get; set; } = "episode_end";
        [JsonPropertyName("result")]             public string Result            { get; set; } = "unknown";
        [JsonPropertyName("ticks_survived")]     public long   TicksSurvived     { get; set; }
        [JsonPropertyName("damage_dealt_total")] public float  DamageDealtTotal  { get; set; }
        [JsonPropertyName("damage_taken_total")] public float  DamageTakenTotal  { get; set; }
        [JsonPropertyName("boss_hp_remaining")]  public float  BossHpRemaining   { get; set; }
    }

    public class ReadyRequestData
    {
        [JsonPropertyName("type")]      public string Type      { get; set; } = "ready";
        [JsonPropertyName("boss_type")] public int    BossType  { get; set; }
    }

    public class ActionData
    {
        /// <summary>movement[0]=left, movement[1]=right, movement[2]=up, movement[3]=down</summary>
        [JsonPropertyName("movement")]  public int[]  Movement  { get; set; } = new int[4];
        [JsonPropertyName("jump")]      public int    Jump      { get; set; }
        [JsonPropertyName("use_item")]  public int    UseItem   { get; set; }
        /// <summary>Angle in radians from player to target (0 = right, PI/2 = down)</summary>
        [JsonPropertyName("aim_angle")] public float  AimAngle  { get; set; }
        [JsonPropertyName("dash")]      public int    Dash      { get; set; }
        [JsonPropertyName("grapple")]   public int    Grapple   { get; set; }
        [JsonPropertyName("heal")]      public int    Heal      { get; set; }
        [JsonPropertyName("type")]      public string Type      { get; set; } = "action";
    }

    // -------------------------------------------------------------------------
    // Serializer helpers
    // -------------------------------------------------------------------------

    public static class MessageSerializer
    {
        private static readonly JsonSerializerOptions _options = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
            WriteIndented = false
        };

        public static string SerializeObservation(ObservationData obs)
            => JsonSerializer.Serialize(obs, _options);

        public static string SerializeEpisodeEnd(EpisodeEndData end)
            => JsonSerializer.Serialize(end, _options);

        public static ActionData? DeserializeAction(string json)
        {
            try
            {
                return JsonSerializer.Deserialize<ActionData>(json, _options);
            }
            catch
            {
                return null;
            }
        }

        public static ReadyRequestData? DeserializeReady(string json)
        {
            try
            {
                return JsonSerializer.Deserialize<ReadyRequestData>(json, _options);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Peek at the "type" field without full deserialization.
        /// </summary>
        public static string? PeekMessageType(string json)
        {
            try
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("type", out var typeEl))
                    return typeEl.GetString();
            }
            catch { }
            return null;
        }
    }
}
