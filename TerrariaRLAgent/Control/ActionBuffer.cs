using System.Threading;
using TerrariaRLAgent.Networking;

namespace TerrariaRLAgent.Control
{
    /// <summary>
    /// Thread-safe single-slot buffer for the latest ActionData received from the
    /// Python ML backend.  The socket (background) thread writes; the game
    /// (main) thread reads.  Reads are non-destructive: the last action is
    /// repeated until a newer one arrives.
    /// </summary>
    public class ActionBuffer
    {
        private ActionData? _current;
        private readonly object _lock = new();

        /// <summary>Returns true if at least one action has been received.</summary>
        public bool HasAction
        {
            get
            {
                lock (_lock) { return _current != null; }
            }
        }

        /// <summary>
        /// Returns the most recent action, or null if none has been received yet.
        /// The action is NOT consumed; the same value is returned on subsequent calls
        /// until SetAction replaces it.
        /// </summary>
        public ActionData? GetAction()
        {
            lock (_lock) { return _current; }
        }

        /// <summary>
        /// Stores a new action from the network thread.
        /// Replaces the previous value atomically.
        /// </summary>
        public void SetAction(ActionData action)
        {
            lock (_lock) { _current = action; }
        }

        /// <summary>Clears the stored action (e.g. at episode reset).</summary>
        public void Clear()
        {
            lock (_lock) { _current = null; }
        }
    }
}
