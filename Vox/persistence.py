# ── Vox Persistence ───────────────────────────────────────────────────────────
#
# Model serialisation and trade logging via the QC ObjectStore.
# ─────────────────────────────────────────────────────────────────────────────

import json
import pickle


class PersistenceManager:
    """
    Persist ML models and trade logs to the QuantConnect ObjectStore.

    ObjectStore is the only durable storage available inside the QC sandbox.
    All write/read operations are wrapped in try/except so that any storage
    failure degrades gracefully (logs a warning) rather than crashing the algo.

    Kill switch
    -----------
    If the key ``kill_key`` exists in the ObjectStore (any content), the
    strategy will refuse to open new positions.  To activate from the QC
    Research environment::

        qb.object_store.save("vox/kill_switch", "1")

    To deactivate::

        qb.object_store.delete("vox/kill_switch")

    Parameters
    ----------
    algorithm  : QCAlgorithm — the running algorithm instance.
    model_key  : str — ObjectStore key for the pickled model.
    log_key    : str — ObjectStore key for the JSONL trade log.
    kill_key   : str — ObjectStore key for the kill switch flag.
    """

    def __init__(
        self,
        algorithm,
        model_key  = "vox/model.pkl",
        log_key    = "vox/trade_log.jsonl",
        kill_key   = "vox/kill_switch",
    ):
        self._algo      = algorithm
        self._model_key = model_key
        self._log_key   = log_key
        self._kill_key  = kill_key

    # ── Model serialisation ───────────────────────────────────────────────────

    def save_model(self, model):
        """
        Pickle *model* and write the bytes to the ObjectStore.

        Parameters
        ----------
        model : object — any pickleable sklearn-compatible model.
        """
        try:
            data = pickle.dumps(model)
            self._algo.object_store.save_bytes(self._model_key, data)
            self._algo.log(
                f"[persistence] Model saved to ObjectStore key={self._model_key}"
            )
        except Exception as exc:
            self._algo.log(f"[persistence] save_model failed: {exc}")

    def load_model(self):
        """
        Load and unpickle the model from the ObjectStore.

        Returns
        -------
        object or None
            The deserialised model, or ``None`` if not found or on error.
        """
        try:
            if not self._algo.object_store.contains_key(self._model_key):
                return None
            data = self._algo.object_store.read_bytes(self._model_key)
            model = pickle.loads(data)
            self._algo.log(
                f"[persistence] Model loaded from ObjectStore key={self._model_key}"
            )
            return model
        except Exception as exc:
            self._algo.log(f"[persistence] load_model failed: {exc}")
            return None

    # ── Trade logging ─────────────────────────────────────────────────────────

    def log_trade(self, entry_dict):
        """
        Append *entry_dict* as a JSON line to the trade log in ObjectStore.

        The log is append-only: each call reads the existing content, appends
        one line, and writes the result back.  This is safe for the single-
        threaded QC environment.

        Parameters
        ----------
        entry_dict : dict — trade record to log (will be JSON-serialised).
        """
        try:
            existing = ""
            if self._algo.object_store.contains_key(self._log_key):
                existing = self._algo.object_store.read(self._log_key)
            new_line = json.dumps(entry_dict, default=str)
            self._algo.object_store.save(
                self._log_key, existing + new_line + "\n"
            )
        except Exception as exc:
            self._algo.log(f"[persistence] log_trade failed: {exc}")

    # ── Kill switch ───────────────────────────────────────────────────────────

    def is_kill_switch_active(self):
        """
        Return True if the kill switch key exists in the ObjectStore.

        Returns
        -------
        bool
        """
        try:
            return self._algo.object_store.contains_key(self._kill_key)
        except Exception as exc:
            self._algo.log(
                f"[persistence] is_kill_switch_active check failed: {exc}"
            )
            return False
