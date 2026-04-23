from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


class ModelRegistry:
    def __init__(self, root="models/artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _config_hash(config):
        payload = json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def save(self, model, symbol, config=None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sym = symbol.replace("/", "_")
        cfg_hash = self._config_hash(config)
        path = self.root / f"{sym}_{ts}_{cfg_hash}.joblib"
        model.save(path)
        meta = {
            "symbol": symbol,
            "timestamp": ts,
            "config_hash": cfg_hash,
            "path": str(path),
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def load_latest(self, symbol):
        sym = symbol.replace("/", "_")
        candidates = sorted(self.root.glob(f"{sym}_*.joblib"))
        if not candidates:
            raise FileNotFoundError(f"No model found for {symbol}")
        return candidates[-1]
