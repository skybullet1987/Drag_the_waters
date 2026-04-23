import importlib.util
from pathlib import Path


def _load_alt_data_module():
    path = Path(__file__).resolve().parents[1] / "alt-data.py"
    spec = importlib.util.spec_from_file_location("alt_data_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyClient:
    def fetch_24h_volume(self):
        return {
            "XBTUSD": 300_000_000,
            "ETHUSD": 200_000_000,
            "TRUMPUSD": 400_000_000,
            "SOLUSD": 150_000_000,
            "ADAUSD": 40_000_000,
            "BTCUSDT": 999_000_000,
        }


def test_select_top_universe_filters_and_orders():
    mod = _load_alt_data_module()
    selected = mod.select_top_universe(DummyClient(), top_n=3, min_volume_usd=50_000_000, quote="USD", exclude_meme=True)
    assert selected == ["XBTUSD", "ETHUSD", "SOLUSD"]
