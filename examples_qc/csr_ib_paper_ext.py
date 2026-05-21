# QuantConnect: upload with main.py + csr_ml_overlay.py for IBKR paper / live (EOD).
# Set ACTIVE_BASELINE = "ib_paper" or research_preset=ib_paper on QC cloud.


def apply_ib_paper_profile(algo):
    """
    IB paper / live-shaped stack: ML overlay v2 defensive sizing + production_safe execution.
    Does NOT enable maximize same-bar (backtest-only).
    """
    from csr_ml_overlay import wire_ml_overlay
    from csr_profiles import apply_production_safe_profile

    wire_ml_overlay(algo)
    algo.ib_paper_active = True
    algo.Debug(
        "IB_PAPER: ml_overlay v2 + production_safe (EOD next-bar, rails, DD guard)."
    )
    algo.maximize_backtest_equity = False
    algo.ml_overlay_mode = "defensive"
    algo.use_ml_overlay = True
    algo.ml_train_bars = 500
    algo.ml_forward_days = 5
    algo.ml_retrain_days = 63
    algo.ml_l2 = 0.01
    algo.ml_veto_prob = 0.32
    algo.ml_floor_mult = 0.78
    algo.ml_boost_cap = 1.12
    algo.ml_bear_offensive_prob = 0.38
    algo.ml_filter_bear_offensive = True
    algo.disable_bull_uvxy = True
    apply_production_safe_profile(algo)
    algo.maximize_backtest_equity = False
    algo.use_eod_next_bar_execution = True
    algo.max_drawdown_pct = max(
        0.05, min(0.95, float(getattr(algo, "max_drawdown_pct", 0.35)))
    )
    if float(getattr(algo, "max_drawdown_pct", 0.35)) > 0.40:
        algo.max_drawdown_pct = 0.38
    algo.use_vix_delever = True
    algo.vix_delever_ratio = 1.20
    algo.vix_delever_mult = 0.70
    algo.vol_etp_confirm_days = max(int(getattr(algo, "vol_etp_confirm_days", 0)), 1)
    algo.min_hold_days = max(int(getattr(algo, "min_hold_days", 0)), 1)
