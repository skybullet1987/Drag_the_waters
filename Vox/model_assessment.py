# model_assessment.py — Per-model accuracy assessment for backtest analysis
#
# After a gatling-mode backtest, use this module to evaluate which individual
# models in the ensemble have genuine predictive power.
#
# Usage (from QuantConnect Research notebook or offline):
#
#   from model_assessment import compute_model_accuracy, rank_models, format_assessment_report
#
#   trades = [...]  # list of dicts with active_votes, realized_return, winner
#   accuracy = compute_model_accuracy(trades, vote_threshold=0.50)
#   ranked = rank_models(accuracy, by="profit_factor")
#   print(format_assessment_report(accuracy))


def compute_model_accuracy(trades, vote_threshold=0.50):
    """Compute per-model accuracy statistics from completed trades.

    Parameters
    ----------
    trades : list[dict]
        Each dict must have:
          - ``active_votes``: dict[str, float] — per-model probabilities
          - ``realized_return``: float — actual return of the trade
          - ``winner``: bool — True if trade was profitable
    vote_threshold : float
        Probability threshold for counting a model as voting "yes".

    Returns
    -------
    dict[str, dict]
        Per-model accuracy statistics keyed by model ID. Each value dict has:
          - yes_count:           times the model voted yes (P >= threshold)
          - no_count:            times the model voted no
          - total_trades:        total trades where model had a vote
          - win_rate_when_yes:   fraction of wins when model voted yes
          - avg_return_when_yes: mean realized return when model voted yes
          - avg_return_when_no:  mean realized return when model voted no
          - profit_factor:       gross_wins / gross_losses when model voted yes
          - edge_vs_no:          avg_return_when_yes - avg_return_when_no
    """
    if not trades:
        return {}

    model_stats = {}

    for trade in trades:
        votes = trade.get("active_votes") or {}
        ret = trade.get("realized_return", 0.0)
        winner = trade.get("winner", ret > 0)

        for model_id, proba in votes.items():
            if model_id not in model_stats:
                model_stats[model_id] = {
                    "yes_returns": [],
                    "no_returns": [],
                    "yes_wins": 0,
                    "yes_count": 0,
                    "no_count": 0,
                }
            s = model_stats[model_id]
            if proba >= vote_threshold:
                s["yes_count"] += 1
                s["yes_returns"].append(ret)
                if winner:
                    s["yes_wins"] += 1
            else:
                s["no_count"] += 1
                s["no_returns"].append(ret)

    result = {}
    for model_id, s in model_stats.items():
        yes_count = s["yes_count"]
        no_count = s["no_count"]
        total = yes_count + no_count

        yes_returns = s["yes_returns"]
        no_returns = s["no_returns"]

        avg_ret_yes = sum(yes_returns) / len(yes_returns) if yes_returns else 0.0
        avg_ret_no = sum(no_returns) / len(no_returns) if no_returns else 0.0

        win_rate_yes = s["yes_wins"] / yes_count if yes_count > 0 else 0.0

        gross_wins = sum(r for r in yes_returns if r > 0)
        gross_losses = abs(sum(r for r in yes_returns if r < 0))
        pf = gross_wins / gross_losses if gross_losses > 0 else (
            float("inf") if gross_wins > 0 else 0.0
        )

        result[model_id] = {
            "yes_count": yes_count,
            "no_count": no_count,
            "total_trades": total,
            "win_rate_when_yes": win_rate_yes,
            "avg_return_when_yes": avg_ret_yes,
            "avg_return_when_no": avg_ret_no,
            "profit_factor": pf,
            "edge_vs_no": avg_ret_yes - avg_ret_no,
        }

    return result


def rank_models(accuracy, by="profit_factor"):
    """Rank models by a given metric.

    Parameters
    ----------
    accuracy : dict[str, dict]
        Output from compute_model_accuracy.
    by : str
        Metric key to sort by (descending). Common choices:
        "profit_factor", "win_rate_when_yes", "avg_return_when_yes", "edge_vs_no"

    Returns
    -------
    list[tuple[str, dict]]
        Sorted (model_id, stats_dict) pairs, best first.
    """
    items = list(accuracy.items())
    items.sort(key=lambda kv: kv[1].get(by, 0.0), reverse=True)
    return items


def format_assessment_report(accuracy, min_trades=3):
    """Format a human-readable model assessment report.

    Parameters
    ----------
    accuracy : dict[str, dict]
        Output from compute_model_accuracy.
    min_trades : int
        Only include models with at least this many "yes" votes.

    Returns
    -------
    str
        Formatted report string.
    """
    if not accuracy:
        return "No model accuracy data available."

    lines = ["=== Model Assessment Report ===", ""]
    header = f"{'Model':<15} {'Yes':>4} {'WR%':>6} {'AvgRet':>8} {'PF':>8} {'Edge':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    ranked = rank_models(accuracy, by="profit_factor")
    for model_id, stats in ranked:
        if stats["yes_count"] < min_trades:
            continue
        wr = stats["win_rate_when_yes"] * 100
        ar = stats["avg_return_when_yes"] * 100
        pf = stats["profit_factor"]
        edge = stats["edge_vs_no"] * 100
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        lines.append(
            f"{model_id:<15} {stats['yes_count']:>4} {wr:>5.1f}% {ar:>+7.2f}% "
            f"{pf_str:>8} {edge:>+7.2f}%"
        )

    lines.append("")
    lines.append("WR% = win rate when model voted yes")
    lines.append("AvgRet = average return when model voted yes")
    lines.append("PF = profit factor (gross_wins / gross_losses) when yes")
    lines.append("Edge = avg_return_when_yes - avg_return_when_no")
    return "\n".join(lines)
