# ── Vox Sizing ────────────────────────────────────────────────────────────────
#
# Position-sizing helpers using fractional Kelly criterion.
# ─────────────────────────────────────────────────────────────────────────────


def kelly_fraction(p, tp, sl, kelly_frac=0.25, max_alloc=0.80):
    """
    Compute the fractional-Kelly allocation for a long trade.

    Full-Kelly formula for a binary bet with win-probability *p*::

        b       = tp / sl          (payoff ratio)
        f_full  = (p × (b + 1) − 1) / b

    The result is then scaled by *kelly_frac* (quarter-Kelly by default) and
    clamped to [0, max_alloc].

    Parameters
    ----------
    p          : float  — model probability P(win) in (0, 1)
    tp         : float  — take-profit fraction (e.g. 0.020)
    sl         : float  — stop-loss fraction   (e.g. 0.012)
    kelly_frac : float  — fractional-Kelly multiplier (default 0.25)
    max_alloc  : float  — hard ceiling on allocation  (default 0.80)

    Returns
    -------
    float
        Recommended allocation as a fraction of portfolio value, in [0, max_alloc].
    """
    b      = tp / sl
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, min(f_full * kelly_frac, max_alloc))


def compute_qty(
    mean_proba,
    tp,
    sl,
    price,
    portfolio_value,
    kelly_frac,
    max_alloc,
    cash_buffer,
    use_kelly,
    allocation,
):
    """
    Compute the quantity (in coin units) to purchase.

    Sizing logic
    ────────────
    1. If *use_kelly* is True, compute the Kelly allocation.
       - If the Kelly fraction is <= 0 (edge is negative), fall back to
         the flat *allocation*.
    2. If *use_kelly* is False, use the flat *allocation* directly.
    3. Apply *cash_buffer* to leave a small cash headroom for fees.
    4. Convert the dollar value to coin units.

    Parameters
    ----------
    mean_proba      : float — ensemble mean P(win)
    tp              : float — take-profit fraction
    sl              : float — stop-loss fraction
    price           : float — current coin price
    portfolio_value : float — total portfolio equity
    kelly_frac      : float — fractional-Kelly multiplier
    max_alloc       : float — hard ceiling on allocation
    cash_buffer     : float — cash headroom multiplier (e.g. 0.99)
    use_kelly       : bool  — True to use Kelly, False for flat sizing
    allocation      : float — fallback flat allocation fraction

    Returns
    -------
    tuple[float, float]
        ``(qty, alloc_fraction)`` where *qty* is in coin units and
        *alloc_fraction* is the fraction of portfolio value actually used.
    """
    if use_kelly:
        alloc = kelly_fraction(mean_proba, tp, sl, kelly_frac, max_alloc)
        if alloc <= 0.0:
            alloc = allocation   # negative Kelly edge → flat fallback
    else:
        alloc = allocation

    dollar_value = portfolio_value * alloc * cash_buffer
    qty          = dollar_value / price if price > 0 else 0.0

    return qty, alloc
