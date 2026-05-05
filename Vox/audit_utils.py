# ── audit_utils.py: compact audit helpers ────────────────────────────────────
#
# Extracted from journals.py to a tiny stable module so that QuantConnect
# imports survive even if journals.py is large or stale.
#
# Usage:
#   from audit_utils import audit_safe_float, audit_trim_votes
# ─────────────────────────────────────────────────────────────────────────────


def audit_safe_float(v, digits=6):
    """Return round(float(v), digits) or None on failure."""
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


def audit_trim_votes(votes):
    """Convert vote dict values to compact floats; skip non-numeric entries."""
    if not isinstance(votes, dict):
        return {}
    out = {}
    for k, v in votes.items():
        f = audit_safe_float(v, 4)
        if f is not None:
            out[str(k)] = f
    return out
