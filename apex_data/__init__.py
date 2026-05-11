"""apex_data — offline data fetchers for Apex.

These scripts run locally (or in a GitHub Action). They produce static
CSVs that get pushed to the QC project root via qc_runner. They DO NOT
import QC or Pulse at runtime; they only need stdlib + urllib.

Usage:
    python3 apex_data/fetch_etf_flows.py        → apex_etf_flows.csv
    python3 apex_data/fetch_token_unlocks.py    → apex_token_unlocks.csv
    python3 apex_data/refresh_all.py            → run all fetchers in sequence
"""
