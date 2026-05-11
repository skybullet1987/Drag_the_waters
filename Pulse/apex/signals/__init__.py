"""Pulse.apex.signals — pure-Python signal helpers.

Each module exports:
  - compute_X(history) → SignalScore                  (offline-testable)
  - register_X(registry)                              (registers callable)
  - QC adapter classes (only meaningful with HAS_QC)  (live data wiring)
"""
