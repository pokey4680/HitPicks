# HitPicks backend

Python prediction engine and FastAPI HTTP layer for HitPicks. See the
[project README](../README.md) for the full model documentation, calibration
results, and design philosophy.

## Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI

```bash
bts predict                      # Today's top picks
bts predict --top 10             # Top 10
bts predict --min-prob 0.75      # Picks above 75%
bts predict --date 2025-07-04    # Historical date

bts backtest --start 2025-05-01 --end 2025-08-31
bts strategy --window 2024-05-01:2024-08-31 --window 2025-05-01:2025-08-31

bts cache warm --season 2025     # One-time, ~15 min
bts cache status
```

## Tests

```bash
pytest
```

## HTTP layer

Coming soon — FastAPI service that wraps `predict`, `backtest`, `strategy`
for the frontend to consume.
