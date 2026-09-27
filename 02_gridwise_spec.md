# GridWise — LLM-Assisted Smart Campus Energy Optimizer

**What it is:** A production HTTP API service that receives 24-hour campus
energy data and unstructured human notes, then returns a mathematically
optimal hourly energy schedule that minimizes electricity cost.

## Tech stack

- Language: Python 3.11
- Web framework: FastAPI with Pydantic v2 for schema validation
- LLM: Google Gemini Flash (via `google-genai` SDK) — the only AI/ML component
- Optimizer: PuLP (Linear Programming library) solved with COIN-OR CBC solver
- Deployment: Render (live URL), Docker container on GitHub Container
  Registry (GHCR)
- CI/CD: GitHub Actions to auto-build and push Docker image to GHCR on
  every push to `main`

## The 3-stage pipeline

**Stage 1 — LLM Directive Interpreter (`interpreter.py`)**
- Receives 1–3 natural language operator notes (e.g. "Solar output drops
  20% from 1 PM to 3 PM", "Don't charge battery between 2–4 PM")
- Sends them to Gemini Flash with battery capacity context in the prompt
- Gemini returns each note classified as one of: `solar_reduction`,
  `no_charge_window`, `no_discharge_window`, `minimum_battery_reserve`,
  or `no_op`, each with `applies` (bool), `directive_type`,
  `structured_adjustment` (hours array + optional factor), and `explanation`
- Model discovery is cached at startup to keep latency under 1.5s

**Stage 2 — Deterministic Guardrail Validator (`guardrails.py`)**
- Clamps hours to unique integers in `[0..23]` ascending order
- Clamps solar reduction factors to `[0.0..1.0]`
- Converts a `minimum_battery_reserve` fraction ≤1.0 into an absolute kWh
  value (handles notes like "keep 50% in reserve")
- Enforces `applies=false` and `structured_adjustment=null` for `no_op`
- Ensures exactly one directive per input note (fills missing as `no_op`)

**Stage 3 — Mathematical LP/MILP Optimizer (`optimizer.py`)**
- PuLP MILP formulation. Decision variables per hour (0..23): `grid_kwh`,
  `solar_used_kwh`, `battery_charge_kwh`, `battery_discharge_kwh`, binary
  `is_charging`, binary `is_discharging`
- Objective: minimize Σ(`grid_kwh[h]` × `tariff[h]`) across 24 hours, with
  a secondary tie-breaker of `1e-4 × peak_grid_kwh` to flatten demand
  spikes during flat-tariff windows
- Constraints per hour: energy balance (`grid + solar_used +
  battery_discharge = demand + battery_charge`), battery state of charge
  bounds, inverter charge/discharge rate limits, mutual exclusivity of
  charge/discharge, solar cap, end-of-day neutrality
  (`E[23] = initial_energy_kwh`)
- Operator directives become additional constraints (solar reduction caps,
  no-charge/no-discharge windows, minimum reserve floors)
- Solved via CBC in under 20 milliseconds

## API

- `GET /health` → `{"status": "ok"}`
- `POST /optimize-energy` → takes `scenario_id`, `operator_notes[]`,
  `hours[]` (`hour`, `demand_kwh`, `solar_kwh`, `tariff_bdt_per_kwh` —
  **must be exactly 24 entries, hours 0-23, or the request is rejected**),
  and `battery` — returns `directive_interpretation[]`, `hourly_plan[]`
  (`grid_kwh`, `solar_used_kwh`, `battery_action`, `battery_kwh`,
  `battery_energy_after_kwh`), `total_grid_kwh`, `total_cost_bdt`,
  `peak_grid_kwh`, `plan_summary`

  **`battery` object — CONFIRMED AGAINST THE LIVE DEPLOYED API (Sept 2026),
  not the original spec text below, which was wrong:**
  `capacity_kwh`, `initial_energy_kwh`, `minimum_energy_kwh` (required,
  not mentioned in the original spec at all), `max_charge_kwh_per_hour`
  (NOT `max_charge_rate_kw` as originally documented),
  `max_discharge_kwh_per_hour` (NOT `max_discharge_rate_kw`). If GridWise's
  schema changes again, re-verify with a raw curl request rather than
  trusting this doc or the original spec text blindly — this mismatch
  is exactly how the dashboard would have silently failed every
  GridWise call without anyone noticing until you tested it live.

## Files

`main.py` (FastAPI app, routes, exception handlers) · `schemas.py`
(Pydantic v2 models) · `interpreter.py` (Gemini call, prompt formatting,
JSON extraction, model caching) · `guardrails.py` (deterministic
sanitization) · `optimizer.py` (PuLP formulation + solve) ·
`requirements.txt` (fastapi, uvicorn, pydantic>=2.6.0, pulp>=2.8.0,
google-genai>=0.1.1, requests) · `Dockerfile` (installs `coinor-cbc`,
runs uvicorn on port 8000) · `.github/workflows/docker.yml` (build + push
to `ghcr.io/<username>/gridwise-hackathon-2026:latest` on push to `main`)

## Known bugs found in production (from judge logs)

- `validation_exception_handler` in `main.py` includes raw Python
  `ValueError` objects in the error dict, which aren't JSON serializable —
  causes 500 instead of 400 on malformed input. **Fix:** convert
  `exc.errors()` entries to strings before passing to `JSONResponse`.
- The optimizer crashes with `PulpSolverError` on infeasible scenarios
  (contradictory constraints). **Fix:** catch the solver error and return
  a descriptive 422 response instead of 500.

## Environment / deployment

- `GEMINI_API_KEY` required (never commit this)
- Live on Render, auto-deploys from GitHub `main`
- Docker: `docker run -d -p 8000:8000 -e GEMINI_API_KEY=xxx
  ghcr.io/<user>/gridwise-hackathon-2026:latest`

## Performance achieved

- 10/10 official benchmark cases passed
- 1.0000 cost quality ratio (exact match to organizer reference)
- P95 latency: ~1.5s under normal load, ~9s on Render cold start

## Integration note (added during this project's planning)

This service needs `CORSMiddleware` added to `main.py` before a browser
dashboard can call `/optimize-energy` directly — FastAPI blocks
cross-origin requests by default. Not yet added as of this file's writing.
