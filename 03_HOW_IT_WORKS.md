# Campus Energy Project — How It Works

A complete explanation of the system: the problem it solves, every
component, how they connect, what's actually finished versus still in
progress, and known limitations. Written to be presented to faculty as-is.

---

## 1. The problem this solves

Bangladesh's power grid routinely can't cover peak demand — generation
falls short and the grid cuts service (load shedding) to cope, rather than
this being an occasional emergency. A campus with dozens of buildings has
no early warning: it finds out it's being shed at the same moment
everyone else does.

This project builds a system that watches a campus's power and HVAC data
continuously, predicts which buildings are heading toward trouble *before*
an outage notice arrives, and decides how to respond (shift load to
battery/solar, flag buildings for attention) rather than reacting blind.

## 2. The system, end to end

```
SimCity-style dashboard (simulated campus: buildings, HVAC, weather)
        |
        |  every simulated hour, for every building:
        v
Two trained ML models (via a small API server)
        +--> Regressor:  predicts next-hour electricity demand (kWh)
        +--> Classifier: predicts a risk tier (Normal/Watch/High-Risk/Shed-Now)
        |
        v
Aggregate campus demand + solar + tariff + battery state, for all 24 hours
        |
        v
GridWise (separate service): Gemini-based note interpreter + a real
linear-programming optimizer, decides the actual hour-by-hour grid/
battery/solar dispatch plan that minimizes cost
        |
        v
Dashboard shows the plan, the cost, which buildings are at risk; a human
operator can watch, override a building's tier, or type a plain-English
note ("don't charge the battery 2-4pm") that GridWise actually obeys
```

Three things make decisions in this pipeline. Only two of them are
machine learning models that this project trained — the regressor and
classifier. GridWise's Gemini call uses an already-trained model (Google's,
called via API, not trained by us), and its dispatch decision is solved by
a linear program (PuLP + CBC solver) — exact mathematical optimization,
not a learned model at all. This matters for grading purposes: the "AI"
in the system is a mix of two custom-trained models, one already-trained
LLM used as a tool, and one classical optimization algorithm, each doing
the part it's actually good at.

## 3. The dataset

**Building Data Genome 2.0 (BDG2)** — a real, public dataset of hourly
meter readings from real buildings (not synthetic), covering:
electricity, chilled water, steam, hot water, gas, water, irrigation, and
solar generation, plus weather and building metadata (square footage,
building type, site). ~1,600 buildings across multiple real campuses,
2016-2017.

This was a deliberate upgrade over the proposal's original plan to use
"simulated or sample building data" — training on real building behavior
is a stronger foundation than synthetic data would have been.

## 4. The two trained models (`train_models.py`)

Both trained on the same feature set, built from the same BDG2 data —
electricity is the target for both, but chilled water / steam / hot
water / gas readings are included as features (lag values and rolling
volatility), because sustained high HVAC flow with little variation is
exactly the "condenser never cycling down" signal called out in the
original proposal as something a static rulebook can't catch but a model
can.

**Model 1 — XGBoost Regressor.** Predicts next-hour electricity demand in
kWh for a given building. This is the forecasting engine — its output
becomes the `demand_kwh` GridWise optimizes against.

*Actual result on real data (30 buildings, 421,195 rows, proper
chronological train/test split — trained on each building's earlier 80%
of hours, tested on its later 20%, so this is a genuine forecast of
unseen future hours, not an inflated score):* MAE 5.97, RMSE 10.98,
R²=0.993. Feature importance shows the model relies almost entirely
(~98%) on recent demand history (`lag_1h`, `lag_24h`,
`rolling_mean_24h`) rather than HVAC/weather signal — a legitimate and
expected result for demand forecasting (recent demand genuinely is the
strongest predictor), but worth stating plainly: this particular model's
accuracy isn't coming from the "equipment health" signals the proposal
emphasized, even though those features are available to it.

**Model 2 — XGBoost Classifier.** Predicts a risk tier per building per
hour: `Normal` (below the 75th percentile of that building's own
historical demand) / `Watch` (75th-90th) / `High-Risk` (90th-97th) /
`Shed-Now` (top 3%). This is the decision-support signal — it's what
would actually tell an operator which buildings need attention.

*Actual result:* accuracy 0.826, but that number is misleading on its
own — it's inflated by the dominant `Normal` class. The metric that
matters here is recall on the critical classes, and the first trained run
showed only 0.20-0.23 recall on `Shed-Now`/`High-Risk` — meaning it
missed roughly 4 out of 5 real critical events, which defeats the
system's actual purpose. A fix (class-balanced sample weighting, so
misclassifying a rare/critical class costs the model more during
training) is implemented in the current `train_models.py` but has not yet
been re-run to confirm improved numbers — **this is an honest, documented
known limitation, not a hidden gap.**

Both models are trained in Google Colab (`train_models.py`, split into 4
cells: config/imports, load+feature-engineer, train regressor, train
classifier) and exported as `regressor_demand.json` /
`classifier_risk_tier.json`.

## 5. The model server (`model_server.py`)

XGBoost models can't run inside a web browser. This is a small FastAPI
service that loads both trained models and exposes them over HTTP:
- `POST /predict` — one building's current readings in, a demand
  prediction + risk tier out
- `POST /predict/batch` — the whole campus in one call (what the
  dashboard actually uses each simulated hour, rather than one HTTP
  request per building)

Includes a safety clamp: raw regressor output is floored at 0, since a
physical demand value can never be negative even if the underlying model
extrapolates there on unusual input.

## 6. GridWise (separate service, already built)

A production-grade optimizer, built independently as its own project. Not
built as part of this dashboard/model work, but integrated with it. Three
stages: (1) an LLM (Gemini) reads plain-English operator notes like
"don't charge the battery 2-4pm" and turns them into structured
constraints; (2) a deterministic validator sanitizes whatever the LLM
returned so malformed output can't reach the solver; (3) a real linear
program (PuLP, solved with the CBC solver) finds the mathematically
optimal hour-by-hour grid/battery/solar dispatch that minimizes cost,
subject to battery physics (charge/discharge limits, state-of-charge
bounds) and whatever constraints the operator's notes added. Solves in
under 20 milliseconds. Independently benchmarked at 10/10 correct on
official test cases.

## 7. The dashboard (`dashboard.html`)

A standalone, self-contained web page (open directly in any browser, no
install/build step) that acts as the "SimCity" simulation and the
operator's control panel in one. Runs an autonomous loop, once per
simulated hour: generates plausible dummy readings for ~24 buildings,
calls the model server for predictions, aggregates the campus total, and
sends it to GridWise for an optimized dispatch plan — all without
requiring the operator to do anything. The operator can watch, pause,
click into any building to force-override its risk tier, or type a note
that flows through to GridWise's Gemini interpreter. If either the model
server or GridWise is unreachable, it falls back to a simple local
heuristic so the demo keeps running rather than freezing.

**Current known issues in the dashboard, honestly stated:**
- The dummy data generator's demand scale was never calibrated against
  the real BDG2 data's actual scale — this causes unrealistic
  predictions (including the negative-demand issue that prompted the
  model-server clamp fix) because the model is being fed inputs far
  outside what it saw in training. Needs the dummy generator's baseline
  ranges adjusted to match real building demand magnitudes.
- GridWise connectivity has been intermittently failing in testing
  (dashboard shows "gridwise: down") — root cause not yet confirmed,
  most likely either the CORS fix not being live on the deployed Render
  service, or a `file://` origin issue from opening the HTML file
  directly rather than serving it. Needs a browser console check to
  confirm which.
- Visual style is currently flat colored building blocks (color = risk
  tier, height = demand), not a true isometric "game" look. An isometric
  rewrite was attempted and intentionally reverted rather than shipped
  broken — the approach is documented in `PROJECT_ROADMAP.md` for
  whoever picks it back up.

## 8. What's genuinely tested versus what isn't

Worth being precise about this for a faculty presentation, since it's the
difference between "I wrote code that should work" and "I verified it
works":
- `model_server.py` was tested with real HTTP requests against real
  (dummy-trained, for testing purposes) models — health check, single
  prediction, and batch prediction all confirmed returning correctly
  shaped responses.
- `dashboard.html`'s JavaScript logic was tested in a headless browser
  environment (not just visually eyeballed) — confirmed it correctly
  calls both servers, falls back cleanly when they're unreachable,
  renders building data driven by real model output, and that operator
  overrides persist correctly across the autonomous loop's ticks.
- `train_models.py` has been run once successfully on the real, full
  BDG2 dataset (421,195 rows) by the student, producing the actual
  metrics quoted in Section 4.
- GridWise's live deployment has been confirmed reachable and correctly
  validating requests (via direct curl testing), which is also how a
  real schema mismatch between the documented spec and the live API was
  caught and fixed before it could cause a silent failure in the
  dashboard.

## 9. Deliverables checklist (against the original proposal)

- [x] Working code and model — built and tested against real building
      data (stronger than the proposal's baseline ask of simulated data)
- [ ] User manual — not yet written as a separate document
- [x] As-built report — this document
- [ ] Version-controlled codebase on GitHub — not yet set up as a repo
