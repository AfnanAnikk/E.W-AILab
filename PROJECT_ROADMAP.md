# Campus Energy Project — Roadmap

Full picture, what's done, what's left. Keep this file updated as things
change — treat it as the source of truth if context/tokens run out and you
need to start a fresh chat and re-orient (me or a future version of me).

## The system, end to end

```
SimCity sim (dummy buildings/HVAC/weather, live in a dashboard)
        |
        +--> XGBoost Regressor  -> next-hour demand per building
        +--> XGBoost Classifier -> risk tier per building (Normal/Watch/High-Risk/Shed-Now)
        |
        v
Aggregate campus demand + solar + tariff + battery state
        |
        v
GridWise /optimize-energy  (Gemini directive interpreter + PuLP LP solver)
        |
        v
Dashboard: hourly grid/battery/solar plan, cost, which buildings get shed,
operator can type free-text notes -> flows into GridWise's operator_notes
```

Three components make decisions; only two are models you train (the
regressor + classifier). GridWise's Gemini call and its LP solver are
already-built, no training needed there.

## Status

### DONE
- [x] Dataset sourced and confirmed real: BDG2 (Building Data Genome 2.0),
      your local copy at `/content/drive/MyDrive/Electricity/`, 747MB, all
      8 meter types + weather + metadata.
- [x] `train_models.py` written, tested against real BDG2 headers, and run
      successfully on your actual data (30 buildings, 421,195 rows).
- [x] Bugs found and fixed in the training script:
      metadata column collision, sklearn API changes, risk-tier label
      scrambling (LabelEncoder issue), random vs. chronological train/test
      split (the last one changes your regressor's real R² — re-run needed).
- [x] Regressor trained once (pre-chronological-split numbers: MAE 5.676,
      RMSE 10.271, R²=0.994 — these need to be regenerated with the fixed
      split, see Immediate Next Steps).
- [x] Classifier trained once but with the label-scrambling bug still
      active — those numbers are invalid, needs a clean re-run.
- [x] GridWise itself: already built and working by you separately
      (FastAPI + Gemini directive interpreter + PuLP LP optimizer,
      10/10 benchmark cases passed).

### IMMEDIATE NEXT STEPS (in order)
1. **Re-run all 4 cells of `train_models.py` in Colab**, top to bottom,
   fresh. This regenerates `data` with the chronological split and trains
   both models on corrected labels. Send back: the row count, the risk
   tier distribution, the regressor's MAE/RMSE/R² + feature importances,
   and the classifier's accuracy/F1/classification report.
2. **Sanity-check the regressor's feature importances** once you have
   them — confirm HVAC/weather features are actually contributing, not
   just `lag_1h` doing persistence forecasting. If `lag_1h` totally
   dominates, that's not wrong, but it means the "equipment health"
   framing from your proposal isn't really driving predictions yet, and
   we may want to try a model without `lag_1h` as a comparison.
3. **Decide if N_BUILDINGS=30 is enough, or scale up.** 30 buildings
   already gave 421,195 rows, which is plenty to train on — but if you
   want the SimCity dashboard to feel like a real ~75-building campus
   (matching your original proposal doc), you'll want more buildings
   represented, even if the model itself doesn't need them all.

### IN PROGRESS / PICK BACK UP HERE
- [ ] **Dashboard is currently flat-tile, not isometric.** You asked for
      an actual SimCity-look (isometric buildings, not flat colored
      squares/rectangles) and that rewrite was started but not finished
      or tested — reverted back to the working flat roof+facade version
      so the file wasn't left broken. The approach for next time: classic
      2:1 diamond-grid isometric projection —
      `isoX = (col - row) * TILE_W/2`, `isoY = (col + row) * TILE_H/2`,
      each building as a 3-face cube (`face-top`/`face-left`/`face-right`
      divs using `clip-path` diamonds/parallelograms, different
      `brightness()` filter per face for fake lighting), positioned
      absolutely on a ground layer of diamond street tiles. Get the tile
      and cube proportions consistent (same W:H ratio throughout) before
      wiring it to real data again, and test structurally via the jsdom
      approach used elsewhere in this file's history before considering
      it done — I can't visually render CSS myself, only verify structure
      and that classes/positions get set as expected, so a fresh attempt
      needs the same careful testing the rest of `dashboard.html` got.

### NOT STARTED YET
- [ ] **Model-serving endpoint.** XGBoost models can't run in browser JS.
      Need a small FastAPI/Flask wrapper that loads `regressor_demand.json`
      and `classifier_risk_tier.json` and exposes a `/predict` endpoint.
      I can write this once training is finalized — it's a short script,
      maybe 60-80 lines.
- [ ] **GridWise CORS setup.** FastAPI blocks cross-origin browser requests
      by default. Add `CORSMiddleware` to GridWise's `main.py` allowing
      the origin the dashboard will run from (or `allow_origins=["*"]`
      for a demo). Needed before the dashboard can actually call GridWise
      from a browser.
- [ ] **The SimCity dashboard itself.** Standalone HTML/JS file (not a
      published claude.ai artifact — that sandbox can't reach your local
      APIs). Plan agreed so far:
      - ~20-30 building grid, clean ops-dashboard look
      - Autonomous hourly tick: generates dummy demand/weather, calls the
        model-serving endpoint for regressor+classifier predictions, calls
        GridWise's `/optimize-energy` with the aggregated result
      - Operator panel: pause/resume, override a building's risk tier,
        free-text notes box that feeds GridWise's `operator_notes`
      - Local fallback heuristic if GridWise/model server is unreachable,
        so the demo still runs standalone
      - You said "almost no involvement from me except as operator if
        needed" — so default to fully autonomous, operator panel is
        there but doesn't block the loop
- [ ] **Wiring real solar data in.** `train_models.py` already splits
      solar out into `solar_for_gridwise.csv` during data loading — this
      needs to actually get read by the dashboard/model-server and fed
      into GridWise's `solar_kwh` field instead of a simulated curve.
- [ ] **End-to-end test**: dashboard -> model server -> GridWise -> back
      to dashboard, one full loop, before calling any of this "working."

### PROJECT DELIVERABLES (from your original AI Lab proposal doc)
Worth keeping in view since these are what actually get graded/submitted:
- [ ] Working code and model — built and tested against simulated or
      sample building data (mostly on track — real BDG2 data, not just
      simulated, which is stronger than what the proposal asked for)
- [ ] User manual — clear enough to configure and operate without
      hand-holding (not started)
- [ ] As-built report — documents the design decisions for whoever
      builds on this next (not started; this roadmap file is a good seed
      for it)
- [ ] Version-controlled codebase — delivered on GitHub or equivalent
      (worth setting up a repo now if you haven't, so training script +
      GridWise + dashboard all live together)

## Known open questions / things to decide later
- Risk-tier thresholds (75th/90th/97th percentile) are a starting guess,
  not tuned against real operational judgment — revisit once you've seen
  how the tiers actually distribute across a full campus.
- Whether to scale N_BUILDINGS beyond 30 for training, and separately,
  how many buildings the dashboard visually represents (these don't have
  to match).
- Whether GridWise gets deployed live (Render) for the dashboard to hit,
  or run locally via Docker during development/demo.

## File map (what's been created so far)
- `train_models.py` — the training script, canonical, single file,
  changelog at the top tracks every fix. Run in Colab, 4 cells marked
  `===== CELL N =====` inside the file.
- `model_server.py` — FastAPI wrapper serving the two trained models over
  HTTP (`/predict`, `/predict/batch`). Tested end-to-end against real
  dummy-trained models, not just written. Run with
  `python model_server.py`, needs `regressor_demand.json` and
  `classifier_risk_tier.json` (from running `train_models.py`) in the
  same folder.
- `dashboard.html` — the SimCity-style frontend + dashboard. Standalone,
  open directly in a browser, no build step. Currently flat-tile visuals
  (see "IN PROGRESS" above for the isometric rewrite that's still owed).
  Autonomous hourly loop, calls `model_server.py` + GridWise, operator
  override panel, local fallback heuristic if either server's down.
- `docs/01_project_proposal.md` — your original AI Lab proposal doc,
  saved as text so it survives even if chat history gets cut.
- `docs/02_gridwise_spec.md` — the GridWise spec doc, same reason. Has a
  note at the bottom flagging the CORS fix GridWise's `main.py` still
  needs before the dashboard can call it directly.
- This file (`PROJECT_ROADMAP.md`) — status + next steps, keep it updated.

## What you still need to run/do on your end (not done yet)
1. Re-run all 4 cells of `train_models.py` fresh (chronological split fix
   needs a clean run — see "Immediate Next Steps" above).
2. Add `CORSMiddleware` to GridWise's `main.py` (see `docs/02_gridwise_spec.md`).
3. Run `model_server.py` locally with the freshly trained model files.
4. Open `dashboard.html`, point it at both running servers, confirm the
   full loop actually works end to end.
5. Decide if/when to pick the isometric visual rewrite back up.
