# EnergyWise: LLM-Assisted Smart Campus Energy Optimization & Microgrid Digital Twin
**Project Architecture, Machine Learning Pipeline, MILP Formulation, and Complete Faculty Defense Guide**  
**Department of Computer Science and Engineering (CSE), United International University (UIU)**  
**Authors:** Juboraz Afnan Mehmud & Team  
**Date:** September 2026  

---

## 1. Executive Summary & Problem Framing

### 1.1 The Real-World Bangladesh Crisis
In Bangladesh, power outages (load-shedding) are not an occasional emergency demand-response event; they are routine grid management. The national peak electricity demand routinely outstrips maximum generation capacity by **1,000 to 1,500 MW**. Power distribution companies—such as the Dhaka Power Distribution Company (DPDC) and Dhaka Electric Supply Company (DESCO)—enforce severe **Time-of-Use (TOU) tariffs**:
- **Off-Peak Tariff (11:00 PM – 5:00 PM / 23:00 – 17:00):** ~7.0 BDT / kWh
- **Peak Hours Tariff (5:00 PM – 11:00 PM / 17:00 – 23:00):** ~12.0 BDT / kWh (a **+71.4% price surge**)

Institutional campuses (universities, medical complexes, research centers) consume vast amounts of electrical and thermal energy across dozens of distinct facilities. Under traditional building management:
1. Campuses behave as **passive consumers**, finding out about feeder disconnects at the exact instant power drops.
2. High-inertia HVAC loops, chillers, and empty lecture halls continue drawing full power during peak tariff windows.
3. Onsite battery energy storage systems (BESS) and rooftop photovoltaic (PV) arrays are dispatched naively or left uncoordinated, failing to protect critical academic and research facilities when grid shedding occurs.

### 1.2 The Solution: A Cyber-Physical Digital Twin
**EnergyWise** is an autonomous, closed-loop Cyber-Physical System (CPS) and digital twin designed to monitor, forecast, optimize, and protect a 24-building university campus in real time. It unifies three computing paradigms:
1. **Supervised Machine Learning (XGBoost):** Trained on 421,195 actual meter readings from real institutional buildings (Building Data Genome 2.0) to predict next-hour load and classify operational risk tiers.
2. **Generative Natural Language Processing (Google Gemini Flash LLM):** Ingests unstructured, conversational operator directives (e.g., *"Don't charge battery between 2-4 PM"* or *"DPDC warning: keep 50% battery in reserve"*) and converts them into structured machine constraints.
3. **Operations Research & Mathematical Optimization (PuLP MILP + COIN-OR CBC):** Formulates and solves a 24-hour Mixed-Integer Linear Program in under 20 milliseconds to guarantee the mathematically optimal grid/solar/battery dispatch that minimizes cost, obeys battery physics, and avoids peak tariffs.
4. **Closed-Loop Protection:** Local microgrid power (solar + battery discharge) is routed to protect at-risk campus buildings, actively preventing blackouts.

---

## 2. System Architecture

The project operates across a continuous four-tier pipeline:

```
[ Tier 1: Real-Time Digital Twin Simulation (dashboard.html) ]
  - 24 Building Twins (Office, Education, Lodging, Retail, Assembly, Public, Warehouse)
  - 8 Sensor Streams (Electricity, Chilled Water, Steam, Hot Water, Gas, Water, Irrigation, Weather)
  - Computes Autoregressive Lags (lag_1h, lag_24h, rolling_mean_24h, rolling_std_6h)
                             │
                             ▼
[ Tier 2: Machine Learning Inference (model_server.py :8001) ]
  - XGBoost Regressor (regressor_demand.json)   --> Next-Hour kWh Forecast (R² = 0.993)
  - XGBoost Classifier (classifier_risk_tier.json) --> Risk Tier (Normal / Watch / High-Risk / Shed-Now)
                             │
                             ▼
[ Tier 3: GridWise Optimization Engine (Render API via proxy.py :8002) ]
  - Stage 1: Google Gemini Flash LLM Directive Interpreter (interpreter.py)
  - Stage 2: Deterministic Guardrails & Normalizer (guardrails.py)
  - Stage 3: PuLP Mixed-Integer Linear Programming Solver (optimizer.py, COIN-OR CBC)
                             │
                             ▼
[ Tier 4: Closed-Loop Dispatch & Microgrid Shielding ]
  - Net Grid Draw Reduction & Hourly Cost Savings
  - Battery State-of-Charge (SoC) Flow Management
  - Critical Building Protection: Offsets load for Shed-Now & High-Risk facilities
```

---

## 3. The 24 Campus Buildings Directory

The campus digital twin models **24 distinct building twins**, cycling across 7 standard building space types and 24 individual building identities.

### 3.1 Complete Building Inventory

| Index | Building Identifier | Space Type | Architectural Role / Campus Function | Baseline Load | Floor Area |
|---|---|---|---|---|---|
| 1 | `Campus_office_Amy` | Office | Central Faculty Offices & Academic Affairs | 85 kWh | 12,400 sq ft |
| 2 | `Campus_lodging_Bea` | Lodging | Student Residential Hall / Dormitory Alpha | 110 kWh | 18,200 sq ft |
| 3 | `Campus_education_Cid` | Education | Main Lecture Hall Complex & Classrooms | 95 kWh | 15,600 sq ft |
| 4 | `Campus_retail_Dan` | Retail | Campus Cafeteria, Dining Hall & Bookstore | 70 kWh | 8,500 sq ft |
| 5 | `Campus_assembly_Eli` | Assembly | Central University Auditorium & Event Arena | 140 kWh | 19,800 sq ft |
| 6 | `Campus_public_Fay` | Public | Central University Library & Study Commons | 80 kWh | 14,000 sq ft |
| 7 | `Campus_warehouse_Gio` | Warehouse | Central Facilities, IT Storage & Maintenance | 45 kWh | 6,200 sq ft |
| 8 | `Campus_office_Hana` | Office | Vice-Chancellor & Administrative Secretariat | 90 kWh | 13,100 sq ft |
| 9 | `Campus_lodging_Ira` | Lodging | International Scholar & Guest Residence | 65 kWh | 9,400 sq ft |
| 10 | `Campus_education_Joy` | Education | Computer Science & Engineering Laboratories | 125 kWh | 17,500 sq ft |
| 11 | `Campus_retail_Kai` | Retail | Student Center & Food Court Kiosks | 55 kWh | 7,800 sq ft |
| 12 | `Campus_assembly_Lea` | Assembly | Multipurpose Gymnasium & Indoor Sports Facility | 130 kWh | 18,900 sq ft |
| 13 | `Campus_public_Moe` | Public | Student Admissions, Registrar & Career Wing | 75 kWh | 11,200 sq ft |
| 14 | `Campus_warehouse_Nia` | Warehouse | Groundskeeping Equipment & Utility Storage | 40 kWh | 5,500 sq ft |
| 15 | `Campus_office_Omar` | Office | Research Centers & Graduate Studies Building | 105 kWh | 16,300 sq ft |
| 16 | `Campus_lodging_Pia` | Lodging | Female Student Residential Dormitory Beta | 115 kWh | 18,700 sq ft |
| 17 | `Campus_education_Quin` | Education | Physics, Chemistry & Robotics Laboratories | 120 kWh | 16,800 sq ft |
| 18 | `Campus_retail_Rey` | Retail | Stationery, Convenience & Copy Center | 50 kWh | 4,900 sq ft |
| 19 | `Campus_assembly_Sam` | Assembly | Alumni Amphitheater & Performing Arts Studio | 135 kWh | 19,100 sq ft |
| 20 | `Campus_public_Tia` | Public | Campus Healthcare Center & Emergency Clinic | 60 kWh | 7,200 sq ft |
| 21 | `Campus_warehouse_Uma` | Warehouse | Central Chiller Plant & High-Voltage Switchgear | 150 kWh | 8,900 sq ft |
| 22 | `Campus_office_Vik` | Office | IT Infrastructure & High-Performance Datacenter | 160 kWh | 11,500 sq ft |
| 23 | `Campus_lodging_Wes` | Lodging | Male Student Residential Dormitory Gamma | 110 kWh | 17,900 sq ft |
| 24 | `Campus_education_Xia` | Education | Architecture, Civil Engineering & Design Studios | 100 kWh | 15,200 sq ft |

### 3.2 Sensor Streams Tracked per Building
Each simulated building records 8 physical telemetry streams every hour:
1. `electricity`: Total electrical consumption (kWh)
2. `chilledwater`: Chilled water loop flow rate (tons/kWh equivalent)
3. `steam`: High-temperature steam flow for heating/sterilization
4. `hotwater`: Domestic hot water usage
5. `gas`: Natural gas flow for kitchens and boilers
6. `water`: General utility water meter (gallons/hr)
7. `irrigation`: Landscape irrigation flow (active predominantly at 05:00)
8. `hour`: Discrete hour index ($0 \le h \le 23$)

---

## 4. Digital Twin Simulation Engine (`dashboard.html`)

### 4.1 Diurnal Human Dynamics Modeling
Real campus demand follows non-linear human occupancy patterns. In `dashboard.html`, hourly electricity demand is calculated using a calibrated sinusoidal curve floored at 40%:
$$\text{dailyCurve}(h) = 0.4 + 0.6 \times \left(0.5 + 0.5 \times \sin\left(\frac{h - 6}{24} \times 2\pi\right)\right)$$
- **Physical Rationale:** Buildings never reach zero load overnight; baseline equipment (servers, emergency lighting, refrigeration, security routers) maintains continuous power draw.
- **Space Multipliers:** `office` and `education` facilities apply a **1.2x weekday multiplier** to reflect class schedules.
- **Stochastic Noise:** A randomized multiplier $\eta \sim \mathcal{U}(0.85, 1.15)$ simulates realistic occupancy volatility.

### 4.2 Environmental & Equipment Volatility Features
The digital twin simulates ambient microclimate conditions:
- **Air Temperature:** Sinusoidal cycle from $26^\circ\text{C}$ to $32^\circ\text{C}$
- **Dew Point:** $20^\circ\text{C} \pm 1.5^\circ\text{C}$
- **Wind Speed:** $1.0 \text{ to } 5.0 \text{ m/s}$

Crucially, the simulation tracks a rolling history of the previous 30 hours to compute rolling standard deviations of HVAC telemetry:
$$\sigma_{\text{chilledwater, 6h}} = \sqrt{\frac{1}{6} \sum_{i=1}^6 (x_i - \bar{x})^2}$$
*Why this matters:* Sustained high chiller flow with zero variance ($\sigma \approx 0$) signals a stuck pneumatic actuator valve or an improperly tuned thermostat loop—an equipment fault that simple static threshold rules cannot detect.

---

## 5. Machine Learning Models

### 5.1 Training Dataset: Building Data Genome 2.0 (BDG2)
Rather than training on synthetic approximations, our models were trained on the **Building Data Genome 2.0 (BDG2)** dataset (Lawrence Berkeley National Lab & National University of Singapore).
- **Scope:** 30 institutional buildings across an entire university campus.
- **Volume:** **421,195 actual hourly meter readings** across 2016–2017.
- **Split Strategy:** Strict **chronological train/test split** (the earliest 80% of hours for training, the final 20% for testing). Random K-Fold splits were deliberately rejected because shuffling time-series data leaks future autoregressive information into past evaluations.

### 5.2 Model 1: XGBoost Regressor (`regressor_demand.json`)
- **Objective:** Forecast building-level electricity demand ($y \in \mathbb{R}^+$) for the next operating hour.
- **Input Features (23 columns):** `hour`, `dayofweek`, `is_weekend`, `month`, `lag_1h`, `lag_24h`, `rolling_mean_24h`, `airTemperature`, `dewTemperature`, `windSpeed`, `sqft`, plus meter lags and 6-hour rolling standard deviations for chilled water, steam, hot water, gas, water, and irrigation.
- **Empirical Test Metrics:**
  - Coefficient of Determination ($R^2$): **0.993**
  - Mean Absolute Error (MAE): **5.97 kWh**
  - Root Mean Squared Error (RMSE): **10.98 kWh**
- **Feature Importance:** Autoregressive memory (`lag_1h`, `lag_24h`, `rolling_mean_24h`) accounts for ~98% of variance, confirming that immediate demand history is the single strongest physical predictor in high-frequency forecasting.
- **Production Guardrail:** Enforced via `max(0.0, demand)` clamp in `model_server.py` to prevent negative demand extrapolation on out-of-distribution inputs.

### 5.3 Model 2: XGBoost Classifier (`classifier_risk_tier.json`)
- **Objective:** Classify each building into an actionable operational risk tier based on historical percentile distributions:
  1. **`Normal`** ($< 75^{\text{th}}$ percentile of historical load): Baseline operations.
  2. **`Watch`** ($75^{\text{th}} - 90^{\text{th}}$ percentile): Elevated consumption; monitoring required.
  3. **`High-Risk`** ($90^{\text{th}} - 97^{\text{th}}$ percentile): Nearing branch breaker capacity.
  4. **`Shed-Now`** ($> 97^{\text{th}}$ percentile / Top 3%): Severe overload; immediate candidate for load shedding.
- **Empirical Performance & Class Balancing:** The classifier achieved an overall accuracy of **82.6%**. However, because >75% of real building hours are `Normal`, a standard model achieves high overall accuracy while missing rare critical events (initial recall on `Shed-Now` was 0.20–0.23). We addressed this in `train_models.py` by applying inverse-frequency sample weighting:
  $$w_c = \frac{N_{\text{total}}}{K \times N_c}$$
  This heavily penalizes the model for failing to predict rare `Shed-Now` emergencies.

---

## 6. GridWise: The Mathematical Optimization & AI Engine

### 6.1 The 3-Stage Optimization Pipeline
GridWise separates natural language reasoning from rigorous numerical optimization:

```
[ Operator Note ] ──► [ Stage 1: Gemini Flash LLM ] ──► [ Stage 2: Guardrails ] ──► [ Stage 3: PuLP MILP Solver ]
```

1. **Stage 1 — LLM Directive Interpreter (`interpreter.py`):** Translates free-form operator text into one of five structured directives: `solar_reduction`, `no_charge_window`, `no_discharge_window`, `minimum_battery_reserve`, or `no_op`.
2. **Stage 2 — Deterministic Guardrail Layer (`guardrails.py`):** Clamps hour arrays to $[0..23]$ unique ascending integers, clamps reduction factors to $[0.0..1.0]$, and normalizes percentage reserves to absolute kWh.
3. **Stage 3 — Mixed-Integer Linear Program (`optimizer.py`):** Solves the global cost-minimization dispatch schedule using the COIN-OR CBC branch-and-cut solver in **< 20 milliseconds**.

### 6.2 Mathematical MILP Formulation

#### Decision Variables (for each hour $h \in \{0, 1, \dots, 23\}$):
- $g_h \ge 0$: Grid electricity import (kWh)
- $s_h \ge 0$: Solar power consumed on campus (kWh)
- $c_h \ge 0$: Energy charged into the battery (kWh)
- $d_h \ge 0$: Energy discharged from the battery (kWh)
- $u^{\text{chg}}_h \in \{0, 1\}$: Binary charging indicator
- $u^{\text{dis}}_h \in \{0, 1\}$: Binary discharging indicator
- $E_h \ge 0$: Battery State of Charge (energy level) at the end of hour $h$ (kWh)
- $P_{\text{peak}} \ge 0$: Auxiliary variable tracking peak grid import across 24 hours

#### Objective Function:
$$\min_{g, s, c, d, E, u} \quad \sum_{h=0}^{23} \left( g_h \times \text{tariff}_h \right) + 10^{-4} \times P_{\text{peak}}$$
- **Primary Term:** Minimizes total grid purchase cost in Bangladeshi Taka (BDT).
- **Regularization Term ($10^{-4} \times P_{\text{peak}}$):** Breaks mathematical degeneracy during flat-tariff hours, smoothing the demand profile and preventing artificial spikes.

#### System Constraints:
1. **Hourly Campus Power Balance:**
   $$g_h + s_h + d_h = \text{demand}_h + c_h \quad \forall h \in \{0, \dots, 23\}$$
2. **Solar Resource Limits:**
   $$s_h \le \text{solar}_h \quad \forall h$$
   *(If a `solar_reduction` directive applies with factor $\alpha$, $s_h \le \alpha \times \text{solar}_h$)*.
3. **Battery Energy Transition:**
   $$E_0 = E_{\text{initial}} + c_0 - d_0$$
   $$E_h = E_{h-1} + c_h - d_h \quad \forall h \in \{1, \dots, 23\}$$
4. **Battery Energy Bounds (Safety Reserves):**
   $$E_{\text{min}} \le E_h \le E_{\text{max}} \quad \forall h$$
   $$(50 \text{ kWh} \le E_h \le 500 \text{ kWh})$$
5. **Inverter C-Rate Limits:**
   $$c_h \le C_{\text{max}} \times u^{\text{chg}}_h \quad (C_{\text{max}} = 100 \text{ kW})$$
   $$d_h \le D_{\text{max}} \times u^{\text{dis}}_h \quad (D_{\text{max}} = 100 \text{ kW})$$
6. **Mutual Exclusivity of Inverter States:**
   $$u^{\text{chg}}_h + u^{\text{dis}}_h \le 1 \quad \forall h$$
   *(Guarantees that a physical battery never attempts to charge and discharge simultaneously)*.
7. **End-of-Day Battery Neutrality:**
   $$E_{23} = E_{\text{initial}} = 250 \text{ kWh}$$
   *(Prevents the optimizer from "cheating" by completely draining the battery to zero to show artificially low cost, leaving the campus defenseless for the next day)*.
8. **Peak Tracking Constraint:**
   $$g_h \le P_{\text{peak}} \quad \forall h$$

---

## 7. The Closed-Loop Microgrid Protection Mechanism

A frequent critique of energy optimizers is that they operate as open-loop calculators without interacting with the physical consumers. **EnergyWise implements a closed loop**:

```
[ Total Campus Demand ] ──────► [ GridWise Optimization ]
                                          │
    ┌─────────────────────────────────────┘
    ▼
[ Microgrid Power Supply: Solar Used + Battery Discharge (s_h + d_h) ]
    │
    ▼
[ Priority Relief Allocation: Shed-Now ──► High-Risk ──► Watch ]
    │
    ▼
[ Buildings Shielded from Outage ──► 🛡️ GridWise Badge Active ]
```

### 7.1 Mathematical Relief Allocation Algorithm
1. Calculate available local clean power:
   $$P_{\text{microgrid}} = s_h + \mathbb{I}(d_h > 0) \times d_h$$
2. Identify all buildings categorized as non-normal ($T_i \in \{\text{Shed-Now, High-Risk, Watch}\}$).
3. Sort at-risk buildings in descending order of criticality:
   $$\text{Priority: } \text{Shed-Now} \succ \text{High-Risk} \succ \text{Watch}$$
4. For each building in sorted order, allocate reserve power to satisfy its critical operating threshold ($40\%$ of predicted demand).
5. When protected:
   - The building is shielded from imminent feeder shedding.
   - A glowing **`🛡️ GridWise`** shield badge renders on the building facade.
   - The top status bar updates dynamically: e.g., `4 at-risk (4 saved)`.

---

## 8. Network Infrastructure & Production Deployment

| Component | Host / Port | Tech Stack | Functional Responsibility |
|---|---|---|---|
| **Frontend Dashboard** | Browser (`dashboard.html`) | HTML5, Canvas, ES6 JS, CSS3 | Real-time digital twin visualization, 24-building skyline, operator overrides |
| **Model Server** | `http://localhost:8001` | FastAPI, Uvicorn, XGBoost, Pandas | Batched inference for XGBoost regressor and classifier |
| **GridWise Proxy** | `http://localhost:8002/gridwise` | FastAPI, HTTPX, CORSMiddleware | Manages 90s connection timeouts, prevents browser CORS blocking, handles cold starts |
| **GridWise Engine** | `https://gridwise-api.onrender.com` | FastAPI, PuLP, COIN-OR CBC, Google Gemini | Production cloud deployment running NLP translation and 24h MILP optimization |

---

## 9. Comprehensive Faculty Defense: Questions & Answers

### Q1: "Why did you use Linear Programming (MILP) instead of Deep Reinforcement Learning (RL) or Neural Networks for battery scheduling?"
**Answer:**  
In critical cyber-physical infrastructure, Deep Reinforcement Learning (DRL) and Neural Networks suffer from fundamental limitations:
1. **Lack of Feasibility Guarantees:** Neural networks output probabilistic approximations. They cannot mathematically guarantee that physical safety constraints (e.g., $E_h \ge 50 \text{ kWh}$ or $u^{\text{chg}} + u^{\text{dis}} \le 1$) will never be violated. A single violated constraint can trigger physical battery thermal runaway or breaker trips.
2. **Computational Overhead:** DRL requires days of training and can fall into poor local optima. In contrast, our **Mixed-Integer Linear Program (MILP)** solved via the branch-and-cut COIN-OR CBC algorithm is **deterministic**, finds the **globally provable mathematical optimum**, and executes in **under 20 milliseconds**.
3. **Appropriate Separation of Concerns:** In our architecture, AI/ML is applied where it excels (learning complex empirical relationships in high-dimensional meter and weather data), while classical Operations Research handles rigorous, safety-critical resource scheduling.

### Q2: "Did you use synthetic toy data to train your models?"
**Answer:**  
No, Sir. While our preliminary project proposal initially considered synthetic data, we deliberately upgraded to the **Building Data Genome 2.0 (BDG2)** dataset from the Lawrence Berkeley National Laboratory (LBNL). We extracted **421,195 rows of hourly real-world telemetry** from 30 actual university facilities across 2016–2017, encompassing all eight utility meters (electricity, chilled water, steam, hot water, gas, water, irrigation) and local weather metrics. Our regressor achieved an $R^2$ of **0.993** on a strict chronological test split of unseen future hours.

### Q3: "What prevents the Google Gemini LLM from hallucinating an invalid constraint and crashing the power grid?"
**Answer:**  
We enforce a strict **zero-trust deterministic guardrail architecture** (`guardrails.py`) between the language model and the numerical optimizer:
1. **Structured Schema Validation:** The LLM's response is parsed against strict Pydantic v2 schemas. If the model outputs non-JSON or unsupported keys, the fallback defaults to `no_op`.
2. **Deterministic Clamping:** Even if Gemini hallucinates an hour outside physical reality (e.g., `hour: 27`), our validator clamps the array to valid integers within $[0..23]$.
3. **Factor and Reserve Normalization:** Any solar reduction factor is clamped to $[0.0, 1.0]$. Percentage reserves (e.g., "keep 50% reserve") are explicitly computed against the physical battery capacity ($0.50 \times 500 = 250 \text{ kWh}$).
4. **Physical Solver Guardrails:** If contradictory constraints are somehow introduced, our exception handlers catch infeasibility and revert to a safe baseline dispatch rather than allowing an unhandled crash.

### Q4: "Why does the regressor feature importance show ~98% reliance on autoregressive lags rather than HVAC and weather features?"
**Answer:**  
In high-frequency (hourly) demand forecasting, electrical load exhibits extreme autocorrelation—the electricity drawn at 1:00 PM is overwhelmingly conditioned on the consumption at 12:00 PM and the exact same hour yesterday (`lag_1h`, `lag_24h`, `rolling_mean_24h`). This is a well-documented empirical phenomenon in electrical engineering literature (persistence forecasting). The HVAC and weather features provide crucial edge-case signals (such as temperature swings and equipment anomalies), while the autoregressive lags establish the baseline load magnitude.

### Q5: "How does the system ensure resilience if the internet connection or the cloud API goes down?"
**Answer:**  
The entire platform is built with **graceful multi-tier fallback**:
1. If the local ML Model Server (`model_server.py`) is offline, `dashboard.html` seamlessly activates an internal heuristic predictor calibrated to historical baseline demands.
2. If the cloud GridWise API is unreachable, the dashboard activates a local 24-hour unoptimized baseline dispatch plan, ensuring the visual digital twin never freezes.
3. If an operator enters an invalid directive, the LLM classifies it as `no_op`, preserving uninterrupted standard cost minimization.

---

## 10. Conclusion

**EnergyWise** demonstrates how modern artificial intelligence and classical mathematical optimization can be synthesized to solve a critical, real-world developing-world infrastructure challenge. By uniting real-world meter telemetry, supervised machine learning forecasting, zero-shot large language model reasoning, and exact mixed-integer linear programming, the system transforms an institutional campus from a passive victim of rolling load-shedding into an active, resilient, and cost-optimized microgrid.
