# AI Lab Project Proposal — Grid Status: Unserved Demand Today

*Juboraz Afnan Mehmud, Fahim Bin Kibria — United International University · CSE*
*Draft project idea page for an AI Lab course project, built around a
Bangladesh campus energy scenario. The core problem framing — co-optimizing
buildings and assets with AI/ML/OR — draws on an industry project brief as
inspiration, with original wording, structure, and design. The digital-twin
details and energy figures are a working interpretation — verify scope and
numbers before treating this as final.*

## The pitch

The grid can't cover demand. The buildings should stop waiting to find out.

Load shedding in Bangladesh isn't a demand-response program, it's routine —
generation falls short of demand and the grid cuts service to cope. This
project builds an AI system that watches an entire campus in real time and
decides, building by building, where to save power before an outage notice
arrives.

- National generation capacity: ~16,500 MW max ever produced
- Peak demand gap: 1,500 MW unserved at past peaks
- Scope: 75 buildings in this campus scenario

## The idea, in plain terms

**Working interpretation:** simulate a university-style campus — dozens of
buildings, each with its own HVAC, lighting, chillers, and meters. A
centralized AI system sits above all of it and continuously decides how to
shed or shift load — not by shutting things off blindly, but by weighing
cost, emissions, and comfort against each other at every moment.

To make good decisions, that system has to see the whole picture at once: a
chiller running hotter than its design spec, a condenser loop that never
stops circulating when it should be cycling, a heat wave arriving in six
hours, tomorrow's load forecast, and how many people are actually in the
building right now. None of these signals mean much in isolation. Together,
they're the difference between an optimizer that's genuinely useful and one
that just reacts.

That's the job of the digital twin underneath it: pull live readings from
sensors, BMS points, occupancy counters, and weather feeds into one
continuously updated model of the campus, so the optimization engine is
always working from the real state of the buildings — not a snapshot from
this morning.

## How the signals become a decision

Four stages, running continuously rather than on a fixed schedule. Each
stage feeds the next, and the results loop back into the twin.

1. **Sensors & BMS** — chiller loads, water flow, occupancy, weather, meters
2. **Digital twin** — one continuously updated model of every building,
   asset, and live condition
3. **AI / ML / OR engine** — forecasts load, weighs cost, comfort, and
   emissions, and picks which assets to shift, shed, or hold
4. **Actions on buildings** — battery dispatch, solar use, EV charge
   windows, HVAC and lighting setpoints

Results feed straight back into the twin, every cycle — a closed loop, not
a one-way pipeline.

## The conditions the engine has to watch for

Three categories, and the engine needs all three at once — a static
rulebook reacts to one signal at a time; this needs AI/ML/OR because it
doesn't.

**Equipment health**
- Chiller running above design — a chiller working harder than it was
  specced for is a cost and a wear signal before it's a comfort problem
- Condenser or chilled water never cycling down — continuous flow when the
  load doesn't call for it usually means a stuck valve or a control loop
  left in a bad state

**Environmental & occupancy**
- Extreme heat or cold arriving — pre-cool or pre-heat ahead of a
  forecasted swing, not scramble to catch up once it hits
- Tomorrow's load and tariff signal — shift flexible load into cheaper,
  cleaner hours instead of reacting in the moment
- Real people, not a fixed schedule — an empty lecture hall on a scheduled
  class day is wasted conditioning; occupancy data tells the twin the truth

**Critical grid state**
- A scheduled or unscheduled load-shed notice — the campus needs to
  already know which loads to drop first, and for how long, before the
  outage lands

## Scenarios worth building first

Early scoping should narrow this down to whatever the first working
prototype can realistically cover:

1. Battery charge and discharge scheduling
2. Local solar use when the grid is strained
3. EV charge and discharge scheduling
4. Finding available building flexibility

## What success looks like

- **Visibility** — a dashboard that makes consumption patterns legible
  instead of buried in point-by-point meter data
- **Cost** — real savings on peak-tariff hours, where a few smart shifts
  across 75 buildings add up fast
- **Emissions** — less CO2e per building, shifting load toward
  solar-available or cleaner-grid hours
- **Resilience** — storage, local solar, and pre-emptive shifting mean an
  outage notice doesn't catch the campus flat-footed

## Scope, skills, and deliverables

**What's useful to know:** digital twin / BMS platform basics; AI/ML/OR
(depth depends on finalized scope); building systems (BMS, energy,
emissions basics); databases (graph and time-series concepts); Python (for
the AI/ML/OR models); REST APIs (for pulling sensor and BMS data); cloud
basics (any provider — helpful, not required)

**What ships at the end:**
- Working code and model — built and tested against simulated or sample
  building data
- User manual — clear enough to configure and operate without hand-holding
- As-built report — documents the design decisions for whoever builds on
  this next
- Version-controlled codebase — delivered on GitHub or equivalent
