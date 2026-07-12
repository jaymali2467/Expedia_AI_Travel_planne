# ✈️ AI Air Travel Companion — Expedia Group Campus Hackathon (Innovation Round)

An AI-powered flight assistant that **infers traveler preferences from messy
history**, routes direct and connecting flights over a **time-dependent flight
graph**, ranks options by a **dollar-grounded, explainable cost model**, and
**explains every recommendation in plain language** — including when it had to
compromise.

**Author:** Jay Balasaheb Mali · ch24b067@smail.iitm.ac.in

---

## What it does

- **Natural-language trip requests** — type prompts like the official benchmark
  ("I need to get from home to Tokyo next month") and an LLM parses origin,
  destination, trip type, dates, and total trip duration. "Home" resolves to
  the user's home airport. Region-level requests ("a multi-city Asia trip")
  resolve to real airports in the dataset.
- **Preference inference from raw history** — structured profile fields
  (price_sensitivity, direct_preference) set priors; an open-source LLM
  (Llama 3.3 70B via Groq) refines them using the unstructured `raw_history`
  text into budget / time / comfort weights, shown live in the sidebar.
- **Time-dependent routing** — direct itineraries plus engine-synthesized
  connections: any two legs are chainable only if the second departs 45 min to
  `max_layover_minutes` after the first arrives. (Floyd–Warshall was evaluated
  and rejected: its static-edge assumption breaks on time-dependent networks.)
- **Explainable ranking (Value of Time)** — instead of opaque normalized
  scores, weights convert into an implicit **$/hour value of the traveler's
  time**, and flights are ranked by _effective cost_ =
  `price + VOT × hours + stop penalty ($)`. Every ranking decision is
  auditable in dollars ("$50 saved beats 2 extra hours at $7/hr").
- **Round trips & multi-city journeys, order-optimized** — tours default to
  ending back home; when the visit order isn't fixed, the planner tries **all
  orderings** (exact for ≤3 stops) and keeps the cheapest complete journey by
  total effective cost, announcing the chosen order. Legs chain time-feasibly
  with stay allocation and a hard deadline when a total duration is stated.
  Region-level asks ("Asia") treat cities as suggestions: unreachable ones are
  substituted with reachable alternatives from the region, disclosed in the UI.
- **Nearest-departure guidance** — when a route has nothing in the requested
  window or the following month, the app reports the nearest future departure
  and offers a one-click search around it (journeys get the same rescue when
  their first leg fails).
- **Graceful constraint relaxation with consent** — if hard constraints (e.g. a
  90-minute max layover) eliminate every option, the engine relaxes the layover
  cap first (keeping the user's dates), apologizes, and badges each flight that
  exceeds the stated preference. Empty date windows are never silently widened:
  the engine probes only the ~30 days after the window and **asks** before
  searching there ("found 4 options in the following month — search that
  window?").
- **Self-transfer synthesis, visibly labeled** — engine-built connections
  (two independently ticketed direct legs) carry a "Self-transfer" badge to
  distinguish them from protected bundled itineraries.
- **Seasonality & scarcity awareness** — holiday-season pricing and low-seat
  warnings surfaced as badges from dataset fields.
- **Explicit trade-off surfacing** — an "Our pick / Cheapest / Fastest" strip
  with dollar-and-hour deltas, an LLM insight grounded only in real
  alternatives (anti-hallucination), and a price-vs-duration scatter chart.

## Architecture / workflow

```
user prompt ─► LLM request parser ──► origin / destination / trip type /
                                       date window / duration budget
user profile ─► structured priors ─► LLM preference refiner ─► α β γ weights
                                                                   │
flights_data.csv ─► time-dependent graph (direct + synthesized     ▼
                    connections, layover & date constraints) ─► effective-cost
                                                                 ranking (VOT)
                                                                   │
                                              4-tier constraint    ▼
                                              relaxation ─► ranked results ─►
                                              LLM explanation + trade-off UI
```

## Setup

1. **Requirements:** Python 3.10+.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data:** place `flights_data.csv` and `user_data.csv` (from the hackathon
   toolkit) in a `data/` folder next to `app.py`. The app refuses to start
   without them — it never fabricates data.
4. **LLM key (recommended):** create `.streamlit/secrets.toml` containing
   exactly (quotes required — TOML strings must be quoted):
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   or set the `GROQ_API_KEY` environment variable. Without a key the app
   still runs: parsing and weights fall back to documented heuristics, and
   the sidebar shows a live 🟢/🔴 AI status with the exact reason.
5. **Run:**
   ```bash
   streamlit run app.py
   ```
   On the very first launch the app writes `.streamlit/config.toml` to lock
   the light theme; refresh once if the first page renders dark.

## Assumptions (documented per toolkit FAQ #6)

1. **Simulated current date.** The dataset lives in a fixed historical window,
   so relative dates in prompts ("next month", "a Tuesday") are resolved
   against a simulated "today" anchored inside the dataset's densest coverage
   (April 1 of its first year, clamped to actual bounds). The assumption is
   displayed in the UI on every search. For this submission it is pinned to
   **2025-08-01** via `SIM_TODAY` in `secrets.toml` — selected with
   `find_sim_today.py` as the date maximizing benchmark-prompt feasibility
   given the dataset's route-level sparsity (see Limitations).
2. **The engine never searches the past.** Every search path is floored at
   simulated today. Seasonal windows that would fall entirely in the past
   ("summer" asked in November) roll forward to their next occurrence, with
   the reinterpretation shown in the UI. Undated and open-ended requests get
   bounded upcoming windows (30 days, widened by any stated trip duration) —
   never the whole dataset. "After May 31st" becomes a 30-day window from
   June 1. Searching beyond a requested window always requires an explicit
   user confirmation.
3. **Value-of-Time model.** Inferred weights map to an implicit hourly value
   of the traveler's time: `VOT = $35 × time_weight / budget_weight`, capped
   at $250/hr. The $35 base rate is a tunable constant, not market-calibrated.
4. **Self-transfer connections.** Engine-synthesized connections combine
   independently ticketed legs (like buying two train tickets); real bundled
   itineraries carry missed-connection protection that self-transfers do not.
   Both are treated as valid options. Minimum connection time: 45 minutes.
5. **Greedy multi-leg planning.** Round trips and multi-city journeys pick
   each leg's best effective-cost option sequentially, not jointly.
6. **Stay allocation.** When a total trip duration is stated, stays are split
   evenly across intermediate cities; otherwise a 2-day default stay applies.
7. **Prices are USD, per traveler, one adult;** multi-leg journey price is the
   simple sum of leg prices.
8. **Missing `layover_airports` on direct flights is expected** (a direct
   flight has no layover) and is never treated as bad data or dropped.

## Limitations

- **Route-level data sparsity bounds search quality.** The provided dataset
  (~50k flights spread across many city pairs and an 18-month window) leaves
  most individual routes with only a handful of departure dates — e.g. some
  intercontinental pairs fly a few days per year. The app therefore never
  pretends continuous availability: it reports the nearest feasible departure
  and asks before planning around it, rather than fabricating options. With a
  production-density schedule, the same engine would return in-window results
  for far more queries.
- Multi-leg journeys are greedily optimal per leg (the visit _order_ is
  optimized exhaustively for ≤3 stops), not jointly optimal across legs.
- Connections are limited to what the dataset contains plus 1-stop synthesis;
  the engine does not synthesize 2+ stop self-transfers.
- Comfort weighting currently prices stops only; cabin class and preferred
  airline appear as information badges but are not priced into the ranking.
- LLM parsing quality depends on the free-tier Groq API being reachable;
  offline fallbacks are simpler keyword/region heuristics.
- No real-time availability, fare rules, or booking — recommendations only,
  over the provided static dataset.

## Future improvements

- Jointly optimal multi-leg search (dynamic programming over the
  time-expanded graph) replacing greedy chaining.
- Permutation of multi-city visit order (exhaustive for ≤4 cities is exact).
- Pricing cabin class and airline preference into effective cost.
- Learning each user's true Value of Time from booking outcomes rather than
  a fixed mapping.
- Calendar heatmap showing cheapest departure days across a flexible window.
- Conversational multi-turn refinement ("make it cheaper", "avoid redeyes").

## Project files

| File                  | Purpose                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `app.py`              | Entire application (UI + parsing + routing + ranking + explanation) |
| `flights_data.csv`    | Toolkit flight dataset (~50k records) — not committed               |
| `user_data.csv`       | Toolkit user dataset (~50 records) — not committed                  |
| `requirements.txt`    | Python dependencies                                                 |
| `SOLUTION_SUMMARY.md` | Mandatory solution summary deliverable                              |

> **Note:** never commit `.streamlit/secrets.toml` or API keys. A
> `.gitignore` entry is recommended.
