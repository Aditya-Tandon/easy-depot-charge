# Overnight Charging Planner

A one-page web app for an electric-van depot manager. It plans tonight's charging so every van is ready by morning, at the lowest electricity cost.

Live: https://algosoup-charging-planner.vercel.app (no login)

## What it does

- Shows tonight's vans (arrival, departure, charge level), half-hourly electricity prices and grid carbon.
- Works out the cheapest charging plan and compares its cost with plugging every van in on arrival.
- Lists any van that cannot reach its target charge, and says why (window too short, charger too slow, or site power limit).
- Reads a pasted driver or dispatcher note, pulls out the changes (for example "Van 03 back at 23:30"), and lets you accept or reject each one.
- Publishes the plan as a numbered revision.

## How to use the page

1. **Refresh prices.** Fetches tonight's prices and carbon forecast.
2. **Optimise.** Builds a plan for all vans.
3. **Review the plan.** Check the cost against charge-on-arrival, the per-van schedule, and any shortfalls with their reasons.
4. **Paste updates.** Paste a note from a driver or dispatcher.
5. **Accept or reject.** Each extracted change is shown with the quote it came from. Invalid ones are shown with the reason.
6. **Re-optimise.** Accepted changes update the vans' sessions. Run the plan again.
7. **Publish.** Saves the plan as the next revision.

## How the plan is worked out

- The night runs from 18:00 to 08:00 London time, split into half-hour slots.
- Each van can charge only between its arrival and departure.
- Each van has a maximum charging rate. The whole site has a power limit shared by all vans.
- Chargers are 90% efficient: for every 1 kWh from the grid, 0.9 kWh reaches the battery.
- A van is never charged beyond its target.
- The goal is the lowest total cost. Prices are Octopus Agile, which change every half hour and can go negative.
- This is solved as a linear program (a standard maths optimisation) with SciPy's HiGHS solver.

If a van cannot reach its target, it still gets as much charge as possible, and the page says exactly why:

- **No charging window**: it leaves before it arrives, or is outside the night.
- **Charger too slow**: e.g. "needs 50.0 kWh; max 11.0 kW x 4.0 h x eta 0.9 = 39.6 kWh -> short 10.4 kWh".
- **Site limit**: the van could make it alone, but the site power limit is fully used at certain times.

**Baseline (charge on arrival):** each van plugs in when it arrives and charges at full rate until it reaches target. Vans that arrived first get power first. It uses the same prices and the same site limit, so the cost comparison is fair.

## How driver notes are handled

1. The note is sent to an LLM (Gemini 3.1 Flash-Lite via Runware; set `RUNWARE_MODEL` to change it) that returns a list of changes as JSON: van, field, new value, and a quote from the note.
2. Allowed fields: arrival time, departure time, arrival charge, target charge.
3. The JSON is checked against a strict schema (pydantic). Anything malformed is rejected.
4. Each change then gets one validation pass:
   - the van exists and has a session tonight;
   - the value is in range and differs from the current one;
   - times and charge levels still make sense together (e.g. departure after arrival, target above arrival charge);
   - the quote is copied word-for-word from the note and actually supports the value.
5. Valid changes are marked **pending**; failed ones **invalid**, with the reason.
6. Nothing changes until the manager accepts a change.

## Run locally

```bash
conda activate algosoup
pip install -r requirements-dev.txt   # runtime deps + pytest + uvicorn
```

Set `DATABASE_URL` and `RUNWARE_API_KEY` in `~/.env` or the environment (see `.env.example`). Without `DATABASE_URL` the app falls back to a local SQLite file.

```bash
python -m app.seed               # one depot, a small fleet, tonight's sessions
uvicorn app.main:app --reload    # http://127.0.0.1:8000
pytest -q                        # tests
```

## Deploy

Deployed on Vercel, entrypoint `app/main.py`.

```bash
vercel env add DATABASE_URL production
vercel env add RUNWARE_API_KEY production
vercel deploy --prod
```

## Stack

Python 3.11, FastAPI, Jinja2, Pico CSS, SQLModel on Neon Postgres, httpx, pydantic, SciPy (HiGHS), pytest, Vercel.

## Project layout

```
app/
  main.py          routes: page, refresh prices, plan, publish, proposals
  optimiser.py     the charging model (pure, no I/O) and the charge-on-arrival baseline
  ingest.py        Octopus Agile prices (London, region C) and Carbon Intensity (London, region 13)
  llm.py           driver note -> checked proposals, via Runware
  models.py        tables: Site, Vehicle, Session, PriceSlot, Plan, Allocation, VanOutcome, Proposal
  db.py            database engine and env loading
  seed.py          seeds one London depot and tonight's vans
  timeutil.py      shared time helpers (UTC, London time, half-hour slot)
  templates/       index.html (the single page)
tests/             pytest tests for optimiser, ingest, LLM checks and the app
requirements.txt
.env.example
```

Data sources: if Octopus has no price for a slot, a fallback price curve is used and marked as such. Missing carbon is left blank.

## Future improvements

- **Postcode per site.** Pick the Octopus region from the postcode, and use the Carbon Intensity postcode endpoint, so prices and carbon come from the right region automatically.
- **Multiple sites.** One page per depot plus a combined view, with a shared fleet.
- **Routing vans between depots.** Assign each van to the depot that best fits tomorrow's first job, given charger capacity and local prices. This is a joint assignment and charging problem.
- **Large or changing fleets.** Add, retire and edit vans from the page instead of the seed script, and make the page work for hundreds of vans: search, filters (e.g. only vans that are short or arriving late), grouping, paging, and a compact fleet grid. The optimiser already handles 50 vans x 48 slots in under 20 ms; at larger sizes, check solve time and consider solving per group.
- **Live charger data.** Use real charger readings instead of planned values, and re-plan during the night.
- **Logins for several managers**, with a record of who accepted or published what.
- **Alerts** when the next day's Agile prices are published (Octopus usually publishes the next day's Agile rates in the late afternoon), so the plan can be run straight away.

## Known issues

These came out of a code-quality review at the end of the build. They are not fixed yet.

- **No login and no CSRF protection.** Every action (refresh prices, optimise, publish, suggest changes, accept or reject) is a plain POST with no auth. Another website could trigger them, and repeated calls spend Runware and price-API quota. There is no rate limit. Fine for a demo; add a login or at least a same-origin check before real use.
- **Database outages are not handled.** Price and LLM outages are tested and fail safely, but if Neon is down the page returns a server error. There are no tests for it.
- **Two routes can still error.** `make_plan` in `app/main.py` assumes a site exists and that every half-hour has a price after a refresh (a missing slot raises `KeyError`). Accepting a suggestion converts the stored value without a guard.
- **`index()` in `app/main.py` does too much.** It runs the queries, works out whether the plan is stale and builds the chart data in one function. Moving that into a separate, testable module would help.
- **Thin app-level tests.** `tests/test_app.py` has one end-to-end test, and it checks page wording, so copy changes break it. Publishing, rejecting, conflicting accepts and the stale-plan warning are not tested over HTTP. The SQLite fixture is copied across three test files instead of living in a `conftest.py`.
- **Schema changes are manual.** Tables are created with `create_all`; the `plan.baseline_load_json` column was added by hand with `ALTER TABLE`. There are no migrations.
- **Local runs use the production database.** `app/db.py` loads `~/.env`, so running locally with that file points at the live Neon database. A missing `DATABASE_URL` falls back to a local SQLite file, which will not work on Vercel.
- **Small optimiser and LLM edge cases.** A site-limit shortfall under 0.05 kWh is left out of the reason text (the delivered figure still shows it). Fairness spreads the worst shortfall evenly, but vans better off than the worst one are split by cost. A charge suggestion within 0.5 percentage points of the current value counts as "no change". The 24-hour limit on departures is measured in real time, so it behaves slightly differently on clock-change nights.
