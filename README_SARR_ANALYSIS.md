# Alex Sarr – Model projections for next 5 Wizards games

The model **projects PTS, TRB, AST, STL, and BLK on its own** for each of the next 5 games. It does **not** use PrizePicks or Underdog lines.

1. **Baseline:** Rolling 10-game means from Sarr’s game log (individual stats).
2. **Adjustments** applied to that baseline per game:
   - **Opponent defensive rank** (1 = best D, 30 = worst) – weaker D boosts scoring/assists/rebounds.
   - **Opponent win streak** – hot opponent = slightly tougher matchup (small downgrade).
   - **Size / matchup** – reflected via wingspan advantage and key players out.
   - **Wingspan advantage** (Sarr vs primary defender) – boosts blocks and rebounds most, then points/steals.
   - **Environment** – home games get a small boost.
   - **Availability** – key defenders/centers out = boost to PTS, TRB, BLK.

Output: **projected PTS, TRB, AST, STL, BLK** per game, plus **breakout score** and **P(over 25 PTS)**.

---

## Do I have to extract data myself?

**Game log — no (optional).** The script uses `sarr_game_log.csv` if present, otherwise tries to fetch the game log from Basketball-Reference.

**Next 5 games and opponent/matchup info — yes.** Edit **`sarr_next5_config.json`** with schedule, defensive rank, opponent win streak, wingspan advantage, and key players out. No prop lines are used.

---

## Setup

```bash
pip install -r requirements.txt
```

## Config: `sarr_next5_config.json`

For each of the next 5 games set:

- **date, opponent, venue** – from NBA.com or Basketball-Reference (Wizards schedule).
- **opponent_defensive_rank** – 1 = best defense, 30 = worst (Basketball-Reference team defensive rating).
- **opponent_win_streak** – current win streak (0 if unknown). Hot opponent = slightly tougher.
- **opponent_missing_key_players** – `true` when important defenders/centers are out.
- **wingspan_advantage_inches** – Sarr’s wingspan vs primary defender (Sarr ~7'4"). Positive = Sarr has the edge.

No PrizePicks or Underdog fields are used; the model projects all stats itself.

---

## Run

```bash
python sarr_prop_analysis.py
```

- Loads game log (CSV or Basketball-Reference).
- Reads `sarr_next5_config.json`.
- Prints baseline (rolling means), then per-game **model projections** (PTS, TRB, AST, STL, BLK) and breakout score.
- Writes **`sarr_next5_analysis.csv`** with one row per game and all projected stats.

## Interpreting output

- **Projected PTS, TRB, AST, STL, BLK** – model output for that game (baseline + opponent, win streak, size, wingspan, environment, availability).
- **Breakout score (0–1)** – higher = more “due” for a big game given matchup and context.
- **prob_over_25_pts** – probability he goes over 25 points from the rolling distribution (no opponent adjustment in that probability).

Use the projected stats as your own reference; there is no comparison to any external lines.

---

## Model validation (STEP 1)

Before trusting projections, the script **backtests** on the last 10–15 games:

- For each of those games it pretends the game was "upcoming" and uses the **rolling baseline from prior games only** (no opponent context in backtest).
- It compares that projection to the **actual** stat and computes:
  - **Error** = actual − projected (positive = model underprojected).
  - **MAE** (mean absolute error) and **bias** (mean error) per stat.

Run the script and check the "STEP 1: Model validation" section. Backtest details are saved to **`sarr_backtest_validation.csv`** (actuals, proj_*, *_error columns). Use this to judge whether the model is reasonable before relying on next-5 projections.

---

## Minutes and matchup (STEP 2 & 3)

- **Minutes:** The model uses **rolling minutes (MP)** and **per-minute production**. It projects **minutes first** (from baseline MP + home/away + key players out), then **stats** = proj_MP × (stat per minute) × matchup adjustments. So projections scale with expected playing time.
- **Matchup (optional):** In **`sarr_next5_config.json`** you can add per game:
  - **rim_protector_present** (bool): applies a small penalty to PTS and BLK.
  - **switch_heavy_defense** (bool): small penalty to AST.
  - **zone_frequency** ("low" | "medium" | "high"): small penalty when zone is used more.

---

## Output (STEP 4)

- **`sarr_next5_projections.csv`** – main output: one row per upcoming game with proj_MP, proj_PTS, proj_TRB, proj_AST, proj_STL, proj_BLK and all context columns. Use this file to feed results into other tools or view projections in a table.
- **`sarr_backtest_validation.csv`** – backtest details (actual vs projected for past games).
