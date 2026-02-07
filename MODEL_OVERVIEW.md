# Sarr Projection Model – 1-Page Summary

**Version 1.0.0** (`__version__` in code) – Contextual NBA stat projection model (validated)  
*Tag: v1.0 – Contextual NBA stat projection model (validated)*

**What’s in v1.0:** Rolling baseline; minutes as first-class input (proj_MP, per-min rates); opponent defensive rank (with extra penalty vs top-10); win streak, wingspan, home/away, key players out; matchup flags (rim protector, switch-heavy D, zone); explanation layer (reason string per stat); backtest 15–25 games with sklearn MAE; confidence bands (proj_lo, proj_hi); segment analysis (venue, B2B, defense tier, rim); noise/volatility report (PTS vs TRB, BLK volatility); vs-good-teams bias check; TRB/BLK emphasis. Outputs: `sarr_next5_projections.csv`, `sarr_backtest_validation.csv`.

---

## 1. Model objective

Produce **interpretable, validated projections** for Alex Sarr’s counting stats (PTS, TRB, AST, STL, BLK) and minutes for the **next 5 Washington Wizards games**, using only information available before each game. The model **prides itself on rebounds (TRB) and blocks (BLK)** and is built so **opponent defensive rating plays a big part** in projection shifts. Outputs include point estimates, **confidence bands**, and **reason strings** (e.g. “+ weak defense, + home, − strong defense (top-10)”) so the model is not a black box.

---

## 2. Inputs

- **Game log:** Rolling 10-game means (and standard deviations) for PTS, TRB, AST, STL, BLK, MP from Basketball-Reference (or CSV).
- **Per-game context (next 5):**
  - Opponent defensive rank (1 = best, 30 = worst)
  - Opponent win streak
  - Home / away
  - Wingspan advantage (Sarr vs primary defender, inches)
  - Key players out (yes/no)
  - Optional: rim protector present, switch-heavy defense, zone frequency (low/medium/high)

---

## 3. Projection method

1. **Baseline:** Rolling 10-game mean for each stat (and MP).
2. **Minutes:** Projected MP from baseline MP × (1 + home boost + key players out boost).
3. **Stats:** Per-minute rate = baseline stat / baseline MP.  
   Projected stat = **proj_MP × per-minute rate × (1 + adjustments)**.  
   Adjustments (multiplicative): **opponent defensive rank** (largest effect; extra penalty vs top-10 defense so we don’t overestimate vs good teams), win streak, wingspan, home/away, key players out, matchup (rim/switch/zone).
4. **Confidence bands:** For each projected stat, **range = (proj − MAE, proj + MAE)** where MAE comes from the backtest (or rolling std if no backtest). So you get e.g. `PTS: 15.8, range: (13.2, 18.4)`.
5. **Reasons:** Each projection has a short explanation string (e.g. “+ weak defense, + home, − opponent win streak”).

---

## 4. Validation results

- **Backtest:** Last 15–25 games are treated as “upcoming.” For each game, the model uses only **prior games** to form the baseline (no future or in-game info). Projections are compared to actuals.
- **Metrics (per stat):** MAE and bias; **TRB and BLK are reported first** (model focus).
- **Noise and volatility:**
  - **Is PTS noisier than TRB?** We report normalized MAE (MAE / mean actual) and std(error) per stat. Typically PTS is noisier than TRB; the report states it explicitly when the backtest supports it.
  - **Are blocks wildly volatile?** BLK usually has high normalized MAE or std(error); the report calls out when blocks are highly volatile vs other stats.
- **Does the model overestimate vs good teams?** Opponent **defensive rating is a central input**: we apply a stronger penalty vs top-10 defense (extra cut to PTS and a smaller cut to other stats) so the model does not systematically overproject vs good teams. When you supply **historical context** (opp_def_rank for past games), we report **bias vs top-10 defense** so you can check: bias &lt; 0 ⇒ model overestimates vs good teams; we tune to avoid that.
- **Segment analysis:** MAE/bias by home vs away, back-to-back, and (with historical context) top-10 vs bottom-10 defense and rim protector. So you can say e.g. “Sarr’s points projection overstates output by 18% vs top-10 defenses with a rim protector” if the data supports it.

Outputs: `sarr_backtest_validation.csv` (actuals, projections, errors) and console reports (noise/volatility, vs-good-teams, segments).

---

## 5. Limitations

- **Backtest is rolling-only:** Past-game backtest does not use opponent defensive rank or matchup flags unless you supply a historical context file (date/opponent → def_rank, rim_protector). So overall MAE is for “rolling baseline only”; segment breakdown by defense/rim is only as good as the historical data you add.
- **No in-game or lineup data:** No play-by-play, no lineup-specific rates, no real-time injury updates beyond the “key players out” flag.
- **Single player:** Built for Sarr only; multi-player or roster-wide use would be a new version (e.g. v2.0).
- **Assumptions:** Linear multiplicative adjustments; normal approximation for P(over 25 PTS); MAE used as symmetric band width.

---

## 6. Outputs and versioning

- **Next 5 games:** `sarr_next5_projections.csv` – date, opponent, venue, proj_MP, proj_PTS/TRB/AST/STL/BLK, proj_*_lo, proj_*_hi, reason_MP, reason_PTS, reason_TRB, reason_AST, reason_STL, reason_BLK.
- **Backtest:** `sarr_backtest_validation.csv` – row_idx, actuals, proj_*, *_error, and (when available) game_date, opponent, is_home, back_to_back. Optional merge with historical_context (opp_def_rank, rim_protector_present) for defense-tier and rim segments.
- **Config:** `sarr_next5_config.json` – next 5 games with opponent_defensive_rank, opponent_win_streak, venue, wingspan_advantage_inches, opponent_missing_key_players, rim_protector_present, switch_heavy_defense, zone_frequency.
- **Code version:** `sarr_prop_analysis.py` defines `__version__ = "1.0.0"`; `VERSION` file holds `1.0.0`.

**Versioning:**  
- **v1.0** – Contextual NBA stat projection model (validated). This release.  
- Planned: v1.1 (e.g. minutes regression), v1.2 (opponent archetypes), v2.0 (multi-player support). Analysts version their work.
