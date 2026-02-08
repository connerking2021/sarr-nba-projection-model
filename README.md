# Alexandre Sarr NBA Projection Model (v1.0)

A contextual NBA stat projection model focused on **rebounds (TRB) and blocks (BLK)**, designed to analyze player performance, validate projections via backtesting, and generate next-game forecasts with confidence bands.

This project uses **Alexandre Sarr’s 2024–25 game log** as a case study, but the framework is extensible to other players.

---

## 📌 Project Objective

The goal of this model is to:
- Project per-game NBA box score stats (PTS, TRB, AST, STL, BLK)
- Emphasize **defensive impact stats** (rebounds and blocks)
- Avoid overestimation against strong defensive opponents
- Quantify uncertainty using backtesting and confidence bands

The model is intended for:
- Sports analytics portfolios
- Front office / scouting discussions
- Player prop research and validation
- Learning applied modeling and evaluation techniques

---

## 🧠 Model Overview

### Core Concepts
- **Rolling Baseline**: Uses recent games to establish per-minute production
- **Minutes as First-Class Input**: Projects stats via projected minutes × per-minute rates
- **Opponent Context**:
  - Defensive rating adjustment
  - Extra penalty vs top-10 defenses
- **Game Context**:
  - Home vs away
  - Back-to-back games
- **Explanation Layer**:
  - Each projection includes `reason_*` fields explaining why it moved

### Model Emphasis
> The model explicitly **prides itself on rebounds (TRB) and blocks (BLK)**, where opponent defensive quality and rim presence produce the largest shifts.

---

## 📊 Validation & Backtesting

### Backtest Design
- Rolling backtest over the last **15–25 games**
- Each game is projected using **only prior data**
- Metrics:
  - Mean Absolute Error (MAE)
  - Bias (over/underestimation)
  - Normalized MAE (scale-invariant noise measure)

### Key Findings
- **PTS is noisier than TRB**, confirmed via normalized MAE
- **BLK is highly volatile**, with higher relative error
- Defensive matchups significantly impact PTS accuracy
- Extra penalty vs top-10 defenses reduces overestimation

Segmented reports include:
- Home vs Away
- Back-to-back vs Rest
- Top-10 vs Bottom-10 defenses (when context provided)

---

## 📈 Outputs

After running the model, the following files are generated:

### 1. `sarr_next5_projections.csv`
Contains projections for the next 5 games:
- `proj_*` columns (PTS, TRB, AST, STL, BLK)
- Confidence bands: `proj_*_lo`, `proj_*_hi`
- Explanation columns: `reason_*`

### 2. `sarr_backtest_validation.csv`
Backtest results including:
- Actual vs projected stats
- Error columns per stat
- Game segmentation fields

---
