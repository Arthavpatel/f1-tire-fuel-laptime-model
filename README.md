# F1 Tire, Fuel & Lap Time Model

This repository contains code and artifacts for modeling Formula 1 lap times as a function of tire state, fuel load, and other race variables. It includes data-prep, modeling notebooks, evaluation scripts, and a small web demo (if included) to interactively explore predictions.

## Overview
Goals:
- Model lap time behavior given tire age, fuel, stint, and track/environmental variables.
- Compare model fits and evaluate predictive power.
- Provide visualizations and a lightweight demo to explore the model interactively.

Languages used:
- Python for data processing and modeling (~40%)
- JavaScript/HTML for visualization or front-end demo (~60% combined)

## Repository layout
- data/ — raw and processed datasets
- notebooks/ — Jupyter notebooks for EDA and modeling experiments
- src/ or scripts/ — training and evaluation scripts (train.py, predict.py)
- web/ or demo/ — small front-end demo (JS/HTML) if present
- results/ — trained models, metrics, and plots

## Quickstart

1. Python environment
   ```
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Prepare data
   - Place any raw data in `data/raw/`.
   - Run preprocessing:
   ```
   python scripts/preprocess.py --input data/raw --output data/processed
   ```

3. Train model
   ```
   python scripts/train.py --data data/processed --model-out results/model.pkl
   ```

4. Predict / evaluate
   ```
   python scripts/predict.py --model results/model.pkl --input data/processed/test.csv --out results/predictions.csv
   python scripts/evaluate.py --predictions results/predictions.csv --metrics results/metrics.json
   ```

5. Launch demo (if web demo included)
   - If there's a static demo in `web/`, you can serve it locally:
   ```
   # using a simple http server
   cd web
   python -m http.server 8000
   # then open http://localhost:8000
   ```

## Modeling details
- Typical features:
  - Tire age (laps on tire), tire compound
  - Fuel load (kg or laps remaining)
  - Track temperature, weather
  - Driver and car identifiers (if used)
- Possible model types:
  - Linear regression with interaction terms
  - Tree-based models (Random Forest, XGBoost)
  - Time-series or rolling-window models to capture stint effects

## Evaluation
- Use RMSE / MAE for continuous lap time errors.
- Visual checks: predicted vs actual lap time scatter, residual plots across tire age and fuel.
- Compare models by cross-validation and out-of-sample laps.

## Reproducibility
- Fix random seeds in scripts and notebook cells.
- Save preprocessing steps so the same features can be recreated.

## Extending the project
- Add driver- and track-specific calibrations.
- Model pit-stop effects and stint transitions explicitly.
- Deploy a Streamlit or Flask app for an interactive model playground.

## License
Add your license here (e.g., MIT).

## Contact
Arthav Patel — GitHub: [@Arthavpatel](https://github.com/Arthavpatel)
