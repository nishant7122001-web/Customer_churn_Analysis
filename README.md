# Telco Customer Churn: Dashboard, Prediction System and SHAP Explainability

A single-file project (`churn_app.py`) that analyses the Telco Customer Churn dataset, handles the class
imbalance, compares multiple models, picks the best one, explains it with SHAP, and serves everything
through a Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt

# 1. Put WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as churn_app.py

# 2. Train and compare all models (about 2 minutes, saves to ./artifacts)
python churn_app.py train            # add --quick for a faster 3-fold run

# 3. Launch the dashboard
streamlit run churn_app.py
```

If you skip step 2, the dashboard trains automatically the first time it opens.
You can also pass a custom csv path: `python churn_app.py train path/to/file.csv`.

## Dataset

7,043 customers, 21 columns, target `Churn` (Yes / No). No missing values, no duplicate ids.

| Issue | Handling |
|---|---|
| `TotalCharges` is text and has 11 blank values | All 11 are customers with tenure 0 (new, nothing billed), set to 0 |
| "No internet service" / "No phone service" levels | Merged into "No", since `InternetService` and `PhoneService` already hold this information |
| Extra feature | `NumAddOns` = number of add-on services (security, backup, protection, support, streaming) |
| `customerID` | Dropped from the model |

## Class imbalance: analysis and handling

Churn is 26.5 percent of customers (about 2.77 retained customers per churner).
A model that predicts "no churn" for everyone would be 73.5 percent accurate and useless, so accuracy is
not used to judge models.

What the project does:

1. **Right metrics.** Models are ranked by PR-AUC, ROC-AUC and F1 at a tuned threshold. Recall and precision
   are reported separately.
2. **Stratified everything.** Train/test split (80/20) and 5-fold cross-validation are stratified.
3. **Four strategies compared for every model:** no handling, class weights, SMOTE oversampling, random undersampling.
   Resampling runs inside an `imbalanced-learn` pipeline, so only training folds are resampled and validation
   folds stay untouched (no leakage).
4. **Threshold tuning.** The decision threshold is chosen on out-of-fold training predictions (maximum F1), not fixed at 0.5.
5. **Untouched test set.** Used once, for the final report of the selected model.

### What the data showed

- The four strategies end up very close (mean PR-AUC 0.645 to 0.654 across the five models). The imbalance is
  moderate, so resampling adds little to the ranking quality.
- The real gain comes from the threshold. On the test set, the selected model at the default 0.5 threshold finds
  only 53 percent of churners (recall 0.527). At the tuned threshold it finds 71 percent (recall 0.711), with F1
  rising from 0.592 to 0.627. The cost is lower accuracy (0.807 to 0.775), which is the right trade when losing a
  customer costs more than a retention offer.
- Resampling and class weights also distort predicted probabilities. Treat the dashboard score as a risk ranking.

## Models compared

Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, each under 4 strategies (20 combinations).
Selection score = mean of cross-validated PR-AUC, ROC-AUC and tuned F1.

Result of the run included in this project: **XGBoost with no imbalance handling** ranked first
(PR-AUC 0.664, ROC-AUC 0.846, F1 0.636 in cross-validation). The top combinations are within fold-to-fold noise of
each other (Logistic Regression is almost identical), so a different seed or fold count may reorder the top few.
The dashboard always reflects whichever combination wins on your machine.

Test-set result of the selected model (threshold 0.354): ROC-AUC 0.845, PR-AUC 0.663, precision 0.560,
recall 0.711, F1 0.627. Numbers will vary slightly with library versions.

## SHAP explainability

- Tree models use `TreeExplainer`, Logistic Regression uses `LinearExplainer`, chosen automatically.
- Global view: mean absolute SHAP bar chart, beeswarm summary plot, per-feature dependence plot.
- Local view: a waterfall plot and the top five factors for each individual prediction.
- Strongest drivers in this run: month-to-month contract, short tenure, fiber optic internet, high monthly charges,
  and electronic check payment. This matches the raw data: month-to-month customers churn at 42.7 percent versus
  2.8 percent on two-year contracts, and fiber optic customers churn at 41.9 percent versus 19.0 percent on DSL.
- SHAP units are the model's raw output (log-odds for XGBoost/LightGBM, probability points for Random Forest/Extra Trees).

## Dashboard sections

| Section | Content |
|---|---|
| Overview | KPIs, churn by contract, tenure and charge distributions, revenue lost to churn |
| Churn drivers | Churn rate by any categorical feature, tenure bands, contract x internet heatmap |
| Class imbalance | Imbalance ratio, why accuracy misleads, strategy comparison, effect of threshold tuning |
| Model comparison | Full table of all 20 combinations and the selection result |
| Model performance | Threshold slider, confusion matrix, ROC and PR curves, decile lift table |
| Explainability (SHAP) | Global importance, beeswarm, dependence plots |
| Predict | Single customer form with gauge and SHAP waterfall, plus batch csv scoring with download |

## Files

```
churn_app.py        training pipeline + dashboard
requirements.txt
README.md
artifacts/          created by training: model.joblib, evaluation.joblib, cv_results.csv, summary.json
```

## Notes and limits

- The final model is trained on the 80 percent training split so that the reported test metrics are honest.
  For production, retrain on all data with the chosen settings.
- Hyperparameters are fixed, sensible defaults. A tuning step (for example Optuna on the winner) could add a little.
- SMOTE runs on one-hot encoded columns, so synthetic binary columns can take fractional values. SMOTENC is an
  alternative if you want strict handling.
- This is a single public snapshot with no time dimension, so it cannot say how the model behaves as customers change.
