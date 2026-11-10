"""
Telco Customer Churn - analysis, model selection, SHAP explainability and dashboard.

One file, two modes:

    python churn_app.py train [path/to/telco.csv] [--quick]   # trains, compares models, saves artifacts/
    streamlit run churn_app.py                                # opens the dashboard

If the dashboard is started before training, it trains automatically on first run.

Imbalance strategy (churn is about 26.5 percent of customers):
  1. Stratified splits everywhere, so every fold keeps the same churn ratio.
  2. Four strategies are compared for every model:
       No handling, Class weights, SMOTE (oversampling), Random undersampling.
     Resampling happens inside the cross-validation pipeline, so validation folds are never touched.
  3. Models are ranked with metrics that are honest under imbalance (PR-AUC, ROC-AUC, F1),
     never plain accuracy.
  4. The decision threshold is tuned on out-of-fold predictions of the training set,
     then evaluated once on a held-out test set.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
SEED = 42
TARGET = "Churn"
DATA_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns"]
CAT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService"] + ADDON_COLS + \
           ["Contract", "PaperlessBilling", "PaymentMethod"]
STRATEGIES = ["No handling", "Class weights", "SMOTE", "Undersampling"]
MODELS = ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost", "LightGBM"]


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def find_data(path=None):
    candidates = [Path(path)] if path else []
    candidates += [BASE / DATA_NAME, Path.cwd() / DATA_NAME,
                   BASE / "data" / DATA_NAME, Path("/mnt/user-data/uploads") / DATA_NAME]
    for p in candidates:
        if p.exists():
            return p
    hits = list(BASE.glob("*Telco*Churn*.csv")) + list((BASE / "data").glob("*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Place {DATA_NAME} next to churn_app.py (or pass the path).")


def load_clean(path=None):
    """Raw csv -> clean dataframe with target as 0/1 and customerID kept for reference."""
    df = pd.read_csv(find_data(path) if not isinstance(path, pd.DataFrame) else path)
    # TotalCharges has 11 blank strings: all are customers with tenure 0 (brand new, nothing billed yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    return df


def engineer(df):
    """Model features. 'No internet service' / 'No phone service' collapse to 'No':
    the information already lives in InternetService / PhoneService."""
    X = df.copy()
    for c in ADDON_COLS:
        X[c] = X[c].replace({"No internet service": "No"})
    X["MultipleLines"] = X["MultipleLines"].replace({"No phone service": "No"})
    X["NumAddOns"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
    return X[CAT_COLS + NUM_COLS]


# ----------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------
def get_estimator(name, strategy, pos_weight):
    cw = strategy == "Class weights"
    if name == "Logistic Regression":
        return LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced" if cw else None)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5, n_jobs=-1,
                                      random_state=SEED,
                                      class_weight="balanced_subsample" if cw else None)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=400, max_depth=14, min_samples_leaf=5, n_jobs=-1,
                                    random_state=SEED, class_weight="balanced" if cw else None)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                             random_state=SEED, scale_pos_weight=pos_weight if cw else 1.0)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                              verbose=-1, class_weight="balanced" if cw else None)
    raise ValueError(name)


def make_preprocessor(scale):
    return ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", NUM_COLS),
         ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CAT_COLS)],
        verbose_feature_names_out=False).set_output(transform="pandas")


def make_pipeline(name, strategy, pos_weight):
    steps = [("pre", make_preprocessor(scale=name == "Logistic Regression"))]
    if strategy == "SMOTE":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif strategy == "Undersampling":
        steps.append(("sampler", RandomUnderSampler(random_state=SEED)))
    steps.append(("clf", get_estimator(name, strategy, pos_weight)))
    return ImbPipeline(steps)


def best_threshold(y, p):
    """Threshold that maximises F1 on the given predictions."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(threshold=float(thr), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred),
                accuracy=float((pred == y).mean()),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))


# ----------------------------------------------------------------------------------------
# SHAP helpers
# ----------------------------------------------------------------------------------------
def build_explainer(pipe, background):
    import shap
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background)
    return shap.TreeExplainer(clf)


def shap_for(explainer, Xt):
    """Return (values [n, f], base_value) for the positive (churn) class, whatever the backend returns."""
    vals = explainer.shap_values(Xt)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = np.atleast_1d(explainer.expected_value)
    base = float(base[1] if len(base) > 1 else base[0])
    return vals, base


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def train(path=None, quick=False):
    ART.mkdir(exist_ok=True)
    df = load_clean(path)
    X, y = engineer(df), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    skf = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=SEED)

    print(f"Rows: {len(df)}  churn rate: {y.mean():.3f}  imbalance ratio: {pos_weight:.2f}:1")
    print(f"Train {len(Xtr)} / Test {len(Xte)} (stratified)\n")

    rows, oof_all = [], {}
    for strategy in STRATEGIES:
        for name in MODELS:
            t0 = time.time()
            oof = np.zeros(len(Xtr))
            fold_pr = []
            for tr, va in skf.split(Xtr, ytr):
                pipe = make_pipeline(name, strategy, pos_weight)
                pipe.fit(Xtr.iloc[tr], ytr.iloc[tr])
                oof[va] = pipe.predict_proba(Xtr.iloc[va])[:, 1]
                fold_pr.append(average_precision_score(ytr.iloc[va], oof[va]))
            thr, _ = best_threshold(ytr, oof)
            m = metrics_at(ytr, oof, thr)
            m.update(model=name, strategy=strategy, pr_auc_std=float(np.std(fold_pr)),
                     composite=float(np.mean([m["pr_auc"], m["roc_auc"], m["f1"]])))
            rows.append(m)
            oof_all[(name, strategy)] = oof
            print(f"{name:20s} {strategy:14s} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
                  f"F1 {m['f1']:.3f}  recall {m['recall']:.3f}  ({time.time() - t0:.0f}s)")

    cv = pd.DataFrame(rows)
    cv = cv.sort_values("composite", ascending=False).reset_index(drop=True)
    best = cv.iloc[0]
    print(f"\nBest overall: {best.model} + {best.strategy} (composite {best.composite:.3f})")

    # Refit the winner on the full training set, evaluate once on the untouched test set
    thr = best.threshold
    final = make_pipeline(best.model, best.strategy, pos_weight).fit(Xtr, ytr)
    p_te = final.predict_proba(Xte)[:, 1]
    test_m = metrics_at(yte, p_te, thr)
    default_m = metrics_at(yte, p_te, 0.5)
    print(f"Test @ tuned threshold {thr:.3f}: " + ", ".join(f"{k} {v:.3f}" for k, v in test_m.items()
                                                           if k != "threshold"))

    # Reference: a "predict nobody churns" model - high accuracy, zero value
    naive_m = dict(precision=0.0, recall=0.0, f1=0.0, balanced_accuracy=0.5,
                   accuracy=float(1 - yte.mean()), pr_auc=float(yte.mean()))

    fpr, tpr, _ = roc_curve(yte, p_te)
    prec, rec, _ = precision_recall_curve(yte, p_te)

    # SHAP on the test set
    pre = final.named_steps["pre"]
    Xte_t = pre.transform(Xte)
    Xbg = pre.transform(Xtr).sample(200, random_state=SEED)
    explainer = build_explainer(final, Xbg)
    sv, base = shap_for(explainer, Xte_t)

    model_bundle = dict(pipeline=final, threshold=float(thr), model_name=best.model,
                        strategy=best.strategy, background=Xbg, feature_columns=CAT_COLS + NUM_COLS)
    joblib.dump(model_bundle, ART / "model.joblib")
    joblib.dump(dict(shap_values=sv, base_value=base, X_test_t=Xte_t,
                     y_test=yte.values, p_test=p_te, fpr=fpr, tpr=tpr, prec=prec, rec=rec,
                     test_metrics=test_m, default_metrics=default_m, naive_metrics=naive_m,
                     test_index=Xte.index.values), ART / "evaluation.joblib")
    cv.to_csv(ART / "cv_results.csv", index=False)
    json.dump(dict(model=best.model, strategy=best.strategy, threshold=float(thr),
                   test=test_m, imbalance_ratio=pos_weight, n_train=len(Xtr), n_test=len(Xte)),
              open(ART / "summary.json", "w"), indent=2)
    print(f"\nSaved artifacts to {ART}")
    return cv


# ----------------------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------------------
def run_dashboard():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import shap
    import streamlit as st

    RED, BLUE, GREY = "#B03A2E", "#1F618D", "#7F8C8D"
    st.set_page_config(page_title="Telco Churn Analytics", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        div[data-testid="stMetric"] {background:#F6F8FA; border:1px solid #E1E4E8; padding:12px 16px; border-radius:6px;}
        div[data-testid="stMetric"] label {color:#57606A;}
        </style>""", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def load_data(src):
        return load_clean(src)

    @st.cache_resource(show_spinner=False)
    def load_artifacts():
        if not (ART / "model.joblib").exists():
            with st.spinner("First run: training and comparing models (a few minutes)..."):
                train()
        return (joblib.load(ART / "model.joblib"), joblib.load(ART / "evaluation.joblib"),
                pd.read_csv(ART / "cv_results.csv"))

    # --- data source -------------------------------------------------------------------
    try:
        data_path = find_data()
    except FileNotFoundError:
        up = st.sidebar.file_uploader("Upload the Telco churn csv", type="csv")
        if up is None:
            st.info("Upload the Telco Customer Churn csv to begin.")
            st.stop()
        data_path = pd.read_csv(up)
    df = load_data(data_path)
    bundle, ev, cv = load_artifacts()
    pipe, thr = bundle["pipeline"], bundle["threshold"]

    st.sidebar.title("Telco Churn Analytics")
    page = st.sidebar.radio("Section", ["Overview", "Churn drivers", "Class imbalance",
                                        "Model comparison", "Model performance",
                                        "Explainability (SHAP)", "Predict"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Selected model: {bundle['model_name']}")
    st.sidebar.caption(f"Imbalance strategy: {bundle['strategy']}")
    st.sidebar.caption(f"Decision threshold: {thr:.3f}")

    def style(fig, h=380):
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), template="simple_white",
                          legend_title_text="")
        return fig

    cmap = {"Retained": BLUE, "Churned": RED}
    dfv = df.copy()
    dfv["Status"] = dfv[TARGET].map({0: "Retained", 1: "Churned"})

    # ===================================================================================
    if page == "Overview":
        st.title("Customer churn overview")
        c = st.columns(5)
        c[0].metric("Customers", f"{len(df):,}")
        c[1].metric("Churn rate", f"{df[TARGET].mean():.1%}")
        c[2].metric("Avg tenure (months)", f"{df.tenure.mean():.1f}")
        c[3].metric("Avg monthly charge", f"{df.MonthlyCharges.mean():.2f}")
        lost = df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()
        c[4].metric("Monthly revenue lost", f"{lost:,.0f}", f"{lost / df.MonthlyCharges.sum():.1%} of total",
                    delta_color="off")

        a, b = st.columns(2)
        g = dfv.groupby("Contract")[TARGET].agg(["mean", "count"]).reset_index()
        fig = px.bar(g, x="Contract", y="mean", text=g["mean"].map("{:.1%}".format),
                     color_discrete_sequence=[RED], title="Churn rate by contract type")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        a.plotly_chart(style(fig), width="stretch")
        fig = px.histogram(dfv, x="tenure", color="Status", nbins=36, barmode="overlay", opacity=0.7,
                           color_discrete_map=cmap, title="Tenure distribution (months)")
        b.plotly_chart(style(fig), width="stretch")

        a, b = st.columns(2)
        fig = px.box(dfv, x="Status", y="MonthlyCharges", color="Status", color_discrete_map=cmap,
                     title="Monthly charges by status")
        fig.update_layout(showlegend=False)
        a.plotly_chart(style(fig), width="stretch")
        g = dfv.groupby("InternetService")[TARGET].mean().reset_index()
        fig = px.bar(g, x="InternetService", y=TARGET, text=g[TARGET].map("{:.1%}".format),
                     color_discrete_sequence=[RED], title="Churn rate by internet service")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        b.plotly_chart(style(fig), width="stretch")

        with st.expander("Data quality notes"):
            st.write("No missing values and no duplicate customer ids. TotalCharges is stored as text and contains "
                     "11 blank values; all belong to customers with tenure 0 (just signed up, nothing billed yet), "
                     "so they are set to 0. For the add-on columns, 'No internet service' is merged into 'No' because "
                     "InternetService already carries that information.")

    # ===================================================================================
    elif page == "Churn drivers":
        st.title("Churn drivers")
        st.caption("Churn rate by segment. The dashed line is the overall churn rate.")
        feats = [c for c in CAT_COLS if c in df.columns]
        sel = st.selectbox("Categorical feature", feats, index=feats.index("Contract"))
        g = dfv.groupby(sel)[TARGET].agg(rate="mean", customers="count").reset_index()
        fig = px.bar(g, x=sel, y="rate", text=g["rate"].map("{:.1%}".format), hover_data=["customers"],
                     color_discrete_sequence=[RED], title=f"Churn rate by {sel}")
        fig.add_hline(y=df[TARGET].mean(), line_dash="dash", line_color=GREY)
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        st.plotly_chart(style(fig), width="stretch")

        a, b = st.columns(2)
        d2 = dfv.copy()
        d2["Tenure band"] = pd.cut(d2.tenure, [-1, 6, 12, 24, 48, 72],
                                   labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
        g = d2.groupby("Tenure band", observed=True)[TARGET].mean().reset_index()
        fig = px.bar(g, x="Tenure band", y=TARGET, text=g[TARGET].map("{:.1%}".format),
                     color_discrete_sequence=[RED], title="Churn rate by tenure band (months)")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        a.plotly_chart(style(fig), width="stretch")

        piv = dfv.pivot_table(index="Contract", columns="InternetService", values=TARGET, aggfunc="mean")
        fig = px.imshow(piv, text_auto=".1%", color_continuous_scale="Reds", aspect="auto",
                        title="Churn rate: contract x internet service")
        fig.update_coloraxes(showscale=False)
        b.plotly_chart(style(fig), width="stretch")

        num = st.selectbox("Numeric feature", ["MonthlyCharges", "tenure", "TotalCharges"])
        fig = px.histogram(dfv, x=num, color="Status", nbins=40, barmode="overlay", opacity=0.7,
                           histnorm="probability density", color_discrete_map=cmap,
                           title=f"{num} distribution, churned vs retained")
        st.plotly_chart(style(fig), width="stretch")

    # ===================================================================================
    elif page == "Class imbalance":
        st.title("Class imbalance analysis")
        ratio = (df[TARGET] == 0).sum() / (df[TARGET] == 1).sum()
        a, b, c = st.columns(3)
        a.metric("Retained", f"{(df[TARGET] == 0).sum():,}", f"{1 - df[TARGET].mean():.1%}", delta_color="off")
        b.metric("Churned", f"{(df[TARGET] == 1).sum():,}", f"{df[TARGET].mean():.1%}", delta_color="off")
        c.metric("Imbalance ratio", f"{ratio:.2f} : 1")

        st.subheader("Why plain accuracy is misleading here")
        st.write(f"A model that predicts 'no churn' for everyone scores {1 - df[TARGET].mean():.1%} accuracy and catches "
                 "zero churners. This project therefore ranks models on PR-AUC (focuses on the minority class), "
                 "ROC-AUC (ranking quality) and F1 at a tuned threshold, and reports recall and precision separately.")

        st.subheader("What was done about it")
        st.markdown(
            "- Stratified train/test split and stratified cross-validation, so every fold keeps the 26.5 percent churn share.\n"
            "- Four strategies tested on every model: no handling, class weights, SMOTE oversampling, random undersampling.\n"
            "- Resampling is applied only inside training folds (imbalanced-learn pipeline), so there is no leakage "
            "into validation data.\n"
            "- The decision threshold is tuned on out-of-fold predictions instead of using the default 0.5.\n"
            "- The test set is left untouched until the single final evaluation.")

        piv = cv.pivot(index="model", columns="strategy", values="pr_auc")[STRATEGIES]
        fig = px.imshow(piv, text_auto=".3f", color_continuous_scale="Blues", aspect="auto",
                        title="Cross-validated PR-AUC: model x imbalance strategy")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig, 340), width="stretch")

        piv = cv.pivot(index="model", columns="strategy", values="recall")[STRATEGIES]
        fig = px.imshow(piv, text_auto=".3f", color_continuous_scale="Reds", aspect="auto",
                        title="Cross-validated recall at each combination's tuned threshold")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig, 340), width="stretch")

        st.subheader("What the comparison shows")
        ranked = cv.groupby("strategy")[["pr_auc", "roc_auc", "f1", "recall"]].mean().reindex(STRATEGIES)
        st.dataframe(ranked.style.format("{:.3f}"), width="stretch")
        spread = cv.groupby("model").pr_auc.apply(lambda s: s.max() - s.min()).max()
        st.write(f"Averaged over the five models, the four strategies end up very close (largest PR-AUC gap within "
                 f"one model is {spread:.3f}). On this dataset the imbalance is moderate, so the biggest gain does not "
                 "come from resampling but from moving the decision threshold away from 0.5. Resampling and class "
                 "weights change the ranking very little, while they distort the predicted probabilities. "
                 "The selection therefore simply follows the composite score.")

        st.subheader("Effect of threshold tuning on the selected model (test set)")
        nm, dm, tm = ev["naive_metrics"], ev["default_metrics"], ev["test_metrics"]
        t = pd.DataFrame({
            "Predict no churn for everyone": nm,
            f"{bundle['model_name']}, default threshold 0.50": dm,
            f"{bundle['model_name']}, tuned threshold {thr:.2f}": tm,
        }).T[["precision", "recall", "f1", "balanced_accuracy", "accuracy"]]
        st.dataframe(t.style.format("{:.3f}"), width="stretch")
        st.caption(f"Recall rises from {dm['recall']:.2f} to {tm['recall']:.2f} when the threshold is tuned. "
                   f"Accuracy moves from {dm['accuracy']:.2f} to {tm['accuracy']:.2f}, which is the price of catching "
                   "more churners and is a good trade when losing a customer costs more than a retention offer.")

    # ===================================================================================
    elif page == "Model comparison":
        st.title("Model comparison")
        st.write("All combinations were evaluated with stratified cross-validation on the training set. "
                 "The composite score is the mean of PR-AUC, ROC-AUC and tuned F1. The highest composite wins.")
        show = cv[["model", "strategy", "pr_auc", "pr_auc_std", "roc_auc", "f1", "precision", "recall",
                   "balanced_accuracy", "threshold", "composite"]].copy()
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c not in ("model", "strategy")})
                     .highlight_max(subset=["composite"], color="#D6EAF8"), width="stretch", height=420)
        top = cv.copy()
        top["label"] = top.model + " | " + top.strategy
        top = top.head(10).iloc[::-1]
        fig = px.bar(top, x="composite", y="label", orientation="h", color_discrete_sequence=[BLUE],
                     title="Top 10 combinations by composite score")
        fig.update_xaxes(range=[top.composite.min() - 0.03, top.composite.max() + 0.01])
        fig.update_yaxes(title="")
        st.plotly_chart(style(fig, 420), width="stretch")
        best = cv.iloc[0]
        st.success(f"Selected: {best.model} with {best.strategy} (composite {best.composite:.3f}, "
                   f"PR-AUC {best.pr_auc:.3f}, ROC-AUC {best.roc_auc:.3f}, F1 {best.f1:.3f}).")
        st.caption("Differences between the top few combinations are often within the fold-to-fold noise "
                   "(see pr_auc_std). Simpler models are not penalised or favoured, the ranking is purely by score.")

    # ===================================================================================
    elif page == "Model performance":
        st.title("Model performance on the held-out test set")
        y, p = ev["y_test"], ev["p_test"]
        t = st.slider("Decision threshold", 0.05, 0.95, float(round(thr, 2)), 0.01,
                      help="Lower threshold catches more churners but flags more customers that would have stayed.")
        m = metrics_at(y, p, t)
        c = st.columns(5)
        c[0].metric("Recall", f"{m['recall']:.3f}")
        c[1].metric("Precision", f"{m['precision']:.3f}")
        c[2].metric("F1", f"{m['f1']:.3f}")
        c[3].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        c[4].metric("PR-AUC", f"{m['pr_auc']:.3f}")

        a, b = st.columns(2)
        cm = confusion_matrix(y, (p >= t).astype(int))
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Pred retained", "Pred churn"],
                        y=["Actual retained", "Actual churn"], title="Confusion matrix")
        fig.update_coloraxes(showscale=False)
        a.plotly_chart(style(fig), width="stretch")

        ts = np.linspace(0.05, 0.95, 91)
        curve = pd.DataFrame({"threshold": ts,
                              "Precision": [precision_score(y, p >= x, zero_division=0) for x in ts],
                              "Recall": [recall_score(y, p >= x) for x in ts],
                              "F1": [f1_score(y, p >= x) for x in ts]}).melt("threshold")
        fig = px.line(curve, x="threshold", y="value", color="variable", title="Precision, recall and F1 vs threshold",
                      color_discrete_sequence=[BLUE, RED, GREY])
        fig.add_vline(x=t, line_dash="dash", line_color="black")
        a2 = b.plotly_chart(style(fig), width="stretch")

        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=ev["fpr"], y=ev["tpr"], line=dict(color=BLUE), name="Model")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dash"), name="Random")
        fig.update_layout(title=f"ROC curve (AUC {m['roc_auc']:.3f})", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        a.plotly_chart(style(fig), width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=ev["rec"], y=ev["prec"], line=dict(color=RED), name="Model")
        fig.add_hline(y=y.mean(), line_dash="dash", line_color=GREY, annotation_text="Baseline = churn rate")
        fig.update_layout(title=f"Precision-recall curve (AP {m['pr_auc']:.3f})", xaxis_title="Recall",
                          yaxis_title="Precision")
        b.plotly_chart(style(fig), width="stretch")

        st.subheader("Targeting value: which customers to contact first")
        d = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
        d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11))
        gt = d.groupby("decile", observed=True).y.agg(churners="sum", customers="count")
        gt["churn rate"] = gt.churners / gt.customers
        gt["cumulative share of all churners"] = gt.churners.cumsum() / d.y.sum()
        gt["lift"] = gt["churn rate"] / d.y.mean()
        st.dataframe(gt.style.format({"churn rate": "{:.1%}", "cumulative share of all churners": "{:.1%}",
                                      "lift": "{:.2f}"}), width="stretch")
        top20 = gt["cumulative share of all churners"].iloc[1]
        st.caption(f"Contacting only the top 20 percent highest-risk customers reaches {top20:.0%} of all churners.")

    # ===================================================================================
    elif page == "Explainability (SHAP)":
        st.title("Model explainability with SHAP")
        sv, Xt = ev["shap_values"], ev["X_test_t"]
        st.write("SHAP values show how much each feature pushed a customer's churn score up (positive) or down "
                 "(negative) relative to the average customer.")
        imp = pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(0)}) \
            .sort_values("mean_abs_shap", ascending=False)
        n = st.slider("Features to show", 5, min(25, len(imp)), 12)
        a, b = st.columns([1, 1.2])
        fig = px.bar(imp.head(n).iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                     color_discrete_sequence=[BLUE], title="Global importance (mean absolute SHAP)")
        fig.update_yaxes(title="")
        fig.update_xaxes(title="")
        a.plotly_chart(style(fig, 460), width="stretch")
        with b:
            plt.figure()
            shap.summary_plot(sv, Xt, max_display=n, show=False, plot_size=(7, 5.5))
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("Each dot is a customer. Red means a high feature value, blue a low one. "
                       "Dots to the right raise churn risk.")

        st.subheader("Feature dependence")
        f = st.selectbox("Feature", list(imp.feature), index=0)
        fig = px.scatter(x=Xt[f], y=sv[:, list(Xt.columns).index(f)], opacity=0.5,
                         color_discrete_sequence=[RED], labels={"x": f, "y": "SHAP value"},
                         title=f"SHAP value of {f}")
        fig.add_hline(y=0, line_dash="dash", line_color=GREY)
        st.plotly_chart(style(fig), width="stretch")
        st.caption("SHAP units are the model's raw output: log-odds for XGBoost and LightGBM, "
                   "probability points for Random Forest and Extra Trees.")

    # ===================================================================================
    else:
        st.title("Predict customer churn")
        explainer = build_explainer(pipe, bundle["background"])
        tab1, tab2 = st.tabs(["Single customer", "Batch scoring (csv)"])

        def risk_band(s):
            return "High" if s >= thr else ("Medium" if s >= 0.6 * thr else "Low")

        with tab1:
            med = df.median(numeric_only=True)
            with st.form("customer"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Account**")
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    tenure = st.number_input("Tenure (months)", 0, 72, 12)
                    monthly = st.number_input("Monthly charges", 0.0, 200.0, float(round(med.MonthlyCharges, 2)))
                    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                    pay = st.selectbox("Payment method", sorted(df.PaymentMethod.unique()))
                with c2:
                    st.markdown("**Services**")
                    phone = st.selectbox("Phone service", ["Yes", "No"])
                    lines = st.selectbox("Multiple lines", ["No", "Yes"])
                    net = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
                    sec = st.selectbox("Online security", ["No", "Yes"])
                    bak = st.selectbox("Online backup", ["No", "Yes"])
                with c3:
                    st.markdown("**Add-ons and profile**")
                    dev = st.selectbox("Device protection", ["No", "Yes"])
                    tech = st.selectbox("Tech support", ["No", "Yes"])
                    stv = st.selectbox("Streaming TV", ["No", "Yes"])
                    smov = st.selectbox("Streaming movies", ["No", "Yes"])
                    senior = st.selectbox("Senior citizen", ["No", "Yes"])
                    partner = st.selectbox("Partner", ["No", "Yes"])
                    dep = st.selectbox("Dependents", ["No", "Yes"])
                    gender = st.selectbox("Gender", ["Female", "Male"])
                submitted = st.form_submit_button("Predict churn risk")

            if submitted:
                total = round(tenure * monthly, 2)
                raw = pd.DataFrame([dict(
                    gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dep, tenure=tenure,
                    PhoneService=phone, MultipleLines=lines if phone == "Yes" else "No phone service",
                    InternetService=net,
                    OnlineSecurity=sec, OnlineBackup=bak, DeviceProtection=dev, TechSupport=tech,
                    StreamingTV=stv, StreamingMovies=smov, Contract=contract, PaperlessBilling=paperless,
                    PaymentMethod=pay, MonthlyCharges=monthly, TotalCharges=total)])
                if net == "No":
                    for c in ADDON_COLS:
                        raw[c] = "No internet service"
                X1 = engineer(raw)
                score = float(pipe.predict_proba(X1)[0, 1])
                band = risk_band(score)

                a, b = st.columns([1, 1.6])
                with a:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=score * 100, number={"suffix": "%", "valueformat": ".1f"},
                        title={"text": "Churn risk score"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if band == "High" else BLUE},
                               "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
                    st.plotly_chart(style(fig, 300), width="stretch")
                    st.metric("Risk band", band)
                    st.caption(f"Black line = decision threshold ({thr:.0%}). Scores at or above it are flagged as "
                               "likely churners. Because imbalance handling reshapes the scores, treat the number as a "
                               "risk ranking rather than a literal probability.")
                with b:
                    Xt1 = pipe.named_steps["pre"].transform(X1)
                    sv1, base1 = shap_for(explainer, Xt1)
                    exp = shap.Explanation(values=sv1[0], base_values=base1, data=Xt1.iloc[0].values,
                                           feature_names=list(Xt1.columns))
                    plt.figure()
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(plt.gcf(), clear_figure=True)
                order = np.argsort(-np.abs(sv1[0]))[:5]
                st.markdown("**Top factors for this customer**")
                for i in order:
                    direction = "raises" if sv1[0][i] > 0 else "lowers"
                    st.write(f"- {Xt1.columns[i]} = {Xt1.iloc[0, i]:.4g} {direction} churn risk ({sv1[0][i]:+.3f})")

        with tab2:
            st.write("Upload a csv with the same columns as the original dataset (Churn column optional).")
            up = st.file_uploader("Customer file", type="csv", key="batch")
            if up is not None:
                new = load_clean(pd.read_csv(up))
                missing = [c for c in CAT_COLS + ["tenure", "MonthlyCharges", "TotalCharges"] if c not in new.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    s = pipe.predict_proba(engineer(new))[:, 1]
                    out = new.copy()
                    out["churn_score"] = s
                    out["flagged"] = (s >= thr).astype(int)
                    out["risk_band"] = [risk_band(v) for v in s]
                    out = out.sort_values("churn_score", ascending=False)
                    st.write(f"{int(out.flagged.sum())} of {len(out)} customers flagged at threshold {thr:.2f}.")
                    cols = (["customerID"] if "customerID" in out.columns else []) + \
                           ["churn_score", "risk_band", "flagged", "Contract", "tenure", "MonthlyCharges"]
                    st.dataframe(out[cols].head(200), width="stretch")
                    st.download_button("Download scored file", out.to_csv(index=False).encode(),
                                       "churn_scored.csv", "text/csv")


# ----------------------------------------------------------------------------------------
def _in_streamlit():
    import logging
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():
        run_dashboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        train(args[0] if args else None, quick="--quick" in sys.argv)
    else:
        print(__doc__)
