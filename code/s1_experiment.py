#!/usr/bin/env python3
"""
S3 — ML per-user (NO aggregation across users), leakage-safe feature engineering.

Unit   = one user-row (per question).
Target = ActualAnswer.
Split  = leave-one-version-out by SurveyVersion (LOVO).
Features = per-user meta + engineered behavioral features only (clicks/times/confidence/human vote/user meta).
           No Santos/Starnie/D3L (dropped).

Group ablations:
  CLICK        : LastClick, IsSingleClick, ClickCount, TimeDiff_FCnLC
  USER_META    : Browser, OS, Age, Education, EngProf, Major, ResolutionLen
  HUMAN_LABEL  : IsExp, ExplanationsLen, ExplanationsLen_scaled, SurveyAnswer01
  DECISIONTIME : DecisionTime, OverallSurveyDT,
                 MIDT, ISDT, MEDT, ESDT, DecisionTimeFract, DecisionTimeFract_scaled
  CONFIDENCE   : ConfidenceLevel, MICL, ISCL, MECL, ESCL, ConfidenceLevel_Dec
  QUANTIFIED   : Majority, NoCY, NoCN, Diff_CAns

Overall metrics for each (model, setup) are computed as
**macro averages over LOVO folds (SurveyVersion)**, i.e., each version contributes
equally to the final accuracy/F1/etc.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---------------- helpers ----------------

def to01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes":1, "y":1, "1":1, "true":1, "t":1, "unionable":1,
        "no":0,  "n":0, "0":0, "false":0, "f":0, "non-unionable":0,
        "nonunionable":0, "non_unionable":0
    }
    out = s.map(mapping)
    if out.isna().mean() > 0.5:  
        out = pd.to_numeric(series, errors="coerce")
    return out

def _minmax_scale(series, vmin, vmax):
    series = series.astype(float)
    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(0.0, index=series.index)
    denom = float(vmax - vmin)
    if denom == 0:
        return pd.Series(0.0, index=series.index)
    return (series - vmin) / denom

def engineer_features(train_df, test_df, uid_col, qid_col, ver_col):
    """
    Given raw per-response data for train/test, compute leakage-safe engineered features.

    Uses TRAIN rows only to compute:
      - MIDT, MICL, ISDT, ISCL (user-internal timing / confidence)
      - MEDT, MECL, ESDT, ESCL (question-external timing / confidence)
      - DecisionTimeFract, DecisionTimeFract_scaled
      - ExplanationsLen_scaled
      - ConfidenceLevel_Dec
      - Majority, NoCY, NoCN, Diff_CAns

    Returns:
      X_tr, X_te, feature_cols
    """
    tr = train_df.copy()
    te = test_df.copy()

    # required base columns
    for col in ["DecisionTime","OverallSurveyDT","ClickCount","ConfidenceLevel","ExplanationsLen"]:
        if col not in tr.columns:
            raise ValueError(f"Missing required base column '{col}' in train_df")
    for col in ["SurveyAnswer","ActualAnswer"]:
        if col not in tr.columns:
            raise ValueError(f"Missing required column '{col}' for QUANTIFIED features")

    # 1) Map answers to 0/1 for internal use 
    tr["SurveyAnswer01"] = to01(tr["SurveyAnswer"])
    te["SurveyAnswer01"] = to01(te["SurveyAnswer"])
    tr["ActualAnswer01"] = to01(tr["ActualAnswer"])
    te["ActualAnswer01"] = to01(te["ActualAnswer"])

    # 2) DecisionTimeFract + scaled (min-max based on TRAIN)
    tr["DecisionTimeFract"] = tr["DecisionTime"] / tr["OverallSurveyDT"].replace(0, np.nan)
    te["DecisionTimeFract"] = te["DecisionTime"] / te["OverallSurveyDT"].replace(0, np.nan)

    dtf_min = tr["DecisionTimeFract"].min(skipna=True)
    dtf_max = tr["DecisionTimeFract"].max(skipna=True)
    tr["DecisionTimeFract_scaled"] = _minmax_scale(tr["DecisionTimeFract"], dtf_min, dtf_max)
    te["DecisionTimeFract_scaled"] = _minmax_scale(te["DecisionTimeFract"], dtf_min, dtf_max)

    # 3) ExplanationsLen_scaled (min-max based on TRAIN)
    exp_min = tr["ExplanationsLen"].min(skipna=True)
    exp_max = tr["ExplanationsLen"].max(skipna=True)
    tr["ExplanationsLen_scaled"] = _minmax_scale(tr["ExplanationsLen"], exp_min, exp_max)
    te["ExplanationsLen_scaled"] = _minmax_scale(te["ExplanationsLen"], exp_min, exp_max)

    # 4) ConfidenceLevel_Dec (row-local)
    tr["ConfidenceLevel_Dec"] = tr["ConfidenceLevel"] / 100.0
    te["ConfidenceLevel_Dec"] = te["ConfidenceLevel"] / 100.0

    # 5) User-level internal aggregates: MIDT, MICL, ISDT, ISCL 
    user_midt = tr.groupby(uid_col)["DecisionTime"].mean()
    user_micl = tr.groupby(uid_col)["ConfidenceLevel"].mean()
    user_dt_min = tr.groupby(uid_col)["DecisionTime"].min()
    user_dt_max = tr.groupby(uid_col)["DecisionTime"].max()
    user_cl_min = tr.groupby(uid_col)["ConfidenceLevel"].min()
    user_cl_max = tr.groupby(uid_col)["ConfidenceLevel"].max()

    def add_user_internal(df):
        df = df.copy()
        df["MIDT"] = df[uid_col].map(user_midt)
        df["MICL"] = df[uid_col].map(user_micl)

        dt_min = df[uid_col].map(user_dt_min)
        dt_max = df[uid_col].map(user_dt_max)
        cl_min = df[uid_col].map(user_cl_min)
        cl_max = df[uid_col].map(user_cl_max)

        dt_range = (dt_max - dt_min).replace(0, np.nan)
        cl_range = (cl_max - cl_min).replace(0, np.nan)

        df["ISDT"] = (df["DecisionTime"] - dt_min) / dt_range
        df["ISCL"] = (df["ConfidenceLevel"] - cl_min) / cl_range
        return df

    tr = add_user_internal(tr)
    te = add_user_internal(te)

    # 6) Question-level external aggregates: MEDT, MECL, ESDT, ESCL 
    group_cols = [ver_col, qid_col]
    q_medt = tr.groupby(group_cols)["DecisionTime"].mean()
    q_mecl = tr.groupby(group_cols)["ConfidenceLevel"].mean()
    q_dt_min = tr.groupby(group_cols)["DecisionTime"].min()
    q_dt_max = tr.groupby(group_cols)["DecisionTime"].max()
    q_cl_min = tr.groupby(group_cols)["ConfidenceLevel"].min()
    q_cl_max = tr.groupby(group_cols)["ConfidenceLevel"].max()

    def add_question_external(df):
        df = df.copy()
        keys = list(zip(df[ver_col], df[qid_col]))
        medt_vals, mecl_vals = [], []
        dt_min_vals, dt_max_vals = [], []
        cl_min_vals, cl_max_vals = [], []
        for k in keys:
            medt_vals.append(q_medt.get(k, np.nan))
            mecl_vals.append(q_mecl.get(k, np.nan))
            dt_min_vals.append(q_dt_min.get(k, np.nan))
            dt_max_vals.append(q_dt_max.get(k, np.nan))
            cl_min_vals.append(q_cl_min.get(k, np.nan))
            cl_max_vals.append(q_cl_max.get(k, np.nan))
        df["MEDT"] = medt_vals
        df["MECL"] = mecl_vals

        dt_min_s = pd.Series(dt_min_vals, index=df.index)
        dt_max_s = pd.Series(dt_max_vals, index=df.index)
        cl_min_s = pd.Series(cl_min_vals, index=df.index)
        cl_max_s = pd.Series(cl_max_vals, index=df.index)

        dt_range = (dt_max_s - dt_min_s).replace(0, np.nan)
        cl_range = (cl_max_s - cl_min_s).replace(0, np.nan)

        df["ESDT"] = (df["DecisionTime"] - dt_min_s) / dt_range
        df["ESCL"] = (df["ConfidenceLevel"] - cl_min_s) / cl_range
        return df

    tr = add_question_external(tr)
    te = add_question_external(te)

    # 7) QUANTIFIED features (Majority, NoCY, NoCN, Diff_CAns) 
    # Majority per (version, question) 
    maj_vals = tr.groupby(group_cols)["SurveyAnswer01"].agg(
        lambda x: 1 if (x.mean() > 0.5) else (0 if x.mean() < 0.5 else 0.5)
    )

    def map_majority(df):
        keys = list(zip(df[ver_col], df[qid_col]))
        return [maj_vals.get(k, np.nan) for k in keys]

    tr["Majority"] = map_majority(tr)
    te["Majority"] = map_majority(te)

    # Per-user correctness counts (NoCY, NoCN, Diff_CAns)
    tr["is_correct"] = (tr["SurveyAnswer01"] == tr["ActualAnswer01"]).astype(int)
    mask_cy = (tr["is_correct"] == 1) & (tr["SurveyAnswer01"] == 1)
    mask_cn = (tr["is_correct"] == 1) & (tr["SurveyAnswer01"] == 0)
    no_cy = tr[mask_cy].groupby(uid_col)["SurveyAnswer01"].size()
    no_cn = tr[mask_cn].groupby(uid_col)["SurveyAnswer01"].size()

    def add_correct_counts(df):
        df = df.copy()
        df["NoCY"] = df[uid_col].map(no_cy).fillna(0).astype(float)
        df["NoCN"] = df[uid_col].map(no_cn).fillna(0).astype(float)
        df["Diff_CAns"] = df["NoCY"] - df["NoCN"]
        return df

    tr = add_correct_counts(tr)
    te = add_correct_counts(te)

    # 8) Final feature list
    feature_cols = [
        # CLICK
        "LastClick", "IsSingleClick", "ClickCount", "TimeDiff_FCnLC",
        # USER META
        "Browser", "OS", "Age", "Education", "EngProf", "Major", "ResolutionLen",
        # HUMAN LABEL
        "IsExp", "ExplanationsLen", "ExplanationsLen_scaled", "SurveyAnswer01",
        # DECISION TIME
        "DecisionTime", "OverallSurveyDT",
        "MIDT", "ISDT", "MEDT", "ESDT", "DecisionTimeFract", "DecisionTimeFract_scaled",
        # CONFIDENCE
        "ConfidenceLevel", "MICL", "ISCL", "MECL", "ESCL", "ConfidenceLevel_Dec",
        # QUANTIFIED
        "Majority", "NoCY", "NoCN", "Diff_CAns",
    ]

    missing = [c for c in feature_cols if c not in tr.columns]
    if missing:
        raise ValueError(f"engineer_features: missing columns after engineering: {missing}")

    X_tr = tr[feature_cols].copy()
    X_te = te[feature_cols].copy()
    return X_tr, X_te, feature_cols

def per_class_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec_yes = 0.0 if (tp+fp)==0 else tp/(tp+fp)
    rec_yes  = 0.0 if (tp+fn)==0 else tp/(tp+fn)
    f1_yes   = 0.0 if (prec_yes+rec_yes)==0 else 2*prec_yes*rec_yes/(prec_yes+rec_yes)
    prec_no  = 0.0 if (tn+fn)==0 else tn/(tn+fn)
    rec_no   = 0.0 if (tn+fp)==0 else tn/(tn+fp)
    f1_no    = 0.0 if (prec_no+rec_no)==0 else 2*prec_no*rec_no/(prec_no+rec_no)
    tnr      = 0.0 if (tn+fp)==0 else tn/(tn+fp)
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision_pos_yes": round(prec_yes,4),
        "recall_pos_yes": round(rec_yes,4),
        "f1_pos_yes": round(f1_yes,4),
        "precision_pos_no": round(prec_no,4),
        "recall_pos_no": round(rec_no,4),
        "f1_pos_no": round(f1_no,4),
        "specificity_tnr": round(tnr,4),
    }

def compute_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)),4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)),4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)),4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)),4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")),4),
        "precision_micro": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)),4),
        "recall_micro": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)),4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro")),4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")),4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)),4),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)),4)
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    out.update(per_class_metrics(y_true, y_pred))
    return out

def safe_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:,1]
    except Exception:
        try:
            from scipy.special import expit
            return expit(pipe.decision_function(X))
        except Exception:
            return None

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="S3 per-user ML (CSV, real user id), leakage-safe feature engineering + group ablations (macro-averaged over LOVO folds)"
    )
    ap.add_argument("--csv", required=True, help="CSV file (e.g., FE_Responses_Final.csv)")
    ap.add_argument("--sep", default=None, help="CSV delimiter (e.g., ';'); if omitted, auto-detect")
    ap.add_argument("--qid", default="QuestionNum")
    ap.add_argument("--version", default="SurveyVersion")
    ap.add_argument("--gold", default="ActualAnswer")
    ap.add_argument("--user", required=True, help="User identifier column (e.g., ByWho)")
    ap.add_argument("--models", default="lr,knn,rf,xgb")
    ap.add_argument("--ablation", default="none", choices=["none","drop_group","only_group"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # load CSV
    df = pd.read_csv(
        args.csv,
        sep=(None if args.sep is None else args.sep),
        engine=("python" if args.sep is None else None),
    )

    # required cols
    for c in [args.qid, args.version, args.gold, args.user]:
        if c not in df.columns:
            raise SystemExit(f"✗ Missing column '{c}'. Found: {list(df.columns)}")

    # drop TUS features 
    df = df.drop(columns=[c for c in ["Santos","Starnie","D3L","santos","starmie","d3l"] if c in df.columns], errors="ignore")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # target
    y_series = to01(df[args.gold])
    keep = ~y_series.isna()
    if keep.sum() == 0:
        raise SystemExit("✗ No valid target rows after mapping to 0/1.")
    df = df.loc[keep].copy()
    y_series = y_series.loc[keep].astype(int)

    ver = args.version
    qid = args.qid
    uid = args.user

    versions = sorted(df[ver].dropna().unique().tolist())
    if not versions:
        raise SystemExit("✗ No SurveyVersion values found.")

    # Precompute engineered features per LOVO fold
    fold_data = {}
    feature_cols = None
    for holdout in versions:
        mask_te = (df[ver] == holdout)
        df_tr_raw = df.loc[~mask_te].copy()
        df_te_raw = df.loc[ mask_te].copy()
        if df_tr_raw.empty or df_te_raw.empty:
            continue

        X_tr_full, X_te_full, feat_cols = engineer_features(df_tr_raw, df_te_raw, uid, qid, ver)
        y_tr = y_series.loc[~mask_te].astype(int).values
        y_te = y_series.loc[ mask_te].astype(int).values
        meta = df.loc[mask_te, [qid, ver, uid]].reset_index(drop=True)

        fold_data[holdout] = {
            "X_tr": X_tr_full,
            "X_te": X_te_full,
            "y_tr": y_tr,
            "y_te": y_te,
            "meta": meta,
        }
        feature_cols = feat_cols  # all folds share same feature set

    if not fold_data:
        raise SystemExit("✗ No non-empty LOVO folds could be constructed.")

    # Define feature groups (by name) on engineered feature set
    CLICK        = ["LastClick","IsSingleClick","ClickCount","TimeDiff_FCnLC"]
    USER_META    = ["Browser","OS","Age","Education","EngProf","Major","ResolutionLen"]
    HUMAN_LABEL  = ["IsExp","ExplanationsLen","ExplanationsLen_scaled","SurveyAnswer01"]
    DECISIONTIME = ["DecisionTime","OverallSurveyDT",
                    "MIDT","ISDT","MEDT","ESDT","DecisionTimeFract","DecisionTimeFract_scaled"]
    CONFIDENCE   = ["ConfidenceLevel","MICL","ISCL","MECL","ESCL","ConfidenceLevel_Dec"]
    QUANTIFIED   = ["Majority","NoCY","NoCN","Diff_CAns"]

    groups = {
        "CLICK": CLICK,
        "USER_META": USER_META,
        "HUMAN_LABEL": HUMAN_LABEL,
        "DECISIONTIME": DECISIONTIME,
        "CONFIDENCE": CONFIDENCE,
        "QUANTIFIED": QUANTIFIED
    }

    # final feature universe 
    all_feats = [c for c in feature_cols if c in set(sum(groups.values(), []))]
    if not all_feats:
        raise SystemExit("✗ No usable features after engineering. Check your columns / feature groups.")

    
    CAT_COLS = ["Browser","OS","Age","Education","EngProf","Major","ResolutionLen","SurveyAnswer01","IsExp"]
    NUM_COLS = [c for c in all_feats if c not in CAT_COLS]

    pre_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    def make_ohe_local():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe_local())
    ])

    def make_pipelines(use_feats):
        use_num = [c for c in use_feats if c in NUM_COLS]
        use_cat = [c for c in use_feats if c in CAT_COLS]
        pre = ColumnTransformer(
            [("num", pre_num, use_num), ("cat", pre_cat, use_cat)],
            remainder="drop"
        )
        pipes = {
            "lr":  Pipeline([("pre", pre),
                            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                                       C=1.0, random_state=args.random_state))]),
            "knn": Pipeline([("pre", pre),
                            ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean"))]),
            "rf":  Pipeline([("pre", pre),
                            ("clf", RandomForestClassifier(
                                n_estimators=400, max_depth=None, min_samples_split=4,
                                n_jobs=-1, random_state=args.random_state))]),
        }
        if HAS_XGB:
            pipes["xgb"] = Pipeline([("pre", pre),
                                    ("clf", XGBClassifier(
                                        n_estimators=500, max_depth=5, learning_rate=0.08,
                                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                        objective="binary:logistic",
                                        random_state=args.random_state,
                                        n_jobs=-1, tree_method="hist"))])
        return pipes

    # ablation setups
    setups = []
    if args.ablation == "none":
        setups = [("all", all_feats)]
    elif args.ablation == "drop_group":
        setups = [("all", all_feats)]
        for g, cols in groups.items():
            setups.append((f"drop_{g}", [c for c in all_feats if c not in set(cols)]))
    elif args.ablation == "only_group":
        for g, cols in groups.items():
            subset = [c for c in cols if c in all_feats]
            if subset:
                setups.append((f"only_{g}", subset))

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    metrics_registry = {}
    ablation_rows = []

    # ------------- main experiments -------------
    for tag, feat_subset in setups:
        model_pipes = make_pipelines(feat_subset)
        for m in list(wanted):
            if m not in model_pipes:
                continue
            pipe = model_pipes[m]
            fold_rows = []
            fold_metrics = {}  

            for holdout, bundle in fold_data.items():
                X_tr_full = bundle["X_tr"]
                X_te_full = bundle["X_te"]
                y_tr = bundle["y_tr"]
                y_te = bundle["y_te"]
                meta = bundle["meta"]

                if X_tr_full.empty or X_te_full.empty:
                    continue

                X_tr = X_tr_full[feat_subset]
                X_te = X_te_full[feat_subset]

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                proba = safe_proba(pipe, X_te)

                # fold-level metrics (per-version)
                fold_metrics[holdout] = compute_metrics(y_te, y_pred, proba)

                fold = meta.copy()
                fold["label"] = np.where(y_pred==1, "Yes", "No")
                fold["gold"]  = np.where(y_te==1, "Yes", "No")
                fold["scenario_id"] = "S3"
                fold["method"] = f"S3_peruser_{m}"
                fold["decision_reason"] = f"groups={tag}+{m}"
                fold["holdout_version"] = holdout
                if proba is not None and len(proba) == len(y_te):
                    fold["proba"] = np.asarray(proba, dtype=float)
                fold_rows.append(fold)

            if not fold_rows:
                continue

            preds = pd.concat(fold_rows, ignore_index=True)
            preds_path = outdir / f"S3_peruser_{m}_{tag}_preds.csv"
            preds.to_csv(preds_path, index=False)

            # ---- overall metrics over folds ----
            # Each version contributes equally.
            metric_keys = list(next(iter(fold_metrics.values())).keys())
            overall = {}
            for key in metric_keys:
                vals = [fm[key] for fm in fold_metrics.values()]
                if key in ["TP","FP","TN","FN"]:
                    # sum confusion components over folds
                    overall[key] = int(np.sum(vals))
                else:
                    # macro-average other metrics
                    overall[key] = float(np.mean(vals))

            
            per_version = {str(v): fm for v, fm in fold_metrics.items()}

            metrics_registry.setdefault(m, {})
            metrics_registry[m][tag] = {
                "overall": overall,
                "per_version": per_version,
                "preds_csv": str(preds_path),
                "features_used": feat_subset
            }

            ablation_rows.append({
                "setup": tag, "model": m,
                "n_features": len(feat_subset),
                "features": ",".join(feat_subset),
                **overall,
                "preds_csv": str(preds_path)
            })

    # ---------- write artifacts ----------
    rows_df = pd.DataFrame(ablation_rows)
    rows_df.to_csv(outdir / "S3_ablation_table.csv", index=False)

    flat = []
    for model, setups_dict in metrics_registry.items():
        for tag, data in setups_dict.items():
            base = {
                "model": model, "setup": tag,
                "preds_csv": data.get("preds_csv",""),
                "features_used": ",".join(data.get("features_used", [])),
                "n_features": len(data.get("features_used", [])),
            }
            for k,v in (data.get("overall",{}) or {}).items():
                base[f"overall_{k}"] = v
            for vval, vb in (data.get("per_version",{}) or {}).items():
                for k,v in (vb or {}).items():
                    base[f"ver{vval}_{k}"] = v
            flat.append(base)
    metrics_df = pd.DataFrame(flat)
    metrics_df.to_csv(outdir / "S3_metrics.csv", index=False)

    # best model by macro-averaged accuracy
    if not rows_df.empty:
        best_idx = rows_df["accuracy"].astype(float).idxmax()
        best_row = rows_df.loc[best_idx].to_dict()
        _m, _tag = best_row["model"], best_row["setup"]

        
        best_wide = metrics_df[(metrics_df["model"]==_m) & (metrics_df["setup"]==_tag)].copy()
        if best_wide.empty:
            best_wide = pd.DataFrame([best_row])
        best_wide.insert(0,"winner_model",_m)
        best_wide.insert(1,"winner_setup",_tag)
        best_wide.insert(2,"winner_features", best_row.get("features",""))
        best_wide.insert(3,"winner_n_features", int(best_row.get("n_features",0)))
        best_wide.to_csv(outdir / "S3_best_report.csv", index=False)

        # overall + per-version
        winner = metrics_registry.get(_m, {}).get(_tag, {})
        feats = winner.get("features_used", [])
        preds_csv = winner.get("preds_csv","")
        long_rows = []

        if winner.get("overall", {}):
            long_rows.append({
                "model": _m, "setup": _tag, "version": "overall",
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **winner["overall"]
            })
        for vval, vb in (winner.get("per_version", {}) or {}).items():
            long_rows.append({
                "model": _m, "setup": _tag, "version": str(vval),
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **(vb or {})
            })

        pd.DataFrame(long_rows).to_csv(outdir / "S3_best_report_long.csv", index=False)

    print(json.dumps({
        "versions": sorted(df[args.version].dropna().unique().tolist()),
        "user_id_col": args.user,
        "models": wanted,
        "ablation_mode": args.ablation,
        "metrics_csv": str(outdir / "S3_metrics.csv"),
        "ablation_table": str(outdir / "S3_ablation_table.csv"),
        "best_report_csv": str(outdir / "S3_best_report.csv"),
        "best_report_long_csv": str(outdir / "S3_best_report_long.csv")
    }, indent=2))

if __name__ == "__main__":
    main()
