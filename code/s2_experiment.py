#!/usr/bin/env python3
"""
S4 — ML per-user (NO aggregation) with TUS + individual human input.

- Unit: one user row.
- Split: leave-one-version-out by SurveyVersion.
- Features: TUS (e.g., Santos, Starnie, D3L) + per-user groups (clicks, decision time, confidence, etc.)
- Models: LR, kNN, RF, XGB.
- Ablations:
    none
    drop_group     (drop each group once, including TUS)
    only_group     (use only one group at a time)
    lofo_tus       (leave-one-TUS-feature-out; keep all other groups)
    only_tus       (TUS only; also each TUS alone)

Outputs (all CSV under --outdir):
  S4_peruser_<model>_<setup>_preds.csv              (test rows across LOVO)
  S4_ablation_table.csv                             (overall metrics per model/setup)
  S4_metrics.csv                                    (wide: overall + per-version + top10*)
  S4_metrics_long.csv                               (long/tidy form to avoid blanks)
  S4_best_report.csv                                (wide, 1 row for winner)
  S4_best_report_long.csv                           (long/tidy for winner incl. per-version + top10)
  S4_top10_users.csv                                (user id, accuracy, n for top10 per model/setup)
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

def to01(s: pd.Series) -> pd.Series:
    m = {"yes":1, "y":1, "1":1, "true":1, "t":1, "unionable":1,
         "no":0,  "n":0, "0":0, "false":0, "f":0, "non-unionable":0,
         "nonunionable":0, "non_unionable":0}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().mean() > 0.5:
        out = pd.to_numeric(s, errors="coerce")
    return out

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
    ap = argparse.ArgumentParser(description="S4 per-user ML with TUS + human groups; CSV outputs and Top-10 users")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=None, help="CSV delimiter (e.g., ';'); if omitted, auto-detect")
    ap.add_argument("--qid", default="QuestionNum")
    ap.add_argument("--version", default="SurveyVersion")
    ap.add_argument("--gold", default="ActualAnswer")
    ap.add_argument("--user", required=True, help="User identifier column (e.g., ByWho)")

    ap.add_argument("--tus_features", default="Santos,Starnie,D3L", help="Comma list of TUS (question) columns")
    ap.add_argument("--models", default="lr,knn,rf,xgb")
    ap.add_argument("--ablation", default="none",
                    choices=["none","drop_group","only_group","lofo_tus","only_tus"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # load CSV
    df = pd.read_csv(args.csv, sep=(None if args.sep is None else args.sep), engine=("python" if args.sep is None else None))

    # keys & target
    for c in [args.qid, args.version, args.gold, args.user]:
        if c not in df.columns:
            raise SystemExit(f"✗ Missing column '{c}'. Found: {list(df.columns)}")

    y_series = to01(df[args.gold])
    keep = ~y_series.isna()
    if keep.sum() == 0:
        raise SystemExit("✗ No valid target rows after mapping to 0/1.")
    df = df.loc[keep].copy()
    y_series = y_series.loc[keep].astype(int)

    # ---------------- feature groups ----------------
    tus_cols = [c.strip() for c in args.tus_features.split(",") if c.strip()]
    tus_cols = [c for c in tus_cols if c in df.columns]
    if not tus_cols:
        raise SystemExit("✗ No TUS columns found in CSV; set --tus_features (e.g., 'Santos,Starnie,D3L').")

    CLICK        = [c for c in ["FirstClick","LastClick","IsSingleClick","ClickCount","TimeDiff_FCnLC","LC-FC"] if c in df.columns]
    USER_META    = [c for c in ["Browser","OS","Age","Education","EngProf","Major","ResolutionLen"] if c in df.columns]
    HUMAN_LABEL  = [c for c in ["IsExp","ExplanationsLen","ExplanationsLen_scaled","SurveyAnswer"] if c in df.columns]
    DECISIONTIME = [c for c in ["DecisionTime","OverallSurveyDT","MIDT","ISDT","MEDT","ESDT","DecisionTimeFract","DecisionTimeFract_scaled"] if c in df.columns]
    CONFIDENCE   = [c for c in ["ConfidenceLevel","MICL","ISCL","MECL","ESCL","ConfidenceLevel_Dec"] if c in df.columns]
    QUANTIFIED   = [c for c in ["Majority","NoCY","NoCN","Diff_CAns"] if c in df.columns]

    GROUPS = {
        "TUS": tus_cols,
        "CLICK": CLICK,
        "USER_META": USER_META,
        "HUMAN_LABEL": HUMAN_LABEL,
        "DECISIONTIME": DECISIONTIME,
        "CONFIDENCE": CONFIDENCE,
        "QUANTIFIED": QUANTIFIED
    }
    all_feats = sorted(set(sum(GROUPS.values(), [])))
    ver = args.version
    qid = args.qid
    uid = args.user

    # numeric vs categorical split
    cat_candidates, num_candidates = [], []
    for c in all_feats:
        if df[c].dtype == "O":
            cat_candidates.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10:
            cat_candidates.append(c)
        else:
            num_candidates.append(c)

    pre_num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pre_cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())])

    def make_pipelines(use_feats):
        use_num = [c for c in use_feats if c in num_candidates]
        use_cat = [c for c in use_feats if c in cat_candidates]
        pre = ColumnTransformer([("num", pre_num, use_num), ("cat", pre_cat, use_cat)], remainder="drop")
        pipes = {
            "lr":  Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=args.random_state))]),
            "knn": Pipeline([("pre", pre), ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean"))]),
            "rf":  Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=4, n_jobs=-1, random_state=args.random_state))]),
        }
        if HAS_XGB:
            pipes["xgb"] = Pipeline([("pre", pre), ("clf", XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="binary:logistic", random_state=args.random_state,
                n_jobs=-1, tree_method="hist"))])
        return pipes

    # ablation setups
    setups = []
    if args.ablation == "none":
        setups = [("all", all_feats)]
    elif args.ablation == "drop_group":
        setups = [("all", all_feats)]
        for g, cols in GROUPS.items():
            setups.append((f"drop_{g}", [c for c in all_feats if c not in set(cols)]))
    elif args.ablation == "only_group":
        for g, cols in GROUPS.items():
            setups.append((f"only_{g}", list(cols)))
    elif args.ablation == "lofo_tus":
        non_tus = [c for c in all_feats if c not in set(tus_cols)]
        setups = [("tus_all", all_feats)]
        for f in tus_cols:
            setups.append((f"drop_{f}", non_tus + [c for c in tus_cols if c != f]))
    elif args.ablation == "only_tus":
        setups = [("only_TUS_all", list(tus_cols))]
        for f in tus_cols:
            setups.append((f"only_{f}", [f]))

    # models and versions
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    versions = sorted(df[ver].dropna().unique().tolist())
    if not versions:
        raise SystemExit("✗ No SurveyVersion values found.")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    metrics_registry = {}
    ablation_rows = []
    top10_records = []

    for tag, feat_subset in setups:
        model_pipes = make_pipelines(feat_subset)
        for m in list(wanted):
            if m not in model_pipes: continue
            pipe = model_pipes[m]

            preds_rows = []
            for holdout in versions:
                mask_te = (df[ver] == holdout)
                X_tr = df.loc[~mask_te, feat_subset]
                y_tr = to01(df.loc[~mask_te, args.gold]).astype(int).values
                X_te = df.loc[ mask_te, feat_subset]
                y_te = to01(df.loc[ mask_te, args.gold]).astype(int).values
                meta = df.loc[mask_te, [qid, ver, uid]]

                if X_tr.empty or X_te.empty: 
                    continue

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                proba = safe_proba(pipe, X_te)

                fold = meta.copy()
                fold["label"] = np.where(y_pred==1, "Yes", "No")
                fold["gold"]  = np.where(y_te==1, "Yes", "No")
                fold["scenario_id"] = "S4"
                fold["method"] = f"S4_peruser_{m}"
                fold["decision_reason"] = f"setup={tag}+{m}"
                fold["holdout_version"] = holdout
                if proba is not None and len(proba) == len(y_te):
                    fold["proba"] = np.asarray(proba, dtype=float)
                preds_rows.append(fold)

            if not preds_rows: 
                continue

            preds = pd.concat(preds_rows, ignore_index=True)
            preds_path = outdir / f"S4_peruser_{m}_{tag}_preds.csv"
            preds.to_csv(preds_path, index=False)

            # overall / per-version metrics
            y_true = (preds["gold"]=="Yes").astype(int).to_numpy()
            y_pred = (preds["label"]=="Yes").astype(int).to_numpy()
            y_proba = preds["proba"].to_numpy(dtype=float) if ("proba" in preds.columns and preds["proba"].notna().sum()==len(preds)) else None

            overall = compute_metrics(y_true, y_pred, y_proba)
            per_version = {}
            for vval, g in preds.groupby("holdout_version"):
                _y_true = (g["gold"]=="Yes").astype(int).to_numpy()
                _y_pred = (g["label"]=="Yes").astype(int).to_numpy()
                _y_proba = g["proba"].to_numpy(dtype=float) if ("proba" in g.columns and g["proba"].notna().sum()==len(g)) else None
                per_version[str(vval)] = compute_metrics(_y_true, _y_pred, _y_proba)

            # ---- Top-10 users by test accuracy (across all folds)
            preds["correct"] = (preds["label"] == preds["gold"]).astype(int)
            user_stats = (
                preds.groupby(uid).agg(acc=("correct","mean"), n=("correct","size"))
                     .reset_index().sort_values(["acc","n",uid], ascending=[False,False,True])
            )
            top10_users = user_stats.head(10)
            for _, r in top10_users.iterrows():
                top10_records.append({
                    "model": m, "setup": tag,
                    "user": r[uid], "user_accuracy": round(float(r["acc"]),4),
                    "n_rows": int(r["n"]), "preds_csv": str(preds_path)
                })

            # metrics for Top-10 cohort
            top10_ids = set(top10_users[uid].tolist())
            top10_subset = preds[preds[uid].isin(top10_ids)]
            top10_overall, top10_per_version = {}, {}
            if not top10_subset.empty:
                _yt = (top10_subset["gold"]=="Yes").astype(int).to_numpy()
                _yp = (top10_subset["label"]=="Yes").astype(int).to_numpy()
                _pp = top10_subset["proba"].to_numpy(dtype=float) if ("proba" in top10_subset.columns and top10_subset["proba"].notna().sum()==len(top10_subset)) else None
                top10_overall = compute_metrics(_yt, _yp, _pp)
                for vval, g in top10_subset.groupby("holdout_version"):
                    _y_true = (g["gold"]=="Yes").astype(int).to_numpy()
                    _y_pred = (g["label"]=="Yes").astype(int).to_numpy()
                    _y_proba = g["proba"].to_numpy(dtype=float) if ("proba" in g.columns and g["proba"].notna().sum()==len(g)) else None
                    top10_per_version[str(vval)] = compute_metrics(_y_true, _y_pred, _y_proba)

            # register
            metrics_registry.setdefault(m, {})
            metrics_registry[m][tag] = {
                "overall": overall,
                "per_version": per_version,
                "top10_overall": top10_overall,
                "top10_per_version": top10_per_version,
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
    rows_df.to_csv(outdir / "S4_ablation_table.csv", index=False)

    # wide metrics
    flat = []
    for model, setups in metrics_registry.items():
        for tag, data in setups.items():
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
            for k,v in (data.get("top10_overall",{}) or {}).items():
                base[f"top10_overall_{k}"] = v
            for vval, vb in (data.get("top10_per_version",{}) or {}).items():
                for k,v in (vb or {}).items():
                    base[f"top10_ver{vval}_{k}"] = v
            flat.append(base)
    metrics_df = pd.DataFrame(flat)
    metrics_df.to_csv(outdir / "S4_metrics.csv", index=False)
    metrics_df.fillna("").to_csv(outdir / "S4_metrics_wide_filled.csv", index=False)

    # long/tidy metrics
    long_rows = []
    for model, setups in metrics_registry.items():
        for tag, data in setups.items():
            base = {
                "model": model, "setup": tag,
                "preds_csv": data.get("preds_csv",""),
                "features_used": ",".join(data.get("features_used", [])),
                "n_features": len(data.get("features_used", [])),
            }
            for k, v in (data.get("overall", {}) or {}).items():
                long_rows.append({**base, "scope": "overall", "version": "", "metric": k, "value": v})
            for ver_val, bundle in (data.get("per_version", {}) or {}).items():
                for k, v in (bundle or {}).items():
                    long_rows.append({**base, "scope": "per_version", "version": str(ver_val), "metric": k, "value": v})
            for k, v in (data.get("top10_overall", {}) or {}).items():
                long_rows.append({**base, "scope": "top10_overall", "version": "", "metric": k, "value": v})
            for ver_val, bundle in (data.get("top10_per_version", {}) or {}).items():
                for k, v in (bundle or {}).items():
                    long_rows.append({**base, "scope": "top10_per_version", "version": str(ver_val), "metric": k, "value": v})
    pd.DataFrame(long_rows).to_csv(outdir / "S4_metrics_long.csv", index=False)

    
    pd.DataFrame(top10_records).to_csv(outdir / "S4_top10_users.csv", index=False)

    # best report (winner by overall accuracy)
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
        best_wide.to_csv(outdir / "S4_best_report.csv", index=False)

        
        winner = metrics_registry.get(_m, {}).get(_tag, {})
        feats = winner.get("features_used", [])
        preds_csv = winner.get("preds_csv","")
        best_long_rows = []
        for label, bundle in [
            ("overall", winner.get("overall", {})),
            ("top10_overall", winner.get("top10_overall", {}))
        ]:
            if bundle:
                best_long_rows.append({
                    "model": _m, "setup": _tag, "version": label,
                    "features_used": ",".join(feats), "n_features": len(feats),
                    "preds_csv": preds_csv, **bundle
                })
        for vval, vb in (winner.get("per_version", {}) or {}).items():
            best_long_rows.append({
                "model": _m, "setup": _tag, "version": str(vval),
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **(vb or {})
            })
        for vval, vb in (winner.get("top10_per_version", {}) or {}).items():
            best_long_rows.append({
                "model": _m, "setup": _tag, "version": f"top10_ver{vval}",
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **(vb or {})
            })
        pd.DataFrame(best_long_rows).to_csv(outdir / "S4_best_report_long.csv", index=False)

    # console summary
    print(json.dumps({
        "versions": versions,
        "models": wanted,
        "ablation_mode": args.ablation,
        "metrics_csv": str(outdir / "S4_metrics.csv"),
        "metrics_long_csv": str(outdir / "S4_metrics_long.csv"),
        "ablation_table": str(outdir / "S4_ablation_table.csv"),
        "best_report_csv": str(outdir / "S4_best_report.csv"),
        "best_report_long_csv": str(outdir / "S4_best_report_long.csv"),
        "top10_users_csv": str(outdir / "S4_top10_users.csv")
    }, indent=2))

if __name__ == "__main__":
    main()
