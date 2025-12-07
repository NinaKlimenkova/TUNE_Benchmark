#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

def to01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes":1, "y":1, "1":1, "true":1, "t":1, "unionable":1,
        "no":0,  "n":0, "0":0, "false":0, "f":0, "non-unionable":0,
        "nonunionable":0, "non_unionable":0
    }
    return s.map(mapping)

def mode_label(vals: pd.Series) -> str:
    from collections import Counter
    cnt = Counter(vals)
    return sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

def build_question_table(df, qid_col, ver_col, gold_col, num_cols, cat_cols):
    cat_cols = [c for c in (cat_cols or []) if c not in (qid_col,)]
    def first_non_null(s: pd.Series):
        for v in s:
            if pd.notna(v) and str(v).strip() != "":
                return v
        return ""
    agg_map = {**{c: "median" for c in num_cols}, gold_col: mode_label}
    for c in cat_cols:
        agg_map[c] = first_non_null
    cols = [qid_col, ver_col, gold_col] + num_cols + cat_cols
    g = (df[cols].groupby([qid_col, ver_col], as_index=False).agg(agg_map))
    y = to01(g[gold_col]); keep = ~y.isna()
    g = g.loc[keep].copy(); g[gold_col] = y.loc[keep].astype(int)
    g = g.rename(columns={ver_col: "SurveyVersion"}).drop(columns=[qid_col])
    return g

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_pipelines(num_cols, cat_cols, random_state=42):
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ]), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    pipes = {
        "lr":  Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=random_state))]),
        "knn": Pipeline([("pre", pre), ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean"))]),
        "rf":  Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, min_samples_split=4,
                                                                     n_jobs=-1, random_state=random_state))]),
    }
    if HAS_XGB:
        pipes["xgb"] = Pipeline([("pre", pre), ("clf", XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", random_state=random_state,
            n_jobs=-1, tree_method="hist"))])
    return pipes

def safe_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        try:
            from scipy.special import expit
            return expit(pipe.decision_function(X))
        except Exception:
            return None

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

def metric_bundle(y_true, y_pred, y_proba):
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

def main():
    ap = argparse.ArgumentParser(description="S2 ML (question-only TUS), no QuestionNum feature/outputs")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--qid", default="QuestionNum")
    ap.add_argument("--version", default="SurveyVersion")
    ap.add_argument("--gold", default="ActualAnswer")
    ap.add_argument("--num_features", required=True)
    ap.add_argument("--cat_features", default="")
    ap.add_argument("--models", default="lr,knn,rf,xgb")
    ap.add_argument("--ablation", default="none", choices=["none","lofo","singletons"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # load data
    df = pd.read_csv(args.csv, sep=(None if args.sep is None else args.sep), engine=("python" if args.sep is None else None))
    num_cols = [c.strip() for c in args.num_features.split(",") if c.strip()]
    cat_cols = [c.strip() for c in args.cat_features.split(",") if c.strip()]
    cat_cols = [c for c in cat_cols if c != args.qid]

    need = [args.qid, args.version, args.gold] + num_cols
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"✗ Missing column '{c}'. Found: {list(df.columns)}")
    if args.qid in (c.strip() for c in (args.cat_features or "").split(",")):
        print(f"! Ignoring {args.qid} in cat_features by design (not allowed).")

    qdf = build_question_table(df, args.qid, args.version, args.gold, num_cols, cat_cols)
    versions = sorted(qdf["SurveyVersion"].dropna().unique().tolist())
    if not versions:
        raise SystemExit("✗ No SurveyVersion values found.")
    y_all = qdf[args.gold].values

    all_pipes = make_pipelines(num_cols, cat_cols, random_state=args.random_state)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    model_names = [m for m in wanted if (m in all_pipes and (m!="xgb" or HAS_XGB))]
    if "xgb" in wanted and not HAS_XGB:
        print("! Skipping xgb (xgboost not installed)")

    full_feats = num_cols + cat_cols
    ablation_sets = []
    if args.ablation == "none":
        ablation_sets = [("all", full_feats)]
    elif args.ablation == "lofo":
        ablation_sets = [("all", full_feats)]
        for f in full_feats:
            ablation_sets.append((f"drop_{f}", [c for c in full_feats if c != f]))
    elif args.ablation == "singletons":
        for f in full_feats:
            ablation_sets.append((f"only_{f}", [f]))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    metrics_json = {}
    rows = []

    for tag, feat_subset in ablation_sets:
        subset_num = [c for c in feat_subset if c in num_cols]
        subset_cat = [c for c in feat_subset if c in cat_cols]
        pipes = make_pipelines(subset_num, subset_cat, random_state=args.random_state)

        for m in model_names:
            pipe = pipes[m]
            y_true_all, y_pred_all, y_proba_all = [], [], []
            preds_all = []

            for holdout in versions:
                mask_te = (qdf["SurveyVersion"] == holdout)
                X_tr = qdf.loc[~mask_te, subset_num + subset_cat]
                y_tr = y_all[~mask_te]
                X_te = qdf.loc[ mask_te, subset_num + subset_cat]
                y_te = y_all[ mask_te]
                meta = qdf.loc[mask_te, ["SurveyVersion"]]

                if X_tr.empty or X_te.empty:
                    continue

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                y_true_all.extend(y_te.tolist())
                y_pred_all.extend(y_pred.tolist())

                proba = safe_proba(pipe, X_te)
                if proba is not None and len(proba)==len(y_te):
                    y_proba_all.extend(proba.tolist())

                fold = meta.copy()
                fold["label"] = np.where(y_pred==1, "Yes", "No")
                fold["gold"]  = np.where(y_te==1, "Yes", "No")
                fold["scenario_id"] = "S2"
                fold["method"] = f"S2_ml_qonly_{m}"
                fold["decision_reason"] = f"cols={','.join(feat_subset)} + {m}"
                fold["holdout_version"] = holdout
                if proba is not None and len(proba) == len(y_te):
                    fold["proba"] = np.asarray(proba, dtype=float)
                preds_all.append(fold)

            if not preds_all:
                continue

            preds = pd.concat(preds_all, ignore_index=True)
            preds_path = outdir / f"S2_{m}_{tag}_preds.csv"
            preds.to_csv(preds_path, index=False)

            y_true = np.array(y_true_all)
            y_pred = np.array(y_pred_all)
            y_proba = np.array(y_proba_all) if len(y_proba_all)==len(y_true_all) else None

            bundle = metric_bundle(y_true, y_pred, y_proba)
            per_version = {}
            for ver, g in preds.groupby("holdout_version"):
                _y_true  = (g["gold"]  == "Yes").astype(int).to_numpy()
                _y_pred  = (g["label"] == "Yes").astype(int).to_numpy()
                _y_proba = None
                if "proba" in g and g["proba"].notna().sum() == len(g):
                    _y_proba = g["proba"].to_numpy(dtype=float)
                per_version[str(ver)] = metric_bundle(_y_true, _y_pred, _y_proba)

            metrics_json.setdefault(m, {})
            metrics_json[m][tag] = {
                "overall": bundle,
                "per_version": per_version,
                "preds_csv": str(preds_path),
                "features_used": feat_subset
            }

            rows.append({
                "setup": tag, "model": m,
                "n_features": len(feat_subset),
                "features": ",".join(feat_subset),
                **bundle,
                "preds_csv": str(preds_path)
            })

    # 1) ablation summary CSV
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(outdir / "S2_ablation_table.csv", index=False)

    # 2) flattened metrics CSV (overall + per-version)
    flat_rows = []
    for model, setups in metrics_json.items():
        for tag, data in setups.items():
            base = {
                "model": model,
                "setup": tag,
                "preds_csv": data.get("preds_csv", ""),
                "features_used": ",".join(data.get("features_used", [])),
                "n_features": len(data.get("features_used", [])),
            }
            for k, v in (data.get("overall", {}) or {}).items():
                base[f"overall_{k}"] = v
            for ver, vb in (data.get("per_version", {}) or {}).items():
                for k, v in vb.items():
                    base[f"ver{ver}_{k}"] = v
            flat_rows.append(base)
    metrics_df = pd.DataFrame(flat_rows)
    metrics_df.to_csv(outdir / "S2_metrics.csv", index=False)

    # 3) best-by-accuracy CSV 
        
    if not rows_df.empty:
        # winner by overall accuracy
        best_idx = rows_df["accuracy"].astype(float).idxmax()
        best_row = rows_df.loc[best_idx].to_dict()
        _m, _tag = best_row["model"], best_row["setup"]

        # (single row with overall + per-version columns)
        best_wide = metrics_df[(metrics_df["model"] == _m) & (metrics_df["setup"] == _tag)]
        if best_wide.empty:
           
            pd.DataFrame([best_row]).to_csv(outdir / "S2_best_report.csv", index=False)
        else:
            best_wide = best_wide.copy()
            best_wide.insert(0, "winner_model", _m)
            best_wide.insert(1, "winner_setup", _tag)
            best_wide.insert(2, "winner_features", best_row["features"])
            best_wide.insert(3, "winner_n_features", int(best_row["n_features"]))
            best_wide.to_csv(outdir / "S2_best_report.csv", index=False)

        # one row per version for the same winner, plus an 'overall' row)
        # Pull the nested metrics from metrics_json to avoid parsing wide columns
        winner_data = metrics_json.get(_m, {}).get(_tag, {})
        winner_feats = winner_data.get("features_used", [])
        winner_preds_csv = winner_data.get("preds_csv", "")
        long_rows = []

        # overall 
        overall = winner_data.get("overall", {}) or {}
        long_rows.append({
            "model": _m,
            "setup": _tag,
            "version": "overall",
            "features_used": ",".join(winner_feats),
            "n_features": len(winner_feats),
            "preds_csv": winner_preds_csv,
            **overall
        })

        # per-version rows
        for ver, vb in (winner_data.get("per_version", {}) or {}).items():
            long_rows.append({
                "model": _m,
                "setup": _tag,
                "version": str(ver),
                "features_used": ",".join(winner_feats),
                "n_features": len(winner_feats),
                "preds_csv": winner_preds_csv,
                **(vb or {})
            })

        pd.DataFrame(long_rows).to_csv(outdir / "S2_best_report_long.csv", index=False)


    # console summary
    print(json.dumps({
        "versions": sorted(qdf['SurveyVersion'].unique().tolist()),
        "models": model_names,
        "ablation_mode": args.ablation,
        "metrics_csv": str(outdir / "S2_metrics.csv"),
        "ablation_table": str(outdir / "S2_ablation_table.csv"),
        "best_report_csv": str(outdir / "S2_best_report.csv")
    }, indent=2))

if __name__ == "__main__":
    main()
