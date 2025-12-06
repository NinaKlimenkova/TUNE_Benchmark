#!/usr/bin/env python3
# S6 — TUS (question) features + aggregated crowd features (one row per question)
# Per-version metrics + Best model report + Top-10 users (overall & per-version)
# Group ablations + Individual feature ablations (LOFO + singletons)
# Outputs:
#   S6_<model>_<setup>_preds.csv
#   S6_ablation_table.csv
#   S6_metrics.csv
#   S6_metrics_wide_filled.csv
#   S6_metrics_long.csv
#   S6_best_report.csv
#   S6_best_report_long.csv

import argparse, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# ---------- helpers ----------
def to01(s: pd.Series) -> pd.Series:
    m = {"yes":1,"y":1,"1":1,"true":1,"t":1,"unionable":1,
         "no":0,"n":0,"0":0,"false":0,"f":0,"non-unionable":0,
         "nonunionable":0,"non_unionable":0}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().mean() > 0.5:
        out = pd.to_numeric(s, errors="coerce")
    return out

def mode_label(vals: pd.Series) -> int:
    cnt = Counter(vals)
    return int(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])

def entropy_from_prop(p_yes: float) -> float:
    import math
    p_no = 1.0 - p_yes
    e = 0.0
    if p_yes > 0: e -= p_yes * math.log(p_yes)
    if p_no  > 0: e -= p_no  * math.log(p_no)
    return e

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

ALL_METRIC_KEYS = [
    "accuracy","balanced_accuracy",
    "precision_macro","recall_macro","f1_macro",
    "precision_micro","recall_micro","f1_micro",
    "precision_weighted","recall_weighted","f1_weighted",
    "mcc","roc_auc",
    "TP","FP","TN","FN","precision_pos_yes","recall_pos_yes","f1_pos_yes",
    "precision_pos_no","recall_pos_no","f1_pos_no","specificity_tnr"
]

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
    if y_proba is not None and len(y_proba)==len(y_true):
        try: out["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)),4)
        except Exception: out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    out.update(per_class_metrics(y_true, y_pred))
    return out

def empty_metrics_dict():
    return {k: None for k in ALL_METRIC_KEYS}

def safe_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:,1]
    except Exception:
        try:
            from scipy.special import expit
            return expit(pipe.decision_function(X))
        except Exception:
            return None

def build_groups(columns_present, tus_cols_present):
    groups = {
        "COUNTS":  [c for c in ["n_total","n_yes","n_no","prop_yes","prop_no"] if c in columns_present],
        "ENTROPY": [c for c in ["vote_entropy"] if c in columns_present],
        "CONF":    [c for c in ["conf_mean","conf_std","mean_conf_yes","mean_conf_no"] if c in columns_present],
        "TIME":    [c for c in ["time_mean","time_std"] if c in columns_present],
        "CLICKS":  [c for c in ["clicks_mean","clicks_std"] if c in columns_present],
        "TUS":     [c for c in tus_cols_present if c in columns_present],
    }
    return {k:v for k,v in groups.items() if len(v)>0}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="S6: TUS + aggregated crowd, with group + per-feature ablations, Top-10, best report")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sep", default=None, help="CSV delimiter (e.g., ';'); if omitted, auto-detect")
    ap.add_argument("--qid", default="QuestionNum")
    ap.add_argument("--version", default="SurveyVersion")
    ap.add_argument("--gold", default="ActualAnswer")
    ap.add_argument("--vote_col", default="SurveyAnswer")
    ap.add_argument("--conf_col", default="ConfidenceLevel")
    ap.add_argument("--time_col", default="DecisionTime")
    ap.add_argument("--clicks_col", default="ClickCount")
    ap.add_argument("--user", default="ByWho", help="User id column for Top-10 users (observational)")

    ap.add_argument("--tus_features", default="Santos,Starnie,D3L", help="Comma list of TUS columns to include")
    ap.add_argument("--models", default="lr,knn,rf,xgb")
    ap.add_argument("--ablation", default="none",
                    choices=["none","lofo","singletons","groups","both","all"],
                    help="groups: block ablations (incl. TUS); lofo/singletons: individual features; both/all: groups + per-feature")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # load
    df = pd.read_csv(args.csv, sep=(None if args.sep is None else args.sep),
                     engine=("python" if args.sep is None else None))

    # sanity
    for c in [args.qid, args.version, args.gold, args.vote_col]:
        if c not in df.columns:
            raise SystemExit(f"✗ Missing column '{c}'. Found: {list(df.columns)}")

    # targets
    gold01 = to01(df[args.gold])
    keep = ~gold01.isna()
    df = df.loc[keep].copy()
    gold01 = gold01.loc[keep]

    # per-user fields 
    vote01 = to01(df[args.vote_col])
    conf   = pd.to_numeric(df[args.conf_col], errors="coerce") if args.conf_col in df.columns else None
    time   = pd.to_numeric(df[args.time_col], errors="coerce") if args.time_col in df.columns else None
    clicks = pd.to_numeric(df[args.clicks_col], errors="coerce") if args.clicks_col in df.columns else None

    # TUS columns
    tus_cols = [c.strip() for c in args.tus_features.split(",") if c.strip()]
    tus_cols = [c for c in tus_cols if c in df.columns]

    # Aggregate crowd to question-level
    keys = [args.version, args.qid]
    g = pd.DataFrame({args.version: df[args.version], args.qid: df[args.qid]})
    g["vote01"] = vote01
    if conf is not None: g["conf"] = conf
    if time is not None: g["time"] = time
    if clicks is not None: g["clicks"] = clicks

    agg = g.groupby(keys).agg(n_total=("vote01","size"), n_yes=("vote01","sum"))
    agg["n_no"] = agg["n_total"] - agg["n_yes"]
    agg["prop_yes"] = agg["n_yes"] / agg["n_total"].clip(lower=1)
    agg["prop_no"]  = 1.0 - agg["prop_yes"]
    agg["vote_entropy"] = agg["prop_yes"].apply(entropy_from_prop)

    aux = g.set_index(keys)
    if conf is not None:
        agg["conf_mean"] = aux.groupby(keys)["conf"].mean()
        agg["conf_std"]  = aux.groupby(keys)["conf"].std()
        cm_yes = aux.groupby(keys).apply(lambda fr: fr[fr["vote01"]==1]["conf"].mean() if (fr["vote01"]==1).any() else np.nan)
        cm_no  = aux.groupby(keys).apply(lambda fr: fr[fr["vote01"]==0]["conf"].mean() if (fr["vote01"]==0).any() else np.nan)
        agg["mean_conf_yes"] = cm_yes.values
        agg["mean_conf_no"]  = cm_no.values
    if time is not None:
        agg["time_mean"] = aux.groupby(keys)["time"].mean()
        agg["time_std"]  = aux.groupby(keys)["time"].std()
    if clicks is not None:
        agg["clicks_mean"] = aux.groupby(keys)["clicks"].mean()
        agg["clicks_std"]  = aux.groupby(keys)["clicks"].std()

    # gold per question 
    gold_q = (pd.DataFrame({args.version: df[args.version], args.qid: df[args.qid], "gold01": gold01})
              .groupby(keys)["gold01"].agg(mode_label).reset_index())

    # one TUS row per question/version
    tus_first = df[[args.version, args.qid] + tus_cols].groupby(keys, as_index=False).first() if tus_cols else pd.DataFrame(columns=keys)

    # merge crowd + TUS + gold
    qdf = (agg.reset_index()
              .merge(tus_first, on=keys, how="left")
              .merge(gold_q, on=keys, how="inner")
              .rename(columns={args.qid:"QuestionNum", args.version:"SurveyVersion", "gold01":"y"}))

    y_all = qdf["y"].astype(int).values

    # feature blocks
    def ex(cols): return [c for c in cols if c in qdf.columns]
    COUNTS  = ex(["n_total","n_yes","n_no","prop_yes","prop_no"])
    ENTROPY = ex(["vote_entropy"])
    CONF    = ex(["conf_mean","conf_std","mean_conf_yes","mean_conf_no"])
    TIME    = ex(["time_mean","time_std"])
    CLICKS  = ex(["clicks_mean","clicks_std"])
    TUS     = ex(tus_cols)

    BLOCKS = {"COUNTS":COUNTS, "ENTROPY":ENTROPY, "CONF":CONF, "TIME":TIME, "CLICKS":CLICKS}
    if TUS: BLOCKS["TUS"] = TUS

    ALL_FEATS = sorted({c for cols in BLOCKS.values() for c in cols})
    if not ALL_FEATS:
        raise SystemExit("✗ No features constructed. Check your inputs.")

    GROUPS = build_groups(ALL_FEATS, tus_cols)

    # models
    pre_std = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    models = {
        "lr":  Pipeline([("pre", pre_std), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=args.random_state))]),
        "knn": Pipeline([("pre", pre_std), ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean"))]),
        "rf":  Pipeline([("pre", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=4, n_jobs=-1, random_state=args.random_state))]),
    }
    if HAS_XGB:
        models["xgb"] = Pipeline([("pre", SimpleImputer(strategy="median")),
                                  ("clf", XGBClassifier(
                                      n_estimators=500, max_depth=5, learning_rate=0.08,
                                      subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                      objective="binary:logistic", random_state=args.random_state,
                                      n_jobs=-1, tree_method="hist"))])

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted = [m for m in wanted if m in models]

    # ablation setups
    def add_per_feature(setups):
        setups.append(("all", ALL_FEATS))
        for f in ALL_FEATS:
            setups.append((f"drop_{f}", [c for c in ALL_FEATS if c != f]))
        for f in ALL_FEATS:
            setups.append((f"only_{f}", [f]))
        return setups

    def add_groups(setups):
        if GROUPS:
            all_union = sorted({c for cols in GROUPS.values() for c in cols})
            setups.append(("all_groups", all_union))
            # drop one group
            for gname, cols in GROUPS.items():
                kept = sorted({c for gn, cc in GROUPS.items() if gn!=gname for c in cc})
                setups.append((f"drop_group_{gname}", kept if kept else []))
            # only one group
            for gname, cols in GROUPS.items():
                setups.append((f"only_group_{gname}", list(cols)))
        return setups

    setups = []
    if args.ablation == "none":
        setups = [("all", ALL_FEATS)]
    elif args.ablation == "lofo":
        setups = add_per_feature([])
        setups = [s for s in setups if s[0].startswith("all") or s[0].startswith("drop_")]
    elif args.ablation == "singletons":
        setups = [("only_"+f, [f]) for f in ALL_FEATS]
    elif args.ablation == "groups":
        setups = add_groups([])
    elif args.ablation in ("both","all"):
        setups = add_groups([])
        setups = add_per_feature(setups)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    versions = sorted(qdf["SurveyVersion"].dropna().unique().tolist())

    metrics_registry = {}
    ablation_rows = []
    top10_records = []

    have_user = args.user in df.columns
    ver_col, q_col, uid = args.version, args.qid, args.user

    for tag, feat_cols in setups:
        if len(feat_cols)==0: 
            continue
        X_all = qdf[feat_cols].values
        y_all_arr = qdf["y"].astype(int).values

        for m in wanted:
            pipe = models[m]
            y_true_all, y_pred_all, y_proba_all = [], [], []
            preds_rows = []

            
            user_correct = defaultdict(int)
            user_total   = defaultdict(int)

            for holdout in versions:
                mask_te = (qdf["SurveyVersion"] == holdout)
                X_tr, y_tr = X_all[~mask_te], y_all_arr[~mask_te]
                X_te, y_te = X_all[ mask_te], y_all_arr[ mask_te]
                meta = qdf.loc[mask_te, ["QuestionNum","SurveyVersion"]]

                if X_tr.size == 0 or X_te.size == 0:
                    continue

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                y_true_all.extend(y_te.tolist()); y_pred_all.extend(y_pred.tolist())

                proba = safe_proba(pipe, X_te)
                if proba is not None and len(proba)==len(y_te):
                    y_proba_all.extend(proba.tolist())

                fold = meta.copy()
                fold["label"] = np.where(y_pred==1, "Yes", "No")
                fold["gold"]  = np.where(y_te==1, "Yes", "No")
                fold["scenario_id"] = "S6"
                fold["method"] = f"S6_tus_plus_crowd_{m}"
                fold["decision_reason"] = f"features={tag}+{m}"
                fold["holdout_version"] = holdout
                preds_rows.append(fold)

                if have_user:
                    df_te = df[df[ver_col] == holdout]
                    corr = (to01(df_te[args.vote_col]) == to01(df_te[args.gold])).astype(float)
                    corr = corr.fillna(0.0).astype(int)
                    for u, g_u in df_te.groupby(uid):
                        user_correct[u] += int(corr.loc[g_u.index].sum())
                        user_total[u]   += int(len(g_u))

            if not preds_rows:
                continue

            preds = pd.concat(preds_rows, ignore_index=True)
            preds_path = outdir / f"S6_{m}_{tag}_preds.csv"
            preds.to_csv(preds_path, index=False)

            # overall / per-version metrics on all questions
            yt = (preds["gold"]=="Yes").astype(int).to_numpy()
            yp = (preds["label"]=="Yes").astype(int).to_numpy()
            ypb = None
            overall = compute_metrics(yt, yp, ypb)

            per_version = {}
            for vval, g in preds.groupby("holdout_version"):
                _y_true = (g["gold"]=="Yes").astype(int).to_numpy()
                _y_pred = (g["label"]=="Yes").astype(int).to_numpy()
                per_version[str(vval)] = compute_metrics(_y_true, _y_pred, None)

            
            top10_overall, top10_per_version = {}, {}
            if have_user and len(user_total) > 0:
                usr = pd.DataFrame({
                    "user": list(user_total.keys()),
                    "n": [user_total[u] for u in user_total.keys()],
                    "correct": [user_correct[u] for u in user_total.keys()]
                })
                usr["acc"] = usr["correct"] / usr["n"].clip(lower=1)
                usr = usr.sort_values(["acc","n","user"], ascending=[False,False,True])
                top10 = usr.head(10)

                for _, r in top10.iterrows():
                    top10_records.append({
                        "model": m, "setup": tag,
                        "user": r["user"], "user_accuracy": round(float(r["acc"]),4),
                        "n_rows": int(r["n"]), "preds_csv": str(preds_path)
                    })

                top10_ids = set(top10["user"].tolist())
                touched_top10_per_version = {v:set() for v in versions}
                for v in versions:
                    if not top10_ids:
                        continue
                    df_v = df[(df[ver_col]==v) & (df[uid].isin(top10_ids))]
                    if not df_v.empty:
                        touched_top10_per_version[v] |= set(zip(df_v[ver_col], df_v[q_col]))

                touched_all = set().union(*touched_top10_per_version.values()) if touched_top10_per_version else set()
                top10_subset = preds[preds.apply(lambda r: (r["SurveyVersion"], r["QuestionNum"]) in touched_all, axis=1)]
                if not top10_subset.empty:
                    _yt = (top10_subset["gold"]=="Yes").astype(int).to_numpy()
                    _yp = (top10_subset["label"]=="Yes").astype(int).to_numpy()
                    top10_overall = compute_metrics(_yt, _yp, None)
                else:
                    top10_overall = empty_metrics_dict()

                for v in versions:
                    g = preds[(preds["holdout_version"]==v) &
                              (preds.apply(lambda r: (r["SurveyVersion"], r["QuestionNum"]) in touched_top10_per_version[v], axis=1))]
                    if not g.empty:
                        _y_true = (g["gold"]=="Yes").astype(int).to_numpy()
                        _y_pred = (g["label"]=="Yes").astype(int).to_numpy()
                        top10_per_version[str(v)] = compute_metrics(_y_true, _y_pred, None)
                    else:
                        top10_per_version[str(v)] = empty_metrics_dict()

            
            metrics_registry.setdefault(m, {})
            metrics_registry[m][tag] = {
                "overall": overall,
                "per_version": per_version,
                "top10_overall": top10_overall,
                "top10_per_version": top10_per_version,
                "preds_csv": str(preds_path),
                "features_used": feat_cols
            }
            ablation_rows.append({
                "setup": tag, "model": m, "n_features": len(feat_cols),
                "features": ",".join(feat_cols),
                **overall, "preds_csv": str(preds_path)
            })

    # ---------- write artifacts ----------
    rows_df = pd.DataFrame(ablation_rows)
    outdir = Path(args.outdir)
    rows_df.to_csv(outdir / "S6_ablation_table.csv", index=False)

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
           
            t10o = data.get("top10_overall", {}) or {}
            for k in ALL_METRIC_KEYS:
                base[f"top10_overall_{k}"] = t10o.get(k, None)
            
            t10pv = data.get("top10_per_version", {}) or {}
            for vval in sorted((data.get("per_version") or {}).keys(), key=str):
                pv = t10pv.get(str(vval), {}) or {}
                for k in ALL_METRIC_KEYS:
                    base[f"top10_ver{vval}_{k}"] = pv.get(k, None)
            flat.append(base)

    metrics_df = pd.DataFrame(flat)
    metrics_df.to_csv(outdir / "S6_metrics.csv", index=False)
    metrics_df.fillna("").to_csv(outdir / "S6_metrics_wide_filled.csv", index=False)

    
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
            t10o = data.get("top10_overall", {}) or {}
            for k in ALL_METRIC_KEYS:
                long_rows.append({**base, "scope": "top10_overall", "version": "", "metric": k, "value": t10o.get(k, None)})
            t10pv = data.get("top10_per_version", {}) or {}
            for ver_val in sorted((data.get("per_version") or {}).keys(), key=str):
                bundle = t10pv.get(str(ver_val), {}) or {}
                for k in ALL_METRIC_KEYS:
                    long_rows.append({**base, "scope": "top10_per_version", "version": str(ver_val), "metric": k, "value": bundle.get(k, None)})
    pd.DataFrame(long_rows).to_csv(outdir / "S6_metrics_long.csv", index=False)

    
    pd.DataFrame(top10_records).to_csv(outdir / "S6_top10_users.csv", index=False)

    # best report (winner by overall accuracy)
    if not rows_df.empty and "accuracy" in rows_df:
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
        best_wide.to_csv(outdir / "S6_best_report.csv", index=False)

        
        winner = metrics_registry.get(_m, {}).get(_tag, {})
        feats = winner.get("features_used", [])
        preds_csv = winner.get("preds_csv","")
        best_long_rows = []
        # overall
        if winner.get("overall"):
            best_long_rows.append({
                "model": _m, "setup": _tag, "version": "overall",
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **winner["overall"]
            })
        # per-version
        for vval, vb in (winner.get("per_version", {}) or {}).items():
            best_long_rows.append({
                "model": _m, "setup": _tag, "version": str(vval),
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **(vb or {})
            })
        
        t10o = winner.get("top10_overall", {}) or {}
        row_o = {"model": _m, "setup": _tag, "version": "top10_overall",
                 "features_used": ",".join(feats), "n_features": len(feats), "preds_csv": preds_csv}
        row_o.update({k: t10o.get(k, None) for k in ALL_METRIC_KEYS})
        best_long_rows.append(row_o)
        for vval in sorted((winner.get("per_version") or {}).keys(), key=str):
            vb = (winner.get("top10_per_version", {}) or {}).get(str(vval), {}) or {}
            row = {"model": _m, "setup": _tag, "version": f"top10_ver{vval}",
                   "features_used": ",".join(feats), "n_features": len(feats), "preds_csv": preds_csv}
            row.update({k: vb.get(k, None) for k in ALL_METRIC_KEYS})
            best_long_rows.append(row)

        pd.DataFrame(best_long_rows).to_csv(outdir / "S6_best_report_long.csv", index=False)

    # console
    print(json.dumps({
        "n_questions": int(len(qdf)),
        "versions": versions,
        "models": wanted,
        "ablation_mode": args.ablation,
        "metrics_csv": str(outdir / "S6_metrics.csv"),
        "metrics_long_csv": str(outdir / "S6_metrics_long.csv"),
        "ablation_table": str(outdir / "S6_ablation_table.csv"),
        "best_report_csv": str(outdir / "S6_best_report.csv"),
        "best_report_long_csv": str(outdir / "S6_best_report_long.csv"),
        "top10_users_csv": str(outdir / "S6_top10_users.csv")
    }, indent=2))

if __name__ == "__main__":
    main()
