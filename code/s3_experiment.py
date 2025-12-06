#!/usr/bin/env python3
# S5 — Crowd aggregation (NO TUS) with per-feature + group ablations


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

def build_groups(columns_present):
    groups = {
        "vote":   [c for c in ["n_total","n_yes","n_no","prop_yes","prop_no","vote_entropy"] if c in columns_present],
        "conf":   [c for c in ["conf_mean","conf_std","mean_conf_yes","mean_conf_no"]        if c in columns_present],
        "time":   [c for c in ["time_mean","time_std"]                                      if c in columns_present],
        "clicks": [c for c in ["clicks_mean","clicks_std"]                                  if c in columns_present],
    }
    
    return {k:v for k,v in groups.items() if len(v)>0}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="S5 crowd aggregation (no TUS) + ablations (per-feature + groups) + CSV + Top-10 metrics")
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
    ap.add_argument("--models", default="lr,knn,rf,xgb")
    ap.add_argument("--ablation", default="none", choices=["none","lofo","singletons","groups","both","all"],
                    help="none: all feats; lofo: drop one feature; singletons: only one feature; groups: only group ablations; both: groups+per-feature; all: alias of both")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # load
    df = pd.read_csv(args.csv, sep=(None if args.sep is None else args.sep),
                     engine=("python" if args.sep is None else None))

    
    needed = [args.qid, args.version, args.gold, args.vote_col]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"✗ Missing column '{c}'. Found: {list(df.columns)}")

    # drop TUS if present
    df = df.drop(columns=[c for c in ["Santos","Starnie","D3L","santos","starnie","d3l"] if c in df.columns],
                 errors="ignore")

    vote01 = to01(df[args.vote_col])
    gold01 = to01(df[args.gold])
    keep = ~gold01.isna()
    df, vote01, gold01 = df.loc[keep].copy(), vote01.loc[keep], gold01.loc[keep]
    df["_gold01"] = gold01
    df["_vote01"] = vote01

    conf   = pd.to_numeric(df[args.conf_col], errors="coerce") if args.conf_col in df.columns else None
    time   = pd.to_numeric(df[args.time_col], errors="coerce") if args.time_col in df.columns else None
    clicks = pd.to_numeric(df[args.clicks_col], errors="coerce") if args.clicks_col in df.columns else None

    keys = [args.version, args.qid]
    g = pd.DataFrame({args.version: df[args.version], args.qid: df[args.qid]})
    g["vote01"] = vote01
    if conf is not None:   g["conf"] = conf
    if time is not None:   g["time"] = time
    if clicks is not None: g["clicks"] = clicks

    # counts & proportions per (version, question)
    agg = g.groupby(keys).agg(n_total=("vote01","size"), n_yes=("vote01","sum"))
    agg["n_no"] = agg["n_total"] - agg["n_yes"]
    agg["prop_yes"] = agg["n_yes"] / agg["n_total"].clip(lower=1)
    agg["prop_no"]  = 1.0 - agg["prop_yes"]
    agg["vote_entropy"] = agg["prop_yes"].apply(entropy_from_prop)

    aux = g.set_index(keys)
    if conf is not None:
        agg["conf_mean"] = aux.groupby(keys)["conf"].mean()
        agg["conf_std"]  = aux.groupby(keys)["conf"].std()
        cm_yes = aux.groupby(keys).apply(
            lambda fr: fr[fr["vote01"]==1]["conf"].mean() if (fr["vote01"]==1).any() else np.nan
        )
        cm_no  = aux.groupby(keys).apply(
            lambda fr: fr[fr["vote01"]==0]["conf"].mean() if (fr["vote01"]==0).any() else np.nan
        )
        agg["mean_conf_yes"] = cm_yes.values
        agg["mean_conf_no"]  = cm_no.values

    if time is not None:
        agg["time_mean"] = aux.groupby(keys)["time"].mean()
        agg["time_std"]  = aux.groupby(keys)["time"].std()

    if clicks is not None:
        agg["clicks_mean"] = aux.groupby(keys)["clicks"].mean()
        agg["clicks_std"]  = aux.groupby(keys)["clicks"].std()

    # gold at question level (mode across annotators)
    gold_q = (pd.DataFrame({args.version: df[args.version], args.qid: df[args.qid], "gold01": gold01})
              .groupby(keys)["gold01"].agg(mode_label).reset_index())

    agg = agg.reset_index().merge(gold_q, on=keys, how="inner")
    agg = agg.rename(columns={args.qid:"QuestionNum", args.version:"SurveyVersion", "gold01":"y"})
    y_all = agg["y"].astype(int).values

    # constructed features
    feat_candidates = [
        "n_total","n_yes","n_no","prop_yes","prop_no","vote_entropy",
        "conf_mean","conf_std","mean_conf_yes","mean_conf_no",
        "time_mean","time_std","clicks_mean","clicks_std"
    ]
    ALL_FEATS = [c for c in feat_candidates if c in agg.columns]
    if not ALL_FEATS:
        raise SystemExit("✗ No crowd features were constructed. Check column names for vote/conf/time/clicks.")

    GROUPS = build_groups(ALL_FEATS)

    # models (all numeric)
    pre_num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    models = {
        "lr":  Pipeline([("pre", pre_num), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=args.random_state))]),
        "knn": Pipeline([("pre", pre_num), ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean"))]),
        "rf":  Pipeline([("pre", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=4, n_jobs=-1, random_state=args.random_state))])
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
    versions = sorted(agg["SurveyVersion"].dropna().unique().tolist())

    metrics_registry = {}
    ablation_rows = []
    top10_records = []

    have_user = args.user in df.columns
    ver_col, q_col, uid = args.version, args.qid, args.user

    for tag, feat_cols in setups:
        if len(feat_cols)==0:
            continue
        X_all = agg[feat_cols].values
        for m in wanted:
            pipe = models[m]

            y_true_all, y_pred_all, y_proba_all = [], [], []
            preds_rows = []

            # Top-10 accumulators from raw rows (overall)
            user_correct = defaultdict(int)
            user_total   = defaultdict(int)

            
            touched_per_version = {v:set() for v in versions}

            for holdout in versions:
                mask_te = (agg["SurveyVersion"] == holdout)
                X_tr, y_tr = X_all[~mask_te], y_all[~mask_te]
                X_te, y_te = X_all[ mask_te], y_all[ mask_te]
                meta = agg.loc[mask_te, ["QuestionNum","SurveyVersion"]]

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
                fold["scenario_id"] = "S5"
                fold["method"] = f"S5_crowdagg_{m}"
                fold["decision_reason"] = f"features={tag}+{m}"
                fold["holdout_version"] = holdout
                preds_rows.append(fold)

                if have_user:
                    df_te = df[df[ver_col] == holdout]
                    # accumulate user correctness on test rows (observational)
                    corr = (df_te["_vote01"] == df_te["_gold01"]).astype(int)
                    for u, g in df_te.groupby(uid):
                        user_correct[u] += int(corr.loc[g.index].sum())
                        user_total[u]   += int(len(g))
                    
                    touched_per_version[holdout] |= set(zip(df_te[ver_col], df_te[q_col]))

            if not preds_rows:
                continue

            preds = pd.concat(preds_rows, ignore_index=True)
            preds_path = outdir / f"S5_{m}_{tag}_preds.csv"
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

            # ---- Top-10 users table + metrics
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

                # overall Top-10 subset across all versions
                touched_all = set()
                for v in versions:
                    touched_all |= touched_top10_per_version[v]
                top10_subset = preds[preds.apply(lambda r: (r["SurveyVersion"], r["QuestionNum"]) in touched_all, axis=1)]
                if not top10_subset.empty:
                    _yt = (top10_subset["gold"]=="Yes").astype(int).to_numpy()
                    _yp = (top10_subset["label"]=="Yes").astype(int).to_numpy()
                    top10_overall = compute_metrics(_yt, _yp, None)
                else:
                    top10_overall = empty_metrics_dict()

                # per-version Top-10: ALWAYS create an entry for every version
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
    rows_df.to_csv(outdir / "S5_ablation_table.csv", index=False)

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
            for vval in sorted(setups[tag].get("per_version", {}).keys()):
                vb = setups[tag]["per_version"][vval] or {}
                for k,v in vb.items():
                    base[f"ver{vval}_{k}"] = v
            
            t10o = data.get("top10_overall", {}) or {}
            for k in ALL_METRIC_KEYS:
                base[f"top10_overall_{k}"] = t10o.get(k, None)
            
            t10pv = data.get("top10_per_version", {}) or {}
            for vval in versions:
                pv = t10pv.get(str(vval), {}) or {}
                for k in ALL_METRIC_KEYS:
                    base[f"top10_ver{vval}_{k}"] = pv.get(k, None)
            flat.append(base)

    metrics_df = pd.DataFrame(flat)
    metrics_df.to_csv(outdir / "S5_metrics.csv", index=False)
    metrics_df.fillna("").to_csv(outdir / "S5_metrics_wide_filled.csv", index=False)

    # metrics 
    long_rows = []
    for model, setups in metrics_registry.items():
        for tag, data in setups.items():
            base = {
                "model": model, "setup": tag,
                "preds_csv": data.get("preds_csv",""),
                "features_used": ",".join(data.get("features_used", [])),
                "n_features": len(data.get("features_used", [])),
            }
            # overall
            for k, v in (data.get("overall", {}) or {}).items():
                long_rows.append({**base, "scope": "overall", "version": "", "metric": k, "value": v})
            # per-version 
            for ver_val, bundle in (data.get("per_version", {}) or {}).items():
                for k, v in (bundle or {}).items():
                    long_rows.append({**base, "scope": "per_version", "version": str(ver_val), "metric": k, "value": v})
            
            t10o = data.get("top10_overall", {}) or {}
            for k in ALL_METRIC_KEYS:
                long_rows.append({**base, "scope": "top10_overall", "version": "", "metric": k, "value": t10o.get(k, None)})
            
            t10pv = data.get("top10_per_version", {}) or {}
            for ver_val in versions:
                bundle = t10pv.get(str(ver_val), {}) or {}
                for k in ALL_METRIC_KEYS:
                    long_rows.append({**base, "scope": "top10_per_version", "version": str(ver_val), "metric": k, "value": bundle.get(k, None)})
    pd.DataFrame(long_rows).to_csv(outdir / "S5_metrics_long.csv", index=False)

    
    pd.DataFrame(top10_records).to_csv(outdir / "S5_top10_users.csv", index=False)

   
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
        best_wide.to_csv(outdir / "S5_best_report.csv", index=False)

        winner = metrics_registry.get(_m, {}).get(_tag, {})
        feats = winner.get("features_used", [])
        preds_csv = winner.get("preds_csv","")
        best_long_rows = []

        # overall & per-version
        for vname, bundle in [("overall", winner.get("overall", {}))]:
            if bundle:
                best_long_rows.append({
                    "model": _m, "setup": _tag, "version": vname,
                    "features_used": ",".join(feats), "n_features": len(feats),
                    "preds_csv": preds_csv, **bundle
                })
        for vval, vb in (winner.get("per_version", {}) or {}).items():
            best_long_rows.append({
                "model": _m, "setup": _tag, "version": str(vval),
                "features_used": ",".join(feats), "n_features": len(feats),
                "preds_csv": preds_csv, **(vb or {})
            })
        
        t10o = winner.get("top10_overall", {}) or {}
        best_long_rows.append({
            "model": _m, "setup": _tag, "version": "top10_overall",
            "features_used": ",".join(feats), "n_features": len(feats),
            "preds_csv": preds_csv, **{k:t10o.get(k, None) for k in ALL_METRIC_KEYS}
        })
        for vval in versions:
            vb = (winner.get("top10_per_version", {}) or {}).get(str(vval), {}) or {}
            row = {"model": _m, "setup": _tag, "version": f"top10_ver{vval}",
                   "features_used": ",".join(feats), "n_features": len(feats),
                   "preds_csv": preds_csv}
            row.update({k: vb.get(k, None) for k in ALL_METRIC_KEYS})
            best_long_rows.append(row)

        pd.DataFrame(best_long_rows).to_csv(outdir / "S5_best_report_long.csv", index=False)

    # console
    print(json.dumps({
        "n_questions": int(len(agg)),
        "versions": versions,
        "models": wanted,
        "ablation_mode": args.ablation,
        "metrics_csv": str(outdir / "S5_metrics.csv"),
        "metrics_long_csv": str(outdir / "S5_metrics_long.csv"),
        "ablation_table": str(outdir / "S5_ablation_table.csv"),
        "best_report_csv": str(outdir / "S5_best_report.csv"),
        "best_report_long_csv": str(outdir / "S5_best_report_long.csv"),
        "top10_users_csv": str(outdir / "S5_top10_users.csv")
    }, indent=2))

if __name__ == "__main__":
    main()
