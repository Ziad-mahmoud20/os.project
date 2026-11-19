#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
os_detection.py - post-processing of the exported OS files.

 * finds which sub-system (windows10 / linux_disk / linux_memory / linux_process) every window comes from
 * checks the chunk -> sub-system mapping (config model order + pipeline offset rule)
 * reports per-sub-system performance (incl. a recall review)
 * trains the 8 baseline models (LR, LDA, kNN, CART, RF, NB, SVM, LSTM) per-device / binary / multi-class and compares them with the pipeline
 * exports os_events.csv / os_predictions.csv / os_config.json in the SAME format as the spec
   (original columns / keys, source = "os") and the sub-system details in a separate os_subsystems.csv

Usage : python os_detection.py [data_dir]     (data_dir holds os_events.csv, os_predictions.csv, os_config.json)
Output: ./export/  (3 spec files + os_subsystems.csv + baseline_results.csv + 3 png plots)
Needs : pandas numpy scikit-learn matplotlib  (+ torch أو tensorflow للـ LSTM — اختياري)
"""
import sys
import matplotlib
matplotlib.use("Agg")
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
warnings.filterwarnings("ignore")

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR  = "export"     # هنا هتتحفظ الملفات
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH = pd.Timestamp("2019-01-01")
WINDOW_SECONDS = 10
BLOCK = 100

# الـ spec فيه قائمة أسماء ثابتة للـ source، فالملفات التلاتة بتتصدّر بـ "os".
# لو الـ leader وافق على أسماء الأنظمة الفرعية (windows10 / linux_disk / ...) خلّيها None.
EXPORT_SOURCE = "os"

# الأعمدة المطلوبة في الـ spec بالترتيب (مفيش أعمدة زيادة)
EVENTS_COLS = ["event_id", "source", "window_start", "prediction", "attack_prob", "attack_type",
               "anomaly_score", "n_records", "true_label", "split"]
PRED_COLS = ["source", "chunk", "win", "split", "p_attack", "y_true", "true_type", "pred_type", "n_records"]

# ترتيب الـ chunks على الـ timeline هو نفس ترتيب الـ pipeline (PARTS)
SUBSYSTEMS = {
    0: {"prefix": "win",  "source": "windows10",     "os_family": "Windows", "os_version": "Windows 10", "component": "system"},
    1: {"prefix": "disk", "source": "linux_disk",    "os_family": "Linux",   "os_version": "Linux",      "component": "disk"},
    2: {"prefix": "mem",  "source": "linux_memory",  "os_family": "Linux",   "os_version": "Linux",      "component": "memory"},
    3: {"prefix": "proc", "source": "linux_process", "os_family": "Linux",   "os_version": "Linux",      "component": "process"},
}

events = pd.read_csv(os.path.join(DATA_DIR, "os_events.csv"), parse_dates=["window_start"])
preds  = pd.read_csv(os.path.join(DATA_DIR, "os_predictions.csv"))
with open(os.path.join(DATA_DIR, "os_config.json")) as f:
    cfg = json.load(f)

# الملفات الداخلة لازم تكون على شكل الـ spec أصلًا
assert list(events.columns) == EVENTS_COLS, list(events.columns)
assert list(preds.columns) == PRED_COLS, list(preds.columns)
CFG_KEYS = list(cfg.keys())

THRESHOLD = cfg["threshold"]
print("events:", events.shape, "| predictions:", preds.shape, "| threshold:", THRESHOLD)
print("source (في الملفات الأصلية):", events["source"].unique(), preds["source"].unique())

# رقم الـ window على الـ timeline الموحد
events["win"] = ((events["window_start"] - EPOCH).dt.total_seconds() // WINDOW_SECONDS).astype(int)
events = events.sort_values("win").reset_index(drop=True)

# chunk جديد عند أي فجوة > BLOCK
events["chunk"] = (events["win"].diff().fillna(0) > BLOCK).cumsum().astype(int)
assert events["chunk"].nunique() == len(SUBSYSTEMS), events["chunk"].unique()

# تأكيد التطابق مع chunk الموجود في predictions
chk = events.merge(preds[["win", "chunk"]], on="win", suffixes=("", "_pred"))
assert (chk["chunk"] == chk["chunk_pred"]).all()
print(f"✔ التقسيم مطابق لـ predictions على {len(chk)} window")

print(events.groupby("chunk")["win"].agg(["min", "max", "count"]))

def label_subsystems(df):
    out = df.copy()
    for col in ("prefix", "source", "os_family", "os_version", "component"):
        out[col] = out["chunk"].map({k: v[col] for k, v in SUBSYSTEMS.items()})
    return out

# ملاحظة: عمود source هنا اسم النظام الفرعي (داخلي للتحليل بس) — الـ export بيرجّعه لـ EXPORT_SOURCE
events = label_subsystems(events)
preds  = label_subsystems(preds)

print(events.groupby(["source", "os_family", "component"]).size())

expected_order = [SUBSYSTEMS[k]["prefix"] for k in sorted(SUBSYSTEMS)]

# check 1: ترتيب الأجزاء في الـ config
for key in ("binary_model", "multiclass_model"):
    order = [part.split(":")[0] for part in cfg[key].split("+")]
    assert order == expected_order, f"{key}: {order} != {expected_order}"
print("✔ check 1: ترتيب الأجزاء في الـ config =", expected_order)

# check 2: قاعدة الـ offset (كل جزء بيبدأ على block جديد)
span = events.groupby("chunk")["win"].agg(["min", "max"])
for k in range(1, len(SUBSYSTEMS)):
    expected_start = (int(span.loc[k - 1, "max"]) // BLOCK + 2) * BLOCK
    assert int(span.loc[k, "min"]) == expected_start, (k, int(span.loc[k, "min"]), expected_start)
print("✔ check 2: بداية كل chunk مطابقة لقاعدة الـ offset في الـ pipeline")

pd.DataFrame({"chunk": list(SUBSYSTEMS), "prefix": expected_order,
              "system": [SUBSYSTEMS[k]["source"] for k in SUBSYSTEMS],
              "first_win": span["min"].values, "last_win": span["max"].values}).set_index("chunk")

summary = events.groupby("source").agg(
    windows=("win", "size"),
    attack_pct=("true_label", lambda s: round(100 * s.mean(), 1)),
    train=("split", lambda s: (s == "train").sum()),
    val=("split", lambda s: (s == "val").sum()),
    test=("split", lambda s: (s == "test").sum()),
    purged=("split", lambda s: (s == "purged").sum()),
).reindex([v["source"] for v in SUBSYSTEMS.values()])
print(summary)

def f1s(y, p): return float(f1_score(y, p, zero_division=0))
def auc_safe(y, s): return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

rows = []
for name, d in [("ALL (OS)", events)] + [(s, g) for s, g in events.groupby("source")]:
    te = d[d["split"] == "test"]
    va = d[d["split"] == "val"]
    pred = te["attack_prob"] >= THRESHOLD
    rows.append({
        "source": name, "test_n": len(te),
        "val_F1": f1s(va["true_label"], va["attack_prob"] >= THRESHOLD),
        "test_F1": f1s(te["true_label"], pred),
        "precision": precision_score(te["true_label"], pred, zero_division=0),
        "recall": recall_score(te["true_label"], pred, zero_division=0),
        "AUC (attack_prob)": auc_safe(te["true_label"], te["attack_prob"]),
        "AUC (anomaly)": auc_safe(te["true_label"], te["anomaly_score"]),
    })
perf = pd.DataFrame(rows).set_index("source").round(3)
print(f"config: val_F1={cfg['val_f1']:.3f} | test_F1={cfg['test_f1']:.3f}")
print(perf)

RECALL_MIN = 0.80

def best_threshold(y, proba, lo=0.01, hi=0.99, n=99):
    ths = np.linspace(lo, hi, n)
    return float(ths[np.argmax([f1s(y, proba >= t) for t in ths])])

rows = []
for src, d in events.groupby("source"):
    va, te = d[d["split"] == "val"], d[d["split"] == "test"]
    th_own = best_threshold(va["true_label"].values, va["attack_prob"].values) if va["true_label"].nunique() > 1 else THRESHOLD
    row = {"source": src}
    for tag, th in (("global", THRESHOLD), ("own", th_own)):
        p = te["attack_prob"] >= th
        row[f"th_{tag}"] = th
        row[f"recall_{tag}"] = recall_score(te["true_label"], p, zero_division=0)
        row[f"precision_{tag}"] = precision_score(te["true_label"], p, zero_division=0)
        row[f"F1_{tag}"] = f1s(te["true_label"], p)
    rows.append(row)

diag = pd.DataFrame(rows).set_index("source").reindex([v["source"] for v in SUBSYSTEMS.values()]).round(3)
diag["weak_recall"] = diag["recall_global"] < RECALL_MIN
print("أنظمة الـ recall فيها أقل من", RECALL_MIN, "->", list(diag.index[diag["weak_recall"]]))
print(diag)

pt = preds[preds["split"] != "purged"]
types = pd.crosstab(pt["source"], pt["true_type"]).reindex([v["source"] for v in SUBSYSTEMS.values()])
print(types)

ax = types.plot(kind="bar", stacked=True, figsize=(10, 4.5), colormap="tab10")
ax.set_title("Window types per sub-system (true_type)")
ax.set_ylabel("windows"); ax.set_xlabel("")
plt.xticks(rotation=0); plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "attack_types_per_system.png"), dpi=120); plt.close()

fig, axes = plt.subplots(len(SUBSYSTEMS), 1, figsize=(11, 8), sharex=False)
for ax, (k, meta) in zip(axes, SUBSYSTEMS.items()):
    d = events[events["chunk"] == k]
    ax.plot(d["win"], d["attack_prob"], lw=.5, label="attack_prob")
    ax.scatter(d.loc[d["true_label"] == 1, "win"], np.full((d["true_label"] == 1).sum(), 1.05), s=2, c="red", label="true attack")
    ax.axhline(THRESHOLD, c="gray", ls="--", lw=.8)
    ax.set_ylabel(meta["source"], rotation=0, ha="right", va="center")
axes[0].legend(loc="upper right", ncol=2, fontsize=8)
axes[-1].set_xlabel("window (10s)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "attack_prob_timeline.png"), dpi=120); plt.close()

# ======================================================================
# 5b) Baseline models (Alsaedi et al.) - 8 موديلات على 3 مستويات:
#     per-device (كل نظام فرعي لوحده) / binary / multi-class
# ======================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 42
SEQ_LEN = 4          # LSTM: بيشوف آخر 4 windows (t-3 .. t) — نفس عمق الـ lags في الـ config

# الـ 8 موديلات زي الجدول (الإعدادات من عمود الملاحظات)
def make_baselines():
    return {
        "LR":   make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "LDA":  make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "kNN":  make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "CART": DecisionTreeClassifier(random_state=SEED),
        "RF":   RandomForestClassifier(n_estimators=10, random_state=SEED, n_jobs=-1),
        "NB":   make_pipeline(StandardScaler(), GaussianNB()),
        "SVM":  make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED)),
        # LSTM بيتعامل معاه لوحده تحت (محتاج sequences)
    }
MODEL_ORDER = ["LR", "LDA", "kNN", "CART", "RF", "NB", "SVM", "LSTM"]

# ---------- الـ features: نفس قايمة الـ stage-2 اللي في os_config.json ----------
# الـ raw features بتاعة TON_IoT مش موجودة في الملفات المصدّرة، فالـ baselines بتتدرّب
# على الـ features اللي تقدر تتبنى من الملفات (نفس أسماء cfg["features"]).
def build_window_features(ev):
    d = ev[["win", "chunk", "attack_prob", "anomaly_score", "n_records"]].copy()
    d = d.rename(columns={"anomaly_score": "sub_anomaly"}).sort_values("win").reset_index(drop=True)
    g = d.groupby("chunk")
    d["prev_n_records"] = g["n_records"].shift(1).fillna(d["n_records"])
    d["gap_prev"] = g["win"].diff().fillna(1)
    for c in ("attack_prob", "sub_anomaly", "n_records"):
        for k in (1, 2, 3):
            d[f"{c}_lag{k}"] = g[c].shift(k).fillna(d[c])
        d[f"{c}_roll3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
        d[f"{c}_chg"] = d[c] - d[f"{c}_lag1"]
    for k, meta in SUBSYSTEMS.items():
        d[f"is_{meta['prefix']}"] = (d["chunk"] == k).astype(int)
    return d

FEATS = build_window_features(events)
base = preds.loc[preds["split"] != "purged", ["win", "chunk", "source", "split", "y_true", "true_type", "pred_type"]]
DATA = base.merge(FEATS.drop(columns="chunk"), on="win", how="inner")
assert len(DATA) == len(base)

FULL_FEATURES  = [f for f in cfg["features"] if f in DATA.columns]
INDEP_FEATURES = [f for f in FULL_FEATURES if not f.startswith("attack_prob")]
FEATURE_SETS = {
    "full (config features)":        FULL_FEATURES,    # فيها attack_prob (مخرجات الـ pipeline كـ input)
    "independent (no attack_prob)":  INDEP_FEATURES,   # من غير attack_prob -> مقارنة عادلة
}
print({k: len(v) for k, v in FEATURE_SETS.items()}, "| rows:", len(DATA))

LABELS = list(cfg["classes"])
CLS2ID = {c: i for i, c in enumerate(LABELS)}

# ---------- LSTM (PyTorch لو موجود، أو TensorFlow/Keras) ----------
def make_seq(df, feats, T=SEQ_LEN):
    base_cols = [c for c in ("attack_prob", "sub_anomaly", "n_records") if c in feats]
    steps = [np.stack([df[c if k == 0 else f"{c}_lag{k}"].values for c in base_cols], axis=1)
             for k in range(T - 1, -1, -1)]
    return np.stack(steps, axis=1).astype("float32")        # (n, T, d)

def lstm_predict_proba(Xtr, ytr, Xte, n_classes, epochs=40, hidden=32, seed=SEED):
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    try:
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        class Net(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.lstm = nn.LSTM(d, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, n_classes)
            def forward(self, x):
                o, _ = self.lstm(x)
                return self.fc(o[:, -1, :])
        net = Net(Xtr.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=1e-2); lossf = nn.CrossEntropyLoss()
        Xt, yt = torch.tensor(Xtr), torch.tensor(np.asarray(ytr), dtype=torch.long)
        for _ in range(epochs):
            net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                idx = perm[i:i + 64]; opt.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(Xte)), 1).numpy()
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([tf.keras.layers.Input(Xtr.shape[1:]), tf.keras.layers.LSTM(hidden),
                                 tf.keras.layers.Dense(n_classes, activation="softmax")])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="sparse_categorical_crossentropy")
        m.fit(Xtr, np.asarray(ytr), epochs=epochs, batch_size=64, verbose=0)
        return m.predict(Xte, verbose=0)
    except ImportError:
        return None

LSTM_OK = True
try:
    import torch
except ImportError:
    try:
        import tensorflow
    except ImportError:
        LSTM_OK = False
if not LSTM_OK:
    print("⚠ مفيش torch ولا tensorflow متسطّب -> LSTM هيتخطّى (pip install torch)")


# ---------- تشغيل التجارب ----------
def score_of(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def run_one(name, tr, te, feats, task):
    """task = 'binary' | 'multiclass' ; بيرجّع dict بالمقاييس أو None لو LSTM مش متاح"""
    ycol = "y_true" if task == "binary" else "true_type"
    ytr = tr[ycol].values if task == "binary" else tr[ycol].map(CLS2ID).values
    yte = te[ycol].values if task == "binary" else te[ycol].map(CLS2ID).values
    if len(np.unique(ytr)) < 2:
        return None
    if name == "LSTM":
        if not LSTM_OK: return None
        k = 2 if task == "binary" else len(LABELS)
        proba = lstm_predict_proba(make_seq(tr, feats), ytr, make_seq(te, feats), k)
        if proba is None: return None
        pred, score = proba.argmax(1), (proba[:, 1] if task == "binary" else None)
    else:
        model = make_baselines()[name].fit(tr[feats].values, ytr)
        pred = model.predict(te[feats].values)
        score = score_of(model, te[feats].values) if task == "binary" else None
    if task == "binary":
        return {"F1": f1s(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                "recall": recall_score(yte, pred, zero_division=0), "AUC": auc_safe(yte, score),
                "accuracy": accuracy_score(yte, pred)}
    return {"F1": float(f1_score(yte, pred, average="macro", zero_division=0)),   # macro-F1
            "accuracy": accuracy_score(yte, pred)}

res = []
for fs_name, feats in FEATURE_SETS.items():
    tr_all, te_all = DATA[DATA["split"] == "train"], DATA[DATA["split"] == "test"]
    for name in MODEL_ORDER:
        for task in ("binary", "multiclass"):                       # مستوى binary + multi-class
            r = run_one(name, tr_all, te_all, feats, task)
            if r: res.append({"level": task, "feature_set": fs_name, "model": name, "device": "ALL (OS)",
                              "test_n": len(te_all), **r})
        for k, meta in SUBSYSTEMS.items():                          # مستوى per-device (binary)
            tr, te = tr_all[tr_all["chunk"] == k], te_all[te_all["chunk"] == k]
            r = run_one(name, tr, te, feats, "binary")
            if r: res.append({"level": "per-device", "feature_set": fs_name, "model": name,
                              "device": meta["source"], "test_n": len(te), **r})
    print("done:", fs_name)

baseline_res = pd.DataFrame(res)

# الـ pipeline بتاعنا للمقارنة (نفس الـ test split ونفس الـ threshold)
ours = []
te_all = DATA[DATA["split"] == "test"]
ours.append({"level": "binary", "feature_set": "pipeline", "model": "OURS (pipeline)", "device": "ALL (OS)",
             "test_n": len(te_all), "F1": perf.loc["ALL (OS)", "test_F1"], "precision": perf.loc["ALL (OS)", "precision"],
             "recall": perf.loc["ALL (OS)", "recall"], "AUC": perf.loc["ALL (OS)", "AUC (attack_prob)"]})
ours.append({"level": "multiclass", "feature_set": "pipeline", "model": "OURS (pipeline)", "device": "ALL (OS)",
             "test_n": len(te_all), "F1": float(f1_score(te_all["true_type"], te_all["pred_type"], average="macro", zero_division=0)),
             "accuracy": accuracy_score(te_all["true_type"], te_all["pred_type"])})
for src in diag.index:
    ours.append({"level": "per-device", "feature_set": "pipeline", "model": "OURS (pipeline)", "device": src,
                 "test_n": int((te_all["source"] == src).sum()), "F1": diag.loc[src, "F1_global"],
                 "precision": diag.loc[src, "precision_global"], "recall": diag.loc[src, "recall_global"]})
baseline_res = pd.concat([baseline_res, pd.DataFrame(ours)], ignore_index=True).round(3)
baseline_res.to_csv(os.path.join(OUT_DIR, "baseline_results.csv"), index=False)
print("saved baseline_results.csv:", baseline_res.shape)


# ---------- الجداول ----------
order = MODEL_ORDER + ["OURS (pipeline)"]
def table(level, fs, metric="F1"):
    d = baseline_res[(baseline_res["level"] == level) & (baseline_res["feature_set"].isin([fs, "pipeline"]))]
    t = d.pivot_table(index="model", columns="device", values=metric).reindex(order)
    return t

fs_main = "independent (no attack_prob)"
print("=== Binary (test) — كل الـ metrics | features:", fs_main)
print(baseline_res[(baseline_res["level"] == "binary") & (baseline_res["feature_set"].isin([fs_main, "pipeline"]))]
      .set_index("model").reindex(order)[["F1", "precision", "recall", "AUC", "accuracy"]])
print("\n=== Multi-class (test) — macro-F1 / accuracy | features:", fs_main)
print(baseline_res[(baseline_res["level"] == "multiclass") & (baseline_res["feature_set"].isin([fs_main, "pipeline"]))]
      .set_index("model").reindex(order)[["F1", "accuracy"]])
print("\n=== Per-device binary F1 (test) | features:", fs_main)
print(table("per-device", fs_main))

fs_full = "full (config features)"
print("\n=== Binary F1 | features:", fs_full, "(فيها attack_prob بتاع الـ pipeline)")
print(table("binary", fs_full))

# ---------- رسم المقارنة ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (lvl, metric, ttl) in zip(axes, [("binary", "F1", "Binary — test F1"),
                                         ("multiclass", "F1", "Multi-class — test macro-F1"),
                                         ("per-device", "F1", "Per-device — test F1 (mean)")]):
    w = 0.38
    for j, fs in enumerate(FEATURE_SETS):
        d = baseline_res[(baseline_res["level"] == lvl) & (baseline_res["feature_set"] == fs)]
        v = d.groupby("model")[metric].mean().reindex(MODEL_ORDER)
        ax.bar(np.arange(len(MODEL_ORDER)) + (j - .5) * w, v.values, w, label=fs)
    o = baseline_res[(baseline_res["level"] == lvl) & (baseline_res["model"] == "OURS (pipeline)")][metric].mean()
    ax.axhline(o, c="red", ls="--", lw=1, label="OURS (pipeline)")
    ax.set_xticks(range(len(MODEL_ORDER))); ax.set_xticklabels(MODEL_ORDER, rotation=45)
    ax.set_title(ttl); ax.set_ylim(0, 1.05)
axes[0].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "baseline_comparison.png"), dpi=120); plt.show(); plt.close()


events_out = events.sort_values("event_id")[EVENTS_COLS].copy()
preds_out  = preds[PRED_COLS].copy()
if EXPORT_SOURCE is not None:
    events_out["source"] = EXPORT_SOURCE
    preds_out["source"]  = EXPORT_SOURCE

# تأكيد إن الشكل هو نفس الـ spec
assert list(events_out.columns) == EVENTS_COLS
assert list(preds_out.columns) == PRED_COLS and preds_out.shape[1] == 9

events_out.to_csv(os.path.join(OUT_DIR, "os_events.csv"), index=False)
preds_out.to_csv(os.path.join(OUT_DIR, "os_predictions.csv"), index=False)

# الـ config من غير أي key زيادة
assert list(cfg.keys()) == CFG_KEYS
with open(os.path.join(OUT_DIR, "os_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# تفاصيل الأنظمة الفرعية في ملف منفصل (برّه الـ 3 ملفات)
events[["win", "source", "os_family", "component"]].drop_duplicates().to_csv(
    os.path.join(OUT_DIR, "os_subsystems.csv"), index=False)

print("saved ->", os.path.abspath(OUT_DIR))
print("source في os_events.csv:", events_out["source"].unique().tolist())
print(events.groupby("source").size())
