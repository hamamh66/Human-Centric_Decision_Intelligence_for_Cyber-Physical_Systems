"""
Regeneration script (FINAL v2 - memory-safe, progressive prints).
Matches the numbers in the revised manuscript exactly.

FIXES vs previous version:
  * fetch_kddcup99(percent10=True)  <- previous version had False, which loads the
    FULL 4.9M-row dataset and exhausts Colab RAM. The paper uses the 10% subset
    (494,021 rows; label distribution 396,743 attack / 97,278 normal).
  * float32 downcasting + explicit deletion of intermediates.
  * UNSW block now implements the manuscript protocol: seeded STRATIFIED 20%
    calibration split of the official training file (its row order is grouped by
    class, not chronological), official test file evaluated untouched.
  * [progress] prints at every stage so a stall/crash is localizable.
  * per-call seeded RNG: every pipeline invocation is independently reproducible,
    so results do not depend on which experiments ran before it.

Data sources for the reported results:
  KDDCup99 10%: sklearn fetch (percent10=True) or the mirror
    https://github.com/IndexFziQ/ML-ATIC/raw/master/kddcup.data_10_percent.gz
    (md5 of decompressed stream: eb43ac454e61166f88aad7943ef1e079)
  UNSW-NB15: official partition CSVs (training 175,341 / testing 82,332),
    Kaggle mrwellsdavid/unsw-nb15.
"""
import numpy as np, pandas as pd, os, json, gc, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss)

SEED = 42
T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

OUT = "/content/drive/MyDrive/Outputs/HCDI_EAAI"   # adjust if running locally
os.makedirs(OUT, exist_ok=True)
log(f"output dir: {OUT}")

def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def run_pipeline(Z, y, tag, split="order", t1=0.50, t2=0.65, t3=0.80,
                 Zte=None, yte=None, figs=True):
    rng = np.random.default_rng(SEED)   # fresh per-call RNG: results independent of call order
    """
    split="order":  order-preserving 64/16/20 split of (Z, y)          [KDD protocol]
    split="strat":  (Z, y) is the official TRAINING file; a seeded stratified
                    20% calibration split is drawn from it; (Zte, yte) is the
                    untouched official TEST file.                       [UNSW protocol]
    """
    log(f"--- pipeline [{tag}] split={split}, n={len(Z):,}")
    if split == "order":
        n = len(Z); n_fit, n_cal = int(0.64*n), int(0.16*n)
        Zf, yf = Z.iloc[:n_fit], y[:n_fit]
        Zc, yc = Z.iloc[n_fit:n_fit+n_cal], y[n_fit:n_fit+n_cal]
        Zt, yt = Z.iloc[n_fit+n_cal:], y[n_fit+n_cal:]
    else:
        Zf, Zc, yf, yc = train_test_split(Z, y, test_size=0.20, stratify=y, random_state=SEED)
        Zt, yt = Zte, yte
    log(f"    fit={len(yf):,} cal={len(yc):,} test={len(yt):,} | "
        f"cal normal-frac={float((yc==0).mean()):.3f} test normal-frac={float((yt==0).mean()):.3f}")

    idx0, idx1 = np.where(yf==0)[0], np.where(yf==1)[0]
    k = min(len(idx0), len(idx1))
    sel = np.concatenate([rng.choice(idx0, k, replace=False), rng.choice(idx1, k, replace=False)])
    rng.shuffle(sel)
    log(f"    balanced fit subset: {2*k:,} records")

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED))
    log("    training logistic regression ...")
    model.fit(Zf.iloc[sel], yf[sel])
    log("    training done; scoring calibration + test splits ...")
    p_cal_raw  = model.predict_proba(Zc)[:, 1]
    p_test_raw = model.predict_proba(Zt)[:, 1]

    log("    fitting isotonic calibration ...")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal_raw, yc)
    r = iso.predict(p_test_raw)
    yhat = (r >= t2).astype(int)                      # binary prediction from SAME r
    tiers = np.select([r > t3, r > t2, r > t1],
                      ["CRITICAL_INTERVENTION","HIGH_PRIORITY_ALERT","MONITOR"], "NORMAL")

    log("    computing metrics ...")
    res = dict(
        accuracy=accuracy_score(yt,yhat), precision=precision_score(yt,yhat),
        recall=recall_score(yt,yhat), f1=f1_score(yt,yhat),
        roc_auc=roc_auc_score(yt,r), ap=average_precision_score(yt,r),
        ece_raw=ece(yt,p_test_raw), ece_cal=ece(yt,r),
        brier_raw=brier_score_loss(yt,p_test_raw), brier_cal=brier_score_loss(yt,r),
        confusion=confusion_matrix(yt,yhat).tolist())
    log(f"    acc={res['accuracy']:.4f} prec={res['precision']:.4f} rec={res['recall']:.4f} "
        f"f1={res['f1']:.4f} auc={res['roc_auc']:.4f} ap={res['ap']:.4f}")
    log(f"    ECE raw={res['ece_raw']:.4f} -> cal={res['ece_cal']:.4f} | "
        f"Brier raw={res['brier_raw']:.4f} -> cal={res['brier_cal']:.4f}")

    tier_tab = pd.crosstab(pd.Series(tiers,name="tier"), pd.Series(yt,name="true"))
    print(tier_tab, flush=True)
    tier_tab.to_csv(f"{OUT}/{tag}_tier_composition.csv")
    json.dump(res, open(f"{OUT}/{tag}_metrics.json","w"), indent=2)

    if figs:
        log("    rendering figures ...")
        def savefig(name):
            plt.tight_layout(); plt.savefig(f"{OUT}/{tag}_{name}.png", dpi=200); plt.close()
            log(f"      saved {tag}_{name}.png")
        cm = np.array(res["confusion"])
        plt.figure(figsize=(5.5,4.6)); plt.imshow(cm, cmap="viridis"); plt.title("Confusion Matrix"); plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",
                         color=("black" if cm[i,j] > cm.max()*.6 else "white"), fontsize=13)
        plt.xticks([0,1]); plt.yticks([0,1]); plt.xlabel("Predicted label"); plt.ylabel("True label")
        savefig("confusion_matrix")
        fpr,tpr,_ = roc_curve(yt,r)
        plt.figure(figsize=(5.5,4.4)); plt.plot(fpr,tpr,label=f"AUC={res['roc_auc']:.3f}")
        plt.plot([0,1],[0,1],"--",color="orange"); plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); savefig("roc_curve")
        pr,rc,_ = precision_recall_curve(yt,r)
        plt.figure(figsize=(5.5,4.4)); plt.plot(rc,pr,label=f"AP={res['ap']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
        savefig("precision_recall_curve")
        plt.figure(figsize=(5.8,4.6))
        for probs,lab in [(p_test_raw,"raw"),(r,"isotonic-calibrated")]:
            edges=np.linspace(0,1,16); xs,ys=[],[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                m=(probs>=lo)&(probs<hi) if hi<1 else (probs>=lo)&(probs<=hi)
                if m.sum(): xs.append(probs[m].mean()); ys.append(yt[m].mean())
            plt.plot(xs,ys,"o-",label=lab)
        plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical positive frequency"); plt.legend()
        plt.title(f"Reliability Diagram (ECE raw={res['ece_raw']:.4f}, cal={res['ece_cal']:.4f})")
        savefig("reliability_diagram")
        pd.Series(tiers).value_counts().plot(kind="bar", figsize=(6,4.4))
        plt.ylabel("count"); plt.title("Decision Distribution"); plt.xticks(rotation=20)
        savefig("decision_distribution")
        coefs = pd.Series(np.abs(model[-1].coef_[0]), index=Z.columns).sort_values(ascending=False)
        coefs.head(12)[::-1].plot(kind="barh", figsize=(6.5,5)); plt.xlabel("Importance")
        plt.title("Top Feature Importances"); savefig("feature_importances")
        coefs.to_csv(f"{OUT}/{tag}_feature_importance.csv")
        log("      top-5 features: " + ", ".join(f"{a}={b:.4f}" for a,b in coefs.head(5).items()))
        plt.figure(figsize=(6,4.4)); plt.hist([r[yt==0],r[yt==1]],bins=40,label=["True class 0","True class 1"])
        plt.xlabel("Calibrated probability"); plt.ylabel("Frequency"); plt.legend()
        plt.title("Calibrated Probability Histogram by True Class"); savefig("score_histogram_by_class")
        plt.figure(figsize=(5,4.6)); plt.boxplot([r[yt==0],r[yt==1]], tick_labels=["True class 0","True class 1"])
        plt.ylabel("Calibrated probability"); plt.title("Calibrated Probability Boxplot by True Class")
        savefig("score_boxplot_by_class")
        plt.figure(figsize=(6,4.2)); plt.plot(np.sort(r)[::-1]); plt.xlabel("Rank")
        plt.ylabel("Calibrated probability"); plt.title("Ranked Risk Scores"); savefig("ranked_risk_scores")
        plt.figure(figsize=(8.5,7.5)); plt.imshow(Z.corr(), cmap="viridis", vmin=-1, vmax=1); plt.colorbar()
        plt.xticks(range(len(Z.columns)), Z.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(Z.columns)), Z.columns, fontsize=7)
        plt.title("Engineered Feature Correlation Matrix"); savefig("feature_correlation_matrix")

        log("    tier-boundary sensitivity sweep ...")
        rows=[]
        for a in np.arange(0.40, 0.601, 0.05):
            for b in np.arange(0.55, 0.751, 0.05):
                for c in np.arange(0.70, 0.901, 0.05):
                    if not (a < b < c): continue
                    t = np.select([r>c, r>b, r>a], ["CRIT","HIGH","MON"], "NORM")
                    rows.append(dict(t1=round(a,2), t2=round(b,2), t3=round(c,2),
                        alert_volume=int(((t=="CRIT")|(t=="HIGH")).sum()),
                        monitor_size=int((t=="MON").sum()),
                        attacks_in_NORMAL=int(((t=="NORM")&(yt==1)).sum()),
                        attacks_in_MONITOR=int(((t=="MON")&(yt==1)).sum())))
        pd.DataFrame(rows).to_csv(f"{OUT}/{tag}_tier_sensitivity.csv", index=False)
        log(f"    sweep done ({len(rows)} configurations)")
    del Zf, Zc, Zt, p_cal_raw, p_test_raw; gc.collect()
    return res

# ================= KDDCup99 (10% subset -- percent10=True is essential) =================
log("loading KDDCup99 10% subset via scikit-learn (percent10=True) ...")
from sklearn.datasets import fetch_kddcup99
raw = fetch_kddcup99(percent10=True, as_frame=True).frame   # 494,021 rows
log(f"loaded: {len(raw):,} rows x {raw.shape[1]} cols")

lab = raw["labels"].astype(str).str.strip("b'\".")
y = (lab != "normal").astype(np.int8).values
log(f"labels: attack={int(y.sum()):,} normal={int((1-y).sum()):,} "
    "(expected 396,743 / 97,278)")

need = ["serror_rate","rerror_rate","duration","src_bytes","dst_bytes",
        "count","srv_count","same_srv_rate","diff_srv_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate"]
num = raw[need].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
del raw, lab; gc.collect()
log("numeric conversion done (float32, only required columns kept)")

Z = pd.DataFrame(index=num.index)
Z["packet_loss"]    = num["serror_rate"].clip(0,1)
Z["latency"]        = np.log1p(num["duration"])
Z["cpu_usage"]      = np.log1p(num["src_bytes"])
Z["memory_usage"]   = np.log1p(num["dst_bytes"])
Z["anomaly_score"]  = (0.6*num["serror_rate"] + 0.4*num["rerror_rate"]).clip(0,1)
Z["integrity_risk"] = (0.5*Z["packet_loss"] + 0.5*Z["anomaly_score"]).clip(0,1)
ln = (Z["latency"]-Z["latency"].min())/(Z["latency"].max()-Z["latency"].min()+1e-12)
Z["trust_score"]    = (1-(0.4*Z["packet_loss"] + 0.4*Z["anomaly_score"] + 0.2*ln)).clip(0,1)
for c in ["count","srv_count","same_srv_rate","diff_srv_rate","dst_host_count",
          "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate"]:
    Z[c] = num[c]
Z = Z.astype(np.float32)
log(f"KDD engineered features built: {Z.shape}")

res_kdd = run_pipeline(Z, y, "kdd", split="order")

# ---- weight-perturbation sensitivity (KDD, no figures) ----
log("weight-perturbation sensitivity (KDD) ...")
rows=[]
for dw in [-0.1, 0.0, +0.1]:
    Z2 = Z.copy()
    Z2["anomaly_score"]  = ((0.6+dw)*num["serror_rate"] + (0.4-dw)*num["rerror_rate"]).clip(0,1).astype(np.float32)
    Z2["integrity_risk"] = (0.5*Z2["packet_loss"] + 0.5*Z2["anomaly_score"]).astype(np.float32)
    m = run_pipeline(Z2, y, f"kdd_w{dw:+.1f}", split="order", figs=False)
    rows.append(dict(perturbation=f"anomaly weights {0.6+dw:.1f}/{0.4-dw:.1f}",
                     roc_auc=round(m["roc_auc"],4), ap=round(m["ap"],4), ece=round(m["ece_cal"],4)))
    del Z2; gc.collect()
pd.DataFrame(rows).to_csv(f"{OUT}/kdd_weight_sensitivity.csv", index=False)
log("weight sensitivity saved")
del Z, num, y; gc.collect()

# ================= UNSW-NB15 (official partition, stratified cal split) =================
UNSW_DIR = "/content/drive/MyDrive/Datasets/Telecom/UNSW_NB15"
try:
    log("loading UNSW-NB15 official partition ...")
    tr = pd.read_csv(f"{UNSW_DIR}/UNSW_NB15_training-set.csv")
    te = pd.read_csv(f"{UNSW_DIR}/UNSW_NB15_testing-set.csv")
    log(f"train={len(tr):,} test={len(te):,} (expected 175,341 / 82,332)")

    def feats_unsw(df):
        tot = (df["spkts"] + df["dpkts"]).replace(0, 1)
        Zu = pd.DataFrame(index=df.index)
        Zu["packet_loss"]   = ((df["sloss"] + df["dloss"]) / tot).clip(0, 1)
        Zu["latency"]       = np.log1p(df["dur"])
        Zu["cpu_usage"]     = np.log1p(df["sbytes"])
        Zu["memory_usage"]  = np.log1p(df["dbytes"])
        er  = (df["sloss"]/tot).clip(0,1); er2 = (df["dloss"]/tot).clip(0,1)
        Zu["anomaly_score"]  = (0.6*er + 0.4*er2).clip(0,1)
        Zu["integrity_risk"] = (0.5*Zu["packet_loss"] + 0.5*Zu["anomaly_score"]).clip(0,1)
        ln = (Zu["latency"]-Zu["latency"].min())/(Zu["latency"].max()-Zu["latency"].min()+1e-12)
        Zu["trust_score"]    = (1-(0.4*Zu["packet_loss"] + 0.4*Zu["anomaly_score"] + 0.2*ln)).clip(0,1)
        for c in ["ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_dst_src_ltm",
                  "ct_state_ttl","smean","dmean"]:
            Zu[c] = df[c].astype(np.float32)
        for c in ["rate","sload","dload"]:
            Zu[c] = np.log1p(df[c].astype(np.float32).clip(lower=0))
        return Zu.fillna(0).astype(np.float32)

    Ztr, ytr = feats_unsw(tr), tr["label"].astype(np.int8).values
    Zte, yte = feats_unsw(te), te["label"].astype(np.int8).values
    del tr, te; gc.collect()
    log(f"UNSW engineered features built: train {Ztr.shape}, test {Zte.shape}")
    res_unsw = run_pipeline(Ztr, ytr, "unsw", split="strat", Zte=Zte, yte=yte)
except FileNotFoundError:
    log(f"UNSW-NB15 CSVs not found in {UNSW_DIR} -- download from Kaggle (mrwellsdavid/unsw-nb15) and retry")

log("ALL DONE.")
