# 组水平分析：读取 QC、应用剔除标准、行为数据
import mne, numpy as np, pandas as pd, glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
mne.set_log_level("ERROR"); rng = 0
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "derivatives")
OUT = os.path.join(ROOT, "output"); os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); os.chdir(OUT)   # 所有结果与图写入 output/
qc = pd.read_csv(f"{D}/qc_all.csv")
# ---------- 剔除标准（看结果之前预先设定）----------
qc["n_interp"] = qc.bads_post_ica.fillna("").apply(lambda s: len([c for c in s.split(";") if c]))
rows = []
import os
for s in qc["sub"]:
    f = f"{D}/compact/{s}_task-nocue_desc-128Hz_epo.fif"
    if not os.path.exists(f): rows.append(dict(sub=s)); continue      # 预处理质量已判定不合格、未传输的被试
    md = mne.read_epochs(f"{D}/compact/{s}_task-nocue_desc-128Hz_epo.fif", preload=False).metadata
    rows.append(dict(sub=s, n_seen=(md.condition=="seen").sum(), n_unseen=(md.condition=="unseen").sum(),
                     hit_rate=(md.query("stimulus=='present'").seen==1).mean(), fa_rate=md.query("stimulus=='absent'").seen.mean(),
                     contrast_seen=md.query("condition=='seen'").contrast.mean(), contrast_unseen=md.query("condition=='unseen'").contrast.mean()))
beh = qc.merge(pd.DataFrame(rows), on="sub")
beh["exclude_reason"] = ""
beh.loc[beh.n_interp > 12.8, "exclude_reason"] += "interp>10%;"
beh.loc[(beh.n_seen < 30) | (beh.n_unseen < 30), "exclude_reason"] += "<30 trials;"
beh.loc[beh.fa_rate > 0.3, "exclude_reason"] += "FA>30%;"
print(beh.loc[beh.exclude_reason != "", ["sub", "n_interp", "n_seen", "n_unseen", "fa_rate", "exclude_reason"]].to_string(index=False))
subs = beh.loc[beh.exclude_reason == "", "sub"].tolist(); N = len(subs); print("N included =", N)
beh["included"] = beh.exclude_reason == ""
beh.to_csv("results_behavior_qc.csv", index=False)
print(beh[["sub","n_interp","n_seen","n_unseen","fa_rate","exclude_reason","included"]].to_string(index=False))
b = beh[beh.exclude_reason == ""]
t, p = stats.ttest_rel(b.contrast_seen, b.contrast_unseen)
print("Behavior: hit rate %.2f±%.2f, FA %.3f±%.3f; contrast seen %.4f vs unseen %.4f (diff %.1f%%), t(%d)=%.2f p=%.3g" %
      (b.hit_rate.mean(), b.hit_rate.std(), b.fa_rate.mean(), b.fa_rate.std(), b.contrast_seen.mean(), b.contrast_unseen.mean(),
       100*(b.contrast_seen.mean()/b.contrast_unseen.mean()-1), N-1, t, p))

# ---------- 逐被试：ERP、时频、解码 ----------
freqs = np.arange(4, 31, 1); post = ["CPz","CPPz","Pz","P1","P2","PO3","PO4","POz","Oz","O1","O2"]
ERP = {"seen": [], "unseen": [], "catch": []}; TF = {"seen": [], "unseen": []}; PRE = []; AUC = []; GEN = []
for s in subs:
    e = mne.read_epochs(f"{D}/compact/{s}_task-nocue_desc-128Hz_epo.fif")
    ec = e.copy().crop(-0.2, 1.0)
    ERP["seen"].append(ec["condition=='seen'"].average()); ERP["unseen"].append(ec["condition=='unseen'"].average())
    ERP["catch"].append(ec["stimulus=='absent'"].average())
    ep = e["stimulus=='present'"]; chs = [c for c in post if c in e.ch_names]; pre = {}
    for cond in ["seen", "unseen"]:
        p_ = ep[f"condition=='{cond}'"].copy().pick(chs).compute_tfr("morlet", freqs=freqs, n_cycles=freqs/2, decim=2, average=True, return_itc=False)
        pre[cond] = 10*np.log10(p_.copy().crop(-0.5, -0.1).data.mean(0)[(freqs>=8)&(freqs<=12)].mean() * 1e12)
        p_.apply_baseline((-0.5, -0.2), mode="logratio"); p_.crop(-0.3, 1.0); TF[cond].append(p_.data.mean(0)*10)
    tf_times = p_.times; PRE.append(pre)
    ed = ep.copy().crop(-0.2, 1.0).resample(64); X = ed.get_data(); y = (ed.metadata.condition=="seen").values.astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    AUC.append(cross_val_multiscore(SlidingEstimator(clf, scoring="roc_auc", n_jobs=8), X, y, cv=5).mean(0))
    GEN.append(cross_val_multiscore(GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=8), X, y, cv=5).mean(0))
    dec_times = ed.times; print(s, "done", flush=True)
np.savez("group_data.npz", AUC=np.array(AUC), GEN=np.array(GEN), TFs=np.array(TF["seen"]), TFu=np.array(TF["unseen"]), tf_times=tf_times, dec_times=dec_times, freqs=freqs)
for k in ERP: mne.write_evokeds(f"erp_{k}-ave.fif", ERP[k], overwrite=True)
pd.DataFrame([dict(sub=s, **p) for s, p in zip(subs, PRE)]).to_csv("results_prestim_alpha.csv", index=False)
print("ALL SUBJECTS DONE")
