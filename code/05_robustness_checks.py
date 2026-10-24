# 回应评审：(a) 时间泛化显著范围 (b) 重复抽样的对比度匹配 + 含试次序号的回归 (c) unseen vs catch 的贝叶斯因子
import mne, numpy as np, pandas as pd, statsmodels.formula.api as smf, pingouin as pg
from scipy import stats
mne.set_log_level("ERROR")
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "derivatives")
OUT = os.path.join(ROOT, "output"); os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); os.chdir(OUT)   # 所有结果与图写入 output/
G = np.load("group_data.npz"); dt = G["dec_times"]*1000
# (a) temporal generalization significant extent
Tg, clu, cp, _ = mne.stats.permutation_cluster_1samp_test(G["GEN"] - .5, n_permutations=1000, tail=1, seed=0, out_type="mask")
for c, p in zip(clu, cp):
    if p < .05:
        m = np.zeros_like(G["GEN"][0], bool); m[c if not isinstance(c, tuple) else c] = True
        tr, te = np.where(m)
        print(f"TG cluster p={p:.4f}: train {dt[tr].min():.0f}–{dt[tr].max():.0f} ms, test {dt[te].min():.0f}–{dt[te].max():.0f} ms, {m.mean()*100:.1f}% of matrix")
GEN = G["GEN"].mean(0); hi = GEN > 0.58; tr, te = np.where(hi); print(f"High-performance region (AUC>0.58): train {dt[tr].min():.0f}–{dt[tr].max():.0f}, test {dt[te].min():.0f}–{dt[te].max():.0f}")
# (b) contrast control: 100 random draws; regression with trial number
beh = pd.read_csv("results_behavior_qc.csv"); subs = beh.loc[beh.exclude_reason.isna() | (beh.exclude_reason == ""), "sub"].tolist()
roi = {"P3b": (["CPz","CPPz","Pz","P1","P2"], 0.35, 0.65), "VAN": (["PO7","PO8","P7","P8","O1","O2"], 0.18, 0.28)}
rng = np.random.default_rng(1); R = []
for s in subs:
    e = mne.read_epochs(f"{D}/compact/{s}_task-nocue_desc-128Hz_epo.fif")["stimulus=='present'"].crop(-0.2, 1.0)
    md = e.metadata.reset_index(drop=True); md["bin"] = pd.qcut(md.contrast.rank(method="first"), 10, labels=False)
    amps = {k: e.copy().pick(ch).crop(t0, t1).get_data().mean((1, 2))*1e6 for k, (ch, t0, t1) in roi.items()}
    seen = (md.condition == "seen").values; row = dict(sub=s)
    diffs = {k: [] for k in roi}; nmatch = []; cdiff = []
    for it in range(100):
        keep_s, keep_u = [], []
        for _, g in md.groupby("bin"):
            si, ui = g.index[g.condition == "seen"].values, g.index[g.condition == "unseen"].values; k = min(len(si), len(ui))
            keep_s += list(rng.choice(si, k, replace=False)); keep_u += list(rng.choice(ui, k, replace=False))
        nmatch.append(len(keep_s)); cdiff.append(md.contrast[keep_s].mean() - md.contrast[keep_u].mean())
        for k in roi: diffs[k].append(amps[k][keep_s].mean() - amps[k][keep_u].mean())
    row["n_matched"] = np.mean(nmatch); row["contrast_diff_matched"] = np.mean(cdiff)
    for k in roi:
        row[f"{k}_matched"] = np.mean(diffs[k])
        df = pd.DataFrame(dict(amp=amps[k], seen=seen.astype(int), contrast=(md.contrast-md.contrast.mean())/md.contrast.std(),
                               trial=(md.trial-md.trial.mean())/md.trial.std()))
        f = smf.ols("amp ~ seen + contrast + trial", df).fit()
        row[f"{k}_b_seen"] = f.params["seen"]; row[f"{k}_b_contrast"] = f.params["contrast"]; row[f"{k}_b_trial"] = f.params["trial"]
    R.append(row)
R = pd.DataFrame(R); R.to_csv("results_contrast_control_v2.csv", index=False); N = len(R)
t, p = stats.ttest_1samp(R.contrast_diff_matched, 0)
print(f"\nN={N}; matched trials/condition mean {R.n_matched.median():.0f} (median across subjects); residual contrast diff {R.contrast_diff_matched.mean():.5f}, t={t:.2f}, p={p:.3f}")
for col in [c for c in R.columns if "_" in c and c not in ("n_matched","contrast_diff_matched")]:
    t, p = stats.ttest_1samp(R[col], 0); print(f"{col:16s} mean {R[col].mean():6.3f} t({N-1})={t:6.2f} p={p:.4f}")
# (c) unseen vs catch Bayes factor (JZS, paired)
E = {k: mne.read_evokeds(f"erp_{k}-ave.fif") for k in ["unseen", "catch"]}
f = lambda evs: np.array([e.copy().pick(roi["P3b"][0]).crop(.35, .65).data.mean()*1e6 for e in evs])
u, c = f(E["unseen"]), f(E["catch"]); res = pg.ttest(u, c, paired=True)
print("\nUnseen vs catch P3b:", res[["T", "p_val", "BF10"]].to_string() if "p_val" in res else res.to_string())
