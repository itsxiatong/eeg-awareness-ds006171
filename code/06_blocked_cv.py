# 敏感性分析：严格按时间分块的交叉验证（KFold，不打乱；每个测试折 = 一段连续的实验时间）
import mne, numpy as np, pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression; from sklearn.model_selection import KFold, StratifiedKFold
from mne.decoding import SlidingEstimator, cross_val_multiscore
mne.set_log_level("ERROR")
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "derivatives")
OUT = os.path.join(ROOT, "output"); os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); os.chdir(OUT)   # 所有结果与图写入 output/
beh = pd.read_csv("results_behavior_qc.csv"); subs = beh.loc[beh.exclude_reason.isna() | (beh.exclude_reason == ""), "sub"].tolist()
AUCb = []; order_ok = []; skf_info = []
for s in subs:
    e = mne.read_epochs(f"{D}/compact/{s}_task-nocue_desc-128Hz_epo.fif")["stimulus=='present'"].crop(-0.2, 1.0).resample(64)
    md = e.metadata; order_ok.append(bool(np.all(np.diff(md.trial.values) > 0)))         # epochs 按实验顺序排列？
    X, y = e.get_data(), (md.condition == "seen").values.astype(int)
    # 记录常规 StratifiedKFold 的测试折在时间上是否连续
    for tr, te in StratifiedKFold(5).split(X, y):
        skf_info.append(dict(sub=s, span=(te.max()-te.min()+1)/len(y), contiguous=len(te)==te.max()-te.min()+1))
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    # 时间分块的折中可能只有一类试次 → 汇总各折的样本外决策值，再统一计算 AUC
    from sklearn.metrics import roc_auc_score
    dec = np.zeros((len(y), X.shape[2]))
    for tr, te in KFold(5, shuffle=False).split(X):
        se = SlidingEstimator(clf, n_jobs=8).fit(X[tr], y[tr]); dec[te] = se.decision_function(X[te])
    AUCb.append(np.array([roc_auc_score(y, dec[:, t]) for t in range(X.shape[2])]))
    times = e.times*1000
A = np.array(AUCb); G = np.load("group_data.npz"); A0 = G["AUC"]
print("epochs in chronological order for all subjects:", all(order_ok))
si = pd.DataFrame(skf_info); print(f"StratifiedKFold test folds strictly contiguous: {si.contiguous.mean()*100:.0f}% of folds; median span {si.span.median()*100:.0f}% of session")
T, clu, cp, _ = mne.stats.permutation_cluster_1samp_test(A - .5, n_permutations=2000, tail=1, seed=0, out_type="mask")
for c, p in zip(clu, cp):
    mk = np.zeros(len(times), bool); mk[c[0] if isinstance(c, tuple) else c] = True; c = mk
    if p < .05: print(f"blocked-CV cluster {times[c][0]:.0f}–{times[c][-1]:.0f} ms, p={p:.4f}")
m, m0 = A.mean(0), A0.mean(0)
print(f"peak AUC blocked {m.max():.3f} at {times[m.argmax()]:.0f} ms vs stratified {m0.max():.3f} at {times[m0.argmax()]:.0f} ms")
b = A[:, times < 0].mean(1); t, p = stats.ttest_1samp(b, .5); print(f"baseline AUC blocked {b.mean():.3f}, t(30)={t:.2f}, p={p:.3f}")
print(f"correlation of group time courses r = {np.corrcoef(m, m0)[0,1]:.3f}; mean difference (stratified - blocked) {np.mean(m0-m):.4f}")
np.save("auc_blocked.npy", A)
# ---- 对比度匹配细节
R = pd.read_csv("results_contrast_control_v2.csv"); d = R.contrast_diff_matched; mn = beh.set_index("sub").loc[R["sub"], ["n_seen", "n_unseen"]].min(axis=1).values
ci = stats.t.interval(.95, len(d)-1, d.mean(), d.std(ddof=1)/np.sqrt(len(d)))
raw = beh.set_index("sub").loc[R["sub"]]; rd = raw.contrast_seen - raw.contrast_unseen
print(f"\nresidual contrast diff after matching: mean {d.mean():.6f}, 95% CI [{ci[0]:.6f}, {ci[1]:.6f}]; before matching mean {rd.mean():.5f} → reduction {100*(1-d.mean()/rd.mean()):.1f}%")
print(f"matched trials per condition: median {R.n_matched.median():.0f}, range {R.n_matched.min():.0f}–{R.n_matched.max():.0f}; % of smaller condition median {np.median(R.n_matched/mn*100):.0f}%, range {np.min(R.n_matched/mn*100):.0f}–{np.max(R.n_matched/mn*100):.0f}%")
