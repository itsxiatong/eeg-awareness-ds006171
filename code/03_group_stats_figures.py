# 组水平统计与作图
import mne, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "output"); os.makedirs(os.path.join(OUT, "figs"), exist_ok=True); os.chdir(OUT)   # 所有结果与图写入 output/
from scipy import stats
mne.set_log_level("ERROR")
ERP = {k: mne.read_evokeds(f"erp_{k}-ave.fif") for k in ["seen", "unseen", "catch"]}; N = len(ERP["seen"])
G = np.load("group_data.npz"); out = {}
info = ERP["seen"][0].info; times = ERP["seen"][0].times; tm = times*1000
roi = {"P3b": ["CPz","CPPz","Pz","P1","P2"], "VAN": ["PO7","PO8","P7","P8","O1","O2"]}

# ---- (1) ERP 时空聚类置换检验：全部 128 导 × 0–800 ms，看见 vs 没看见
X = np.array([s.data - u.data for s, u in zip(ERP["seen"], ERP["unseen"])])        # 被试 × 导联 × 时间
tsel = (times >= 0) & (times <= 0.8); Xc = X[:, :, tsel].transpose(0, 2, 1)          # → 被试 × 时间 × 导联
adj, _ = mne.channels.find_ch_adjacency(info, "eeg")
T, clu, cp, _ = mne.stats.spatio_temporal_cluster_1samp_test(Xc, adjacency=adj, n_permutations=2000, tail=0, seed=0, n_jobs=8,
                                                            threshold=stats.t.ppf(0.995, N-1))   # 簇形成阈值 p<0.01
clus = []
for c, p in zip(clu, cp):
    if p < 0.05:
        ti, ci = c; tt = tm[tsel][np.unique(ti)]; sign = np.sign(T[ti, ci].mean())
        clus.append(dict(sign="positive" if sign > 0 else "negative", start_ms=round(tt.min()), end_ms=round(tt.max()),
                         n_channels=len(np.unique(ci)), p=round(p, 4),
                         channels=",".join(np.array(info.ch_names)[np.unique(ci)][:12])))
clus = pd.DataFrame(clus).sort_values("start_ms"); print("ERP spatio-temporal clusters (seen − unseen):\n", clus.to_string(index=False)); clus.to_csv("results_erp_clusters.csv", index=False)

# ---- (2) ROI 平均幅值
def amp(evs, chs, t0, t1): return np.array([e.copy().pick(chs).crop(t0, t1).data.mean()*1e6 for e in evs])
rr = []
for name, chs, w in [("VAN (occipito-temporal)", roi["VAN"], (0.18, 0.28)), ("P3b (centro-parietal)", roi["P3b"], (0.35, 0.65))]:
    a, u = amp(ERP["seen"], chs, *w), amp(ERP["unseen"], chs, *w); t, p = stats.ttest_rel(a, u); d = (a-u).mean()/(a-u).std(ddof=1)
    rr.append([name, f"{int(w[0]*1000)}-{int(w[1]*1000)} ms", a.mean(), u.mean(), t, p, d, int(((a-u) > 0).sum() if "P3b" in name else ((a-u) < 0).sum())])
rr = pd.DataFrame(rr, columns=["component", "window", "seen µV", "unseen µV", f"t({N-1})", "p", "Cohen's dz", f"subjects in expected direction /{N}"]).round(4)
rr["p (Bonferroni, 2 tests)"] = np.minimum(rr["p"] * 2, 1).round(4)
print(rr.to_string(index=False)); rr.to_csv("results_erp_roi.csv", index=False)

# 图 1：ROI 波形
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (name, chs), win in zip(axs, [("Occipito-temporal (VAN)", roi["VAN"]), ("Centro-parietal (P3b)", roi["P3b"])], [(180, 280), (350, 650)]):
    for cond, col in [("seen", "#c0504d"), ("unseen", "#4f81bd"), ("catch", "#999999")]:
        sj = np.array([e.copy().pick(chs).data.mean(0) for e in ERP[cond]])*1e6; m, se = sj.mean(0), sj.std(0, ddof=1)/np.sqrt(N)
        ax.plot(tm, m, color=col, label=cond if cond != "catch" else "catch (no stimulus)"); ax.fill_between(tm, m-se, m+se, color=col, alpha=.2)
    ax.axvspan(*win, color="orange", alpha=.1); ax.axvline(0, c="k", lw=.6); ax.axhline(0, c="k", lw=.6)
    ax.set_title(f"{name}, N = {N}"); ax.set_xlabel("Time (ms)"); ax.set_ylabel("µV"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig1_erp_roi.png", dpi=300); plt.close()
# 图 2：差异波地形图
diff = mne.combine_evoked([mne.grand_average(ERP["seen"]), mne.grand_average(ERP["unseen"])], weights=[1, -1])
f = diff.plot_topomap(times=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7], ch_type="eeg", show=False, time_unit="ms", vlim=(-2.5, 2.5))
f.suptitle(f"Seen − Unseen (µV), N = {N}"); f.savefig("figs/fig2_topo_diff.png", dpi=300); plt.close()
# 图 3：聚类时间-导联图
fig, ax = plt.subplots(figsize=(8, 4.5)); Tm = T.T.copy(); mask = np.zeros_like(Tm, bool)
for c, p in zip(clu, cp):
    if p < 0.05: mask[c[1], c[0]] = True
order = np.argsort([info["chs"][i]["loc"][1] for i in range(len(info.ch_names))])[::-1]           # 前→后排序
im = ax.imshow(np.where(mask, Tm, np.nan)[order], aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8, extent=[0, 800, 128, 0])
ax.imshow(np.where(mask, np.nan, Tm)[order], aspect="auto", cmap="Greys", vmin=-8, vmax=8, alpha=.25, extent=[0, 800, 128, 0])
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Channels (anterior → posterior)"); ax.set_title("Significant clusters (t-values), seen − unseen"); plt.colorbar(im, ax=ax, label="t")
plt.tight_layout(); plt.savefig("figs/fig3_clusters.png", dpi=300); plt.close()

# ---- (3) 时频：时间×频率聚类置换检验
tf_t, fr = G["tf_times"], G["freqs"]; Dtf = G["TFs"] - G["TFu"]
Ttf, clu_tf, cp_tf, _ = mne.stats.permutation_cluster_1samp_test(Dtf, n_permutations=2000, tail=0, seed=0, threshold=stats.t.ppf(0.995, N-1), n_jobs=8, out_type="mask")
tfc = []
for c, p in zip(clu_tf, cp_tf):
    if p < 0.05:
        fi, ti = np.where(c); tfc.append(dict(sign="+" if Ttf[c].mean() > 0 else "−", freq=f"{fr[fi].min()}-{fr[fi].max()} Hz", time=f"{round(tf_t[ti].min()*1000)}-{round(tf_t[ti].max()*1000)} ms", p=round(p, 4), p_permutation_unrounded=repr(float(p))))
tfc = pd.DataFrame(tfc); print("TF clusters:\n", tfc.to_string(index=False)); tfc.to_csv("results_tf_clusters.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.7)); ext = [tf_t[0]*1000, tf_t[-1]*1000, fr[0], fr[-1]]
for a, dat, ttl, v in [(ax[0], G["TFs"].mean(0), "Seen", 2.5), (ax[1], G["TFu"].mean(0), "Unseen", 2.5), (ax[2], Dtf.mean(0), "Seen − Unseen", 1.0)]:
    im = a.imshow(dat, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v); plt.colorbar(im, ax=a, label="dB")
    a.axvline(0, c="k", lw=.6); a.set_title(f"{ttl} (posterior ROI)"); a.set_xlabel("Time (ms)"); a.set_ylabel("Hz")
sigm = np.zeros_like(Ttf, bool)
for c, p in zip(clu_tf, cp_tf):
    if p < 0.05: sigm |= c
if sigm.any(): ax[2].contour(sigm, levels=[.5], colors="k", linewidths=1, extent=ext, origin="lower")
plt.tight_layout(); plt.savefig("figs/fig4_tfr.png", dpi=300); plt.close()

# ---- (4) 解码：时间聚类置换检验 + 时间泛化
A = G["AUC"]; dt = G["dec_times"]*1000; m, se = A.mean(0), A.std(0, ddof=1)/np.sqrt(N)
Ta, clu_a, cp_a, _ = mne.stats.permutation_cluster_1samp_test(A - .5, n_permutations=2000, tail=1, seed=0, out_type="mask")
sig = np.zeros(len(dt), bool); dc = []
def to_mask(c, n):
    m_ = np.zeros(n, bool); m_[c if not isinstance(c, tuple) else c[0]] = True; return m_
for c, p in zip(clu_a, cp_a):
    c = to_mask(c, len(dt))
    if p < 0.05: sig |= c; dc.append(dict(start_ms=round(dt[c][0]), end_ms=round(dt[c][-1]), p=round(p, 4)))
dc = pd.DataFrame(dc); print("Decoding clusters:\n", dc.to_string(index=False)); dc.to_csv("results_decoding_clusters.csv", index=False)
print("Decoding peak AUC %.3f ± %.3f at %d ms" % (m.max(), se[m.argmax()], dt[m.argmax()]))
GEN = G["GEN"].mean(0)
Tg, clu_g, cp_g, _ = mne.stats.permutation_cluster_1samp_test(G["GEN"] - .5, n_permutations=1000, tail=1, seed=0, out_type="mask")
gm = np.zeros_like(GEN, bool)
for c, p in zip(clu_g, cp_g):
    if p < 0.05:
        if isinstance(c, tuple): mm = np.zeros_like(GEN, bool); mm[c] = True; c = mm
        gm |= c
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
ax[0].plot(dt, m, color="#c0504d", lw=2); ax[0].fill_between(dt, m-se, m+se, color="#c0504d", alpha=.25)
ax[0].axhline(.5, c="k", ls="--", lw=.7); ax[0].axvline(0, c="k", lw=.6)
if sig.any(): ax[0].fill_between(dt, .47, .48, where=sig, color="k", step="mid", label="cluster p < 0.05")
ax[0].set_xlabel("Time (ms)"); ax[0].set_ylabel("AUC"); ax[0].set_title(f"Decoding seen vs unseen (N = {N})"); ax[0].legend(fontsize=8)
im = ax[1].imshow(GEN, origin="lower", extent=[dt[0], dt[-1], dt[0], dt[-1]], cmap="RdBu_r", vmin=.4, vmax=.6)
if gm.any(): ax[1].contour(gm, levels=[.5], colors="k", linewidths=.8, extent=[dt[0], dt[-1], dt[0], dt[-1]], origin="lower")
ax[1].set_xlabel("Testing time (ms)"); ax[1].set_ylabel("Training time (ms)"); ax[1].set_title("Temporal generalization"); plt.colorbar(im, ax=ax[1], label="AUC")
plt.tight_layout(); plt.savefig("figs/fig5_decoding.png", dpi=300); plt.close()

# ---- (5) 刺激前 alpha
pre = pd.read_csv("results_prestim_alpha.csv"); t, p = stats.ttest_rel(pre.seen, pre.unseen)
print("Pre-stimulus alpha (8–12 Hz, −500 to −100 ms): seen %.2f vs unseen %.2f dB, t(%d)=%.2f, p=%.4f, %d/%d lower before seen" %
      (pre.seen.mean(), pre.unseen.mean(), N-1, t, p, (pre.seen < pre.unseen).sum(), N))
