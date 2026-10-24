"""
第 1–2 步：读取原始 EEG → 滤波 → 降采样 → 重参考 → ICA 去眼电 → 按刺激分段
用法：python3 01_preprocess.py sub-01
"""
import sys, os, gc, numpy as np, pandas as pd, mne
mne.set_log_level("WARNING")
sub = sys.argv[1]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
eeg_dir = f"{root}/ds006171/{sub}/eeg"; out = f"{root}/derivatives"; os.makedirs(out, exist_ok=True)
bdf = f"{eeg_dir}/{sub}_task-nocue_eeg.bdf"

# ---------- 1. 分批读取导联、滤波 0.1–40 Hz、降采样到 256 Hz（节省内存）----------
# 0.1 Hz 高通：去掉缓慢漂移，同时保留 P3b 这类慢成分；40 Hz 低通：去掉肌电和 50 Hz 工频
hdr = mne.io.read_raw_bdf(bdf, preload=False)
eeg_chs = [c for c in hdr.ch_names if c[0] in "ABCD" and c[1:].isdigit()]   # A1–D32 共 128 导
eog_chs = ["EXG1", "EXG2", "EXG3", "EXG4"]                                 # VEOG1/2, HEOG1/2
import time; T0 = time.time(); BUDGET = 140          # 运行环境每次最多 3 分钟 → 分段保存、可断点续跑
tmp = f"{out}/tmp_{sub}"; os.makedirs(tmp, exist_ok=True)
groups = [(eeg_chs + eog_chs)[i:i + 17] for i in range(0, len(eeg_chs) + 4, 17)]
for gi, chs in enumerate(groups):
    f = f"{tmp}/chunk{gi:02d}_raw.fif"
    if os.path.exists(f): continue
    if time.time() - T0 > BUDGET: print(sub, "PARTIAL: filtering", gi, "/", len(groups)); sys.exit(3)
    r = mne.io.read_raw_bdf(bdf, preload=False); r.pick(chs); r.load_data()
    r.filter(0.1, 40, n_jobs=1); r.resample(256); r.save(f, fmt="single", overwrite=True); del r; gc.collect()
pieces = [mne.io.read_raw_fif(f"{tmp}/chunk{gi:02d}_raw.fif", preload=True) for gi in range(len(groups))]
raw = pieces[0].add_channels(pieces[1:], force_update_info=True); del pieces; gc.collect()
raw.set_channel_types({c: "eog" for c in eog_chs})
raw.set_montage("biosemi128")                       # Biosemi 128 导 ABC 排布的标准电极位置

# 把 A1…D32 改成数据集提供的 10-20 扩展命名（Cz、Pz…），方便解读
names = pd.read_csv(f"{eeg_dir}/{sub}_task-nocue_channels.tsv", sep="\t").query("type=='EEG'").name.tolist()
raw.rename_channels(dict(zip(eeg_chs, names)))
raw.rename_channels({"EXG1": "VEOG1", "EXG2": "VEOG2", "EXG3": "HEOG1", "EXG4": "HEOG2"})
# 双极眼电：上下电极相减 = 垂直眼电（眨眼），左右相减 = 水平眼电（眼跳）
raw = mne.set_bipolar_reference(raw, "VEOG1", "VEOG2", ch_name="VEOG", drop_refs=True)
raw = mne.set_bipolar_reference(raw, "HEOG1", "HEOG2", ch_name="HEOG", drop_refs=True)
raw.set_channel_types({"VEOG": "eog", "HEOG": "eog"})

# ---------- 2. 坏导检测（方差异常）+ 平均参考 ----------
data = raw.get_data(picks="eeg"); sd = data.std(1); z = (sd - np.median(sd)) / (1.4826 * np.median(np.abs(sd - np.median(sd))))
# 标准：信号标准差相对所有导联的稳健 z 值 > 3.5（噪声过大），或几乎平坦（接触不良）
# 注意：额叶导联因眨眼而幅度大是正常的，交给 ICA 处理，不当作坏导
raw.info["bads"] = [raw.ch_names[i] for i in np.where((np.abs(z) > 3.5) | (sd < 1e-7))[0]]
print(sub, "bad channels:", raw.info["bads"])
raw.interpolate_bads(reset_bads=True)
raw.set_eeg_reference("average")

# ---------- 3. ICA 去除眨眼和眼动 ----------
# ICA 把信号分解成相互独立的成分；与眼电通道高度相关的成分就是眨眼/眼动，去掉即可
ica_file = f"{tmp}/ica.fif"
if os.path.exists(ica_file):
    ica = mne.preprocessing.read_ica(ica_file)
else:
    if time.time() - T0 > BUDGET - 20: print(sub, "PARTIAL: before ICA"); sys.exit(3)
    ica_raw = raw.copy().filter(1, None)             # ICA 在 1 Hz 高通的副本上拟合更稳定
    ica = mne.preprocessing.ICA(n_components=25, method="fastica", random_state=0, max_iter=800)
    ica.fit(ica_raw, picks="eeg", decim=3)
    eog_idx, eog_scores = ica.find_bads_eog(ica_raw, ch_name=["VEOG", "HEOG"], measure="correlation", threshold=0.3)
    ica.exclude = eog_idx[:4]; ica.save(ica_file, overwrite=True)
    del ica_raw; gc.collect()
    if time.time() - T0 > BUDGET - 10: print(sub, "PARTIAL: ICA saved"); sys.exit(3)
print(sub, "ICA components removed:", ica.exclude)
ica.apply(raw)

# ---------- 4. 事件：用光电二极管（128）校正刺激呈现时间 ----------
ev = pd.read_csv(f"{eeg_dir}/{sub}_task-nocue_events.tsv", sep="\t")
rows = []
for i, r in ev.iterrows():
    if r.trial_type in (11, 12, 21, 22, 31, 32):
        nxt = ev.iloc[i + 1] if i + 1 < len(ev) else None
        onset = nxt.onset if (nxt is not None and nxt.trial_type == 128) else r.onset   # 真正上屏时刻
        rows.append(dict(onset=onset, code=int(r.trial_type), trial=int(r.trial_number),
                         contrast=r.gabor_contrast, seen=r.subjective_r))
md = pd.DataFrame(rows)
md["stimulus"] = np.where(md.code >= 31, "absent", "present")
md["side"] = np.where(md.code.isin([11, 12, 31]), "left", "right")
md["report"] = md.seen.map({1: "seen", 0: "unseen"})
md["condition"] = np.where(md.stimulus == "absent", "catch_" + md.report, md.report)
events = np.c_[np.round(md.onset * raw.info["sfreq"]).astype(int), np.zeros(len(md), int), md.code.values]

# ---------- 5. 分段：刺激前 800 ms 到刺激后 1500 ms（时频分析需要较长的边缘缓冲），以 -200~0 ms 为基线 ----------
epochs = mne.Epochs(raw, events, tmin=-0.8, tmax=1.5, baseline=(-0.2, 0), metadata=md,
                    reject=None, picks=["eeg", "eog"], preload=True, event_repeated="drop")
# ---------- 6. ICA 之后再找坏导：在 >20% 试次里峰峰值 >200 µV 的导联（-500~1200 ms 窗口） → 插值 ----------
d = epochs.copy().crop(-0.5, 1.2).get_data(picks="eeg"); ptp = d.max(2) - d.min(2)   # 只在 -500~1200 ms 内检查伪迹
eeg_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True)]
frac = (ptp > 200e-6).mean(0)
epochs.info["bads"] = [eeg_names[i] for i in np.where(frac > 0.2)[0]]
print(sub, "bad channels (post-ICA):", epochs.info["bads"])
if epochs.info["bads"]: epochs.interpolate_bads(reset_bads=True)
# ---------- 7. 剔除仍然超标的试次 ----------
n0 = len(epochs)
d = epochs.copy().crop(-0.5, 1.2).get_data(picks="eeg"); bad_trial = ((d.max(2) - d.min(2)) > 200e-6).any(1)
epochs.drop(np.where(bad_trial)[0], reason="PTP>200uV")
print(sub, "epochs kept:", len(epochs), "/", n0); print(epochs.metadata.condition.value_counts().to_dict())
epochs.save(f"{out}/{sub}_task-nocue_epo.fif", overwrite=True, fmt="single")
ica.save(f"{out}/{sub}_task-nocue_ica.fif", overwrite=True)
pd.DataFrame([dict(sub=sub, bads=";".join(epochs.info["bads"] or []), bads_post_ica=";".join(eeg_names[i] for i in np.where(frac > 0.2)[0]),
                   ica_excluded=str([int(x) for x in ica.exclude]), kept=len(epochs), total=n0)]).to_csv(f"{out}/{sub}_qc.csv", index=False)
print(sub, "DONE")
