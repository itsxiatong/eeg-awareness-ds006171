"""第 1b 步：把每位被试的分段数据降采样到 128 Hz（组分析使用），并汇总质控表。
用法：python3 code/01b_make_compact.py"""
import mne, glob, os, pandas as pd
mne.set_log_level("ERROR")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "derivatives")
os.makedirs(f"{D}/compact", exist_ok=True)
for f in sorted(glob.glob(f"{D}/sub-*_task-nocue_epo.fif")):
    o = f"{D}/compact/" + os.path.basename(f).replace("_epo.fif", "_desc-128Hz_epo.fif")
    if os.path.exists(o) and os.path.getmtime(o) > os.path.getmtime(f): continue
    mne.read_epochs(f).pick("eeg").resample(128).save(o, fmt="single", overwrite=True); print("saved", o)
pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/sub-*_qc.csv"))]).to_csv(f"{D}/qc_all.csv", index=False)
