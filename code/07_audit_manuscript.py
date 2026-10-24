"""审计：把论文中报告的每个关键数字，与 output/ 中最终运行的结果逐项对照。
用法：python3 code/07_audit_manuscript.py path/to/manuscript.txt   （论文纯文本：pandoc -t plain --wrap=none paper.docx -o paper.txt）
输出：output/AUDIT.md"""
import sys, os, re, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "output")
ms = re.sub(r"\s+", " ", open(sys.argv[1], encoding="utf-8").read()).replace("−", "-").replace("–", "-")
q = pd.read_csv(f"{OUT}/results_behavior_qc.csv"); q["included"] = q.included.astype(str).str.lower().isin(["true", "1"]); inc = q[q.included]
roi = pd.read_csv(f"{OUT}/results_erp_roi.csv").set_index("component")
tf = pd.read_csv(f"{OUT}/results_tf_clusters.csv"); dec = pd.read_csv(f"{OUT}/results_decoding_clusters.csv")
cc = pd.read_csv(f"{OUT}/results_contrast_control_v2.csv"); pre = pd.read_csv(f"{OUT}/results_prestim_alpha.csv")
G = np.load(f"{OUT}/group_data.npz"); A = G["AUC"]; dt = G["dec_times"]*1000; m = A.mean(0); se = A.std(0, ddof=1)/np.sqrt(len(A))
DZ = "Cohen's dz"
van, p3b = roi.loc["VAN (occipito-temporal)"], roi.loc["P3b (centro-parietal)"]
f2 = lambda x: f"{x:.2f}"; from decimal import Decimal, ROUND_HALF_UP
def rnd(x, nd):   # 十进制四舍五入（避免二进制浮点使 0.0065 被舍为 0.006）
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
pfmt = lambda p: "p < .001" if p < .001 else ("p = " + rnd(p, 3 if p < .10 else 2).replace("0.", ".", 1))
tt = lambda col: stats.ttest_1samp(cc[col], 0)
rows = []
def add(section, claim, value, source):
    v = value.replace("−", "-"); rows.append(dict(section=section, claim=claim, value=value, source=source, in_manuscript="yes" if v in ms else "NO"))
add("Sample", "participants included", f"N = {len(inc)}", "results_behavior_qc.csv")
add("Behavior", "hit rate (mean ± SD)", f"{inc.hit_rate.mean()*100:.1f}% ± {inc.hit_rate.std()*100:.1f}%", "results_behavior_qc.csv")
add("Behavior", "false-alarm rate", f"{inc.fa_rate.mean()*100:.1f}% ± {inc.fa_rate.std()*100:.1f}%", "results_behavior_qc.csv")
t, p = stats.ttest_rel(inc.contrast_seen, inc.contrast_unseen)
add("Behavior", "contrast seen vs unseen", f"{inc.contrast_seen.mean():.4f} vs {inc.contrast_unseen.mean():.4f}", "results_behavior_qc.csv")
add("Behavior", "contrast difference test", f"t(30) = {t:.2f}", "results_behavior_qc.csv")
add("Behavior", "participants with higher contrast on seen trials", f"{(inc.contrast_seen > inc.contrast_unseen).sum()} of 31", "results_behavior_qc.csv")
add("ERP", "VAN amplitudes", f"{van['seen µV']:.2f} vs {van['unseen µV']:.2f} µV", "results_erp_roi.csv")
add("ERP", "VAN test", f"t(30) = {van['t(30)']:.2f}, {pfmt(van['p'])}", "results_erp_roi.csv")
add("ERP", "VAN effect size", f"dz = {van[DZ]:.2f}", "results_erp_roi.csv")
add("ERP", "VAN consistency", f"{int(van.iloc[6])} of 31", "results_erp_roi.csv")
add("ERP", "P3b amplitudes", f"{p3b['seen µV']:.2f} vs {p3b['unseen µV']:.2f} µV", "results_erp_roi.csv")
add("ERP", "P3b test", f"t(30) = {p3b['t(30)']:.2f}, {pfmt(p3b['p'])}", "results_erp_roi.csv")
add("ERP", "P3b effect size", f"dz = {p3b[DZ]:.2f}", "results_erp_roi.csv")
add("ERP", "P3b consistency", f"{int(p3b.iloc[6])} of 31", "results_erp_roi.csv")
pos, neg = tf[tf.sign == "+"].iloc[0], tf[tf.sign != "+"].iloc[0]
add("Time-frequency", "positive cluster range", f"{pos.freq.replace(' Hz','')} Hz, {pos.time}", "results_tf_clusters.csv")
add("Time-frequency", "positive cluster p", pfmt(pos.p), "results_tf_clusters.csv")
add("Time-frequency", "negative cluster range", f"{neg.freq.replace(' Hz','')} Hz and {neg.time}", "results_tf_clusters.csv")
t, p = stats.ttest_rel(pre.seen, pre.unseen)
add("Time-frequency", "prestimulus alpha", f"{pre.seen.mean():.2f} vs {pre.unseen.mean():.2f} dB; t(30) = {t:.2f}, {pfmt(p)}", "results_prestim_alpha.csv")
add("Decoding", "first significant time point", f"from {int(dec.start_ms.iloc[0])} ms", "results_decoding_clusters.csv")
add("Decoding", "peak AUC", f"AUC = {m.max():.3f} ± {se[m.argmax()]:.3f}", "group_data.npz")
add("Decoding", "peak latency", f"peaking at {dt[m.argmax()]:.0f} ms", "group_data.npz")
b = A[:, (dt >= -200) & (dt < 0)].mean(1); t, p = stats.ttest_1samp(b, .5)   # 与论文一致：-200 至 0 ms
add("Decoding", "baseline AUC", f"AUC = {b.mean():.3f}, t(30) = {t:.2f}, {pfmt(p)}", "group_data.npz")
if os.path.exists(f"{OUT}/auc_blocked.npy"):
    Ab = np.load(f"{OUT}/auc_blocked.npy"); mb = Ab.mean(0)
    add("Decoding", "blocked-CV peak AUC", f"peak AUC = {mb.max():.3f} at {dt[mb.argmax()]:.0f} ms", "auc_blocked.npy")
    add("Decoding", "blocked-CV correlation with main analysis", f"r = {np.corrcoef(m, mb)[0,1]:.3f}".replace("0.", "."), "auc_blocked.npy")
for lab, col, unit in [("P3b contrast-matched difference", "P3b_matched", "µV"), ("VAN contrast-matched difference", "VAN_matched", "µV")]:
    t, p = tt(col); add("Contrast control", lab, f"{cc[col].mean():.2f} µV (t(30) = {t:.2f}, {pfmt(p)})", "results_contrast_control_v2.csv")
for lab, col in [("P3b beta(seen)", "P3b_b_seen"), ("VAN beta(seen)", "VAN_b_seen"), ("P3b beta(contrast)", "P3b_b_contrast"), ("VAN beta(contrast)", "VAN_b_contrast")]:
    t, p = tt(col); add("Contrast control", lab, f"β = {cc[col].mean():.2f}", "results_contrast_control_v2.csv"); add("Contrast control", lab + " test", f"t(30) = {t:.2f}, {pfmt(p)}", "results_contrast_control_v2.csv")
d = pd.DataFrame(rows)
with open(f"{OUT}/AUDIT.md", "w") as fh:
    fh.write(f"# Manuscript audit\n\n{(d.in_manuscript=='yes').sum()} / {len(d)} values found verbatim in the manuscript.\n\n"
             "**Scope.** This audit checks that the statistics reported in the manuscript match the final result files in `output/`. "
             "Simple statistics (t tests, means, proportions) are recomputed here from per-subject values; cluster-permutation results, "
             "decoding time courses and time-frequency arrays are read from the files written by scripts 02-06 and are not recomputed from raw EEG. "
             "The audit therefore verifies consistency between results and text; it does not by itself verify the correctness of the analysis pipeline "
             "or its statistical assumptions, and it is not a substitute for an independent rerun from raw data.\n\n")
    fh.write(d.to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string(index=False))
print(d.to_string(index=False)); print(f"\n{(d.in_manuscript=='yes').sum()} / {len(d)} found verbatim")
