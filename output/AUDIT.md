# Manuscript audit

34 / 34 values found verbatim in the manuscript.

**Scope.** This audit checks that the statistics reported in the manuscript match the final result files in `output/`. Simple statistics (t tests, means, proportions) are recomputed here from per-subject values; cluster-permutation results, decoding time courses and time-frequency arrays are read from the files written by scripts 02-06 and are not recomputed from raw EEG. The audit therefore verifies consistency between results and text; it does not by itself verify the correctness of the analysis pipeline or its statistical assumptions, and it is not a substitute for an independent rerun from raw data.

| section          | claim                                            | value                                      | source                          | in_manuscript   |
|:-----------------|:-------------------------------------------------|:-------------------------------------------|:--------------------------------|:----------------|
| Sample           | participants included                            | N = 31                                     | results_behavior_qc.csv         | yes             |
| Behavior         | hit rate (mean ± SD)                             | 42.6% ± 11.4%                              | results_behavior_qc.csv         | yes             |
| Behavior         | false-alarm rate                                 | 6.6% ± 7.3%                                | results_behavior_qc.csv         | yes             |
| Behavior         | contrast seen vs unseen                          | 0.0337 vs 0.0326                           | results_behavior_qc.csv         | yes             |
| Behavior         | contrast difference test                         | t(30) = 7.69                               | results_behavior_qc.csv         | yes             |
| Behavior         | participants with higher contrast on seen trials | 29 of 31                                   | results_behavior_qc.csv         | yes             |
| ERP              | VAN amplitudes                                   | -1.26 vs -0.57 µV                          | results_erp_roi.csv             | yes             |
| ERP              | VAN test                                         | t(30) = -2.54, p = .017                    | results_erp_roi.csv             | yes             |
| ERP              | VAN effect size                                  | dz = -0.46                                 | results_erp_roi.csv             | yes             |
| ERP              | VAN consistency                                  | 23 of 31                                   | results_erp_roi.csv             | yes             |
| ERP              | P3b amplitudes                                   | 4.40 vs 1.82 µV                            | results_erp_roi.csv             | yes             |
| ERP              | P3b test                                         | t(30) = 9.29, p < .001                     | results_erp_roi.csv             | yes             |
| ERP              | P3b effect size                                  | dz = 1.67                                  | results_erp_roi.csv             | yes             |
| ERP              | P3b consistency                                  | 30 of 31                                   | results_erp_roi.csv             | yes             |
| Time-frequency   | positive cluster range                           | 4-9 Hz, 109-391 ms                         | results_tf_clusters.csv         | yes             |
| Time-frequency   | positive cluster p                               | p = .007                                   | results_tf_clusters.csv         | yes             |
| Time-frequency   | negative cluster range                           | 4-30 Hz and 281-1000 ms                    | results_tf_clusters.csv         | yes             |
| Time-frequency   | prestimulus alpha                                | 20.61 vs 20.81 dB; t(30) = -2.06, p = .048 | results_prestim_alpha.csv       | yes             |
| Decoding         | first significant time point                     | from 156 ms                                | results_decoding_clusters.csv   | yes             |
| Decoding         | peak AUC                                         | AUC = 0.649 ± 0.012                        | group_data.npz                  | yes             |
| Decoding         | peak latency                                     | peaking at 375 ms                          | group_data.npz                  | yes             |
| Decoding         | baseline AUC                                     | AUC = 0.497, t(30) = -0.81, p = .42        | group_data.npz                  | yes             |
| Decoding         | blocked-CV peak AUC                              | peak AUC = 0.627 at 422 ms                 | auc_blocked.npy                 | yes             |
| Decoding         | blocked-CV correlation with main analysis        | r = .996                                   | auc_blocked.npy                 | yes             |
| Contrast control | P3b contrast-matched difference                  | 2.47 µV (t(30) = 8.74, p < .001)           | results_contrast_control_v2.csv | yes             |
| Contrast control | VAN contrast-matched difference                  | -0.72 µV (t(30) = -2.62, p = .014)         | results_contrast_control_v2.csv | yes             |
| Contrast control | P3b beta(seen)                                   | β = 2.44                                   | results_contrast_control_v2.csv | yes             |
| Contrast control | P3b beta(seen) test                              | t(30) = 9.36, p < .001                     | results_contrast_control_v2.csv | yes             |
| Contrast control | VAN beta(seen)                                   | β = -0.67                                  | results_contrast_control_v2.csv | yes             |
| Contrast control | VAN beta(seen) test                              | t(30) = -2.42, p = .022                    | results_contrast_control_v2.csv | yes             |
| Contrast control | P3b beta(contrast)                               | β = 0.29                                   | results_contrast_control_v2.csv | yes             |
| Contrast control | P3b beta(contrast) test                          | t(30) = 3.92, p < .001                     | results_contrast_control_v2.csv | yes             |
| Contrast control | VAN beta(contrast)                               | β = 0.07                                   | results_contrast_control_v2.csv | yes             |
| Contrast control | VAN beta(contrast) test                          | t(30) = 0.88, p = .39                      | results_contrast_control_v2.csv | yes             |