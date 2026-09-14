# Derived manuscript tables

These eight CSVs contain derived counts, summaries, split memberships, and checkpoint provenance for *Recurrent clip-level failures in video polyp segmentation*. They reproduce the accompanying manuscript's supplementary tables as supplied, without changing values or identifiers. They are not per-frame prediction receipts, original images, masks, or model weights.

| File | Rows | Contents |
|---|---:|---|
| `Supplementary_Data_1_checkpoint_provenance.csv` | 13 | Additional-model weight origins, training/selection records, known test exposure, and unresolved details |
| `Supplementary_Data_2_configuration_counts.csv` | 12 | Evaluated/failing clip counts and overlap with the reference failure set |
| `Supplementary_Data_3_SUN_counts.csv` | 258 | 86 SUN-SEG evaluation clips for each model and their combined runs, with preselection/retained errors and count bounds |
| `Supplementary_Data_4_fixed_quota_clips.csv` | 3096 | Clip-level outputs for every fixed-budget model, score, allocation, and tie-order comparison |
| `Supplementary_Data_5_source_split_clips.csv` | 2076 | Clip-level outputs for clip-ID/source-group calibration and both evaluation directions |
| `Supplementary_Data_6_fixed_quota_summary.csv` | 36 | Fixed-budget aggregate results and conditional intervals |
| `Supplementary_Data_7_source_split_summary.csv` | 36 | Calibration bookkeeping, aggregate results, and conditional intervals for the split comparisons |
| `Supplementary_Data_8_split_membership.csv` | 346 | Clip and source-group membership under the two split schemes |

## Read the counts

A frame fails when Dice is below 0.3. In tables 4–7, `N` is the number of frame–run observations, `B` their failures, `A` retained observations, and `E` retained failures. Risk is E/A and retention is A/N. Runs are pooled within the named model group; these counts are not independent images or patients.

`risk_pass` uses retained risk at most 0.15. `coverage_pass` uses retention at least 0.5, and `joint_pass` requires both. A zero retained count gives undefined risk. The count bound is max(0, B − (N − A))/A for A > 0. Table 3 uses expanded names for these same quantities.

Fixed-budget and source-group analyses are post hoc. Source-group intervals resample groups while holding predictions, selection outputs, and thresholds fixed. They describe conditional composition uncertainty rather than refitting or new-patient performance. Two-direction summaries use two fitted thresholds. Tables 4–8 provide derived results; the corresponding full experiment implementation and private inputs are not included in this public subset.

Checkpoint provenance records include incomplete training and selection histories. A recorded checksum identifies an artifact; it does not establish an independent training history or permission to redistribute weights. Blank or unresolved fields should not be interpreted as confirmed absence of test exposure.

Original dataset terms remain applicable. This directory does not grant a new license for the tables or redistribute provider datasets.
