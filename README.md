# Recurrent clip-level failures in video polyp segmentation

Report how many predictions remain in each video clip and how often those predictions fail. This repository contains a general CSV reporter, the code for two post hoc analyses, synthetic examples, and eight derived tables accompanying the manuscript *Recurrent clip-level failures in video polyp segmentation* by Tian Xia, Zhenqing Wang, and Liping Sun.

## Run a small example

The general reporter uses Python 3.11 or later and the standard library:

```sh
python generic_report.py --input generic_example/inputs.csv --output-dir results/generic_demo --expected-run repeat-red --expected-run repeat-blue --synthetic-demo
python -m unittest test_generic_report -v
```

The example is entirely synthetic. The command writes `per_clip.csv`, `summary.json`, and `receipt.json`; use a new or empty output directory. [GENERIC_REPORT.md](GENERIC_REPORT.md) describes the input columns, count definitions, and missing-run checks.

## Run the paired-analysis demonstration

The analysis engine requires NumPy and Matplotlib. Install the recorded dependencies in a separate Python environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/make_demo.py
python run.py --input-dir examples/synthetic_inputs --config examples/demo_protocol.json --output-dir results/paired_demo --allow-demo
```

The recorded dependency pins are NumPy 2.4.6 and Matplotlib 3.11.1. A separate compatibility check passed with Python 3.13.2, NumPy 2.4.3, and Matplotlib 3.10.8; this was not a rebuild in the pinned environment.

`make_demo.py` generates synthetic inputs beside the script and rewrites `examples/demo_protocol.json`. The synthetic flag is carried into the results and figures. These outputs demonstrate software behavior; they are not experimental results.

## Analyze supplied study-format measurements

```sh
python run.py --input-dir my_measurements --config protocol.json --output-dir results/paired_analysis
```

This entry point replays SUN-SEG selection thresholds and calculates paired before/after clip risks. It also fits PolypGen selection thresholds in both directions of a fixed sequence split. It expects:

```text
my_measurements/
  sun/*.json
  pg/*.json
  reference/sun_summary.json
  reference/sun_retained.csv
```

Each dataset must supply the `uac_base` and `pranet_plain` model identifiers with seeds 44–51. SUN-SEG JSON records use top-level `family`, `seed`, and `per_frame`; PolypGen uses `summary.family`, `summary.seed`, and `frames`. Each frame record contains clip/frame identity, Dice, maximum foreground probability, ground-truth/predicted foreground counts, and the empty-mask case. The clip field is `clip` for SUN-SEG and `clip_id` for PolypGen. The generated synthetic files demonstrate both formats. SUN-SEG references provide each model group's recorded threshold under `K6.tau` and the corresponding `baseline` retained-count rows.

The fixed protocol expects 29,592 frames and 173 SUN-SEG clips per run, and 2,225 frames and 23 PolypGen sequences per run. The PolypGen analysis keeps ground-truth-positive frames from 21 sequences. It checks frame identities, ground-truth consistency, missing or extra runs, duplicate keys, invalid values, and replayed counts. Real per-frame measurements and reference inputs are not included here.

## Interpret the outputs

A frame fails at Dice below 0.3. The engine selects predictions using `1 - max_prob`, a calibration target of 0.1, and a descriptive clip-risk criterion of 0.15. PolypGen results include both calibration/evaluation directions and a complete-tie-group sensitivity analysis. Each sequence enters the two-direction summary once, using its direction-specific threshold. That summary does not represent one fitted threshold.

The paired and retained-risk extensions are post hoc descriptive analyses. Bootstrap intervals resample evaluation clips while keeping trained runs and thresholds fixed. They describe uncertainty conditional on those runs and thresholds; they do not include refitting uncertainty. The original SUN-SEG clip split shares source-polyp groups across calibration and evaluation, which these clip intervals do not model. The 0.15 criterion is an analysis choice, not a clinical safety standard. Clips with zero retention remain listed with undefined risk.

The engine writes per-clip CSVs, summary JSON, input checks, and source/output hashes. It also renders the paired-risk plots as SVG, PDF, and 600 dpi PNG. Generated reports record local input paths; review generated metadata before sharing your own outputs.

## What is included

- `generic_report.py`: report supplied selections and the minimum error count at each clip's retained count; no threshold fitting.
- `h2_eval/` and `run.py`: the paired SUN-SEG and PolypGen retained-risk calculations.
- `data/`: eight derived manuscript tables. [Data notes](data/README.md) explain their scope and columns.
- Tests and synthetic examples.

The `h2_eval` package name and `h2_posthoc_protocol` schema are retained for compatibility. The derived fixed-budget and source-group tables are included, but the scripts for every manuscript experiment are not part of this subset. This repository does not contain original images, masks, model weights, private frame receipts, or all materials needed to reconstruct every experiment. No acceptance or publication status is implied.

## Citation and use

Author and title information is provided in [CITATION.cff](CITATION.cff). No new software or data license is assigned in this repository. Public availability does not replace permission for reuse. The original dataset providers' terms continue to apply to their data; those datasets are not redistributed here.
