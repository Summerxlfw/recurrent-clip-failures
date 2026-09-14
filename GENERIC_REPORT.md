# Report selected predictions by clip

Use `generic_report.py` to summarize selection outcomes from any dataset/model/run identifiers. It accepts a UTF-8 CSV and reports each clip's errors, retained count, risk, and retained-count lower bound. It uses Python's standard library and does not fit a selector.

## Input

Required columns are `dataset,model,run,clip,frame,retained` plus either `failure` or `dice`.

- Without `--dice-threshold`, supply `failure` as a Boolean.
- With `--dice-threshold 0.3`, failure means strictly `dice < 0.3`. Dice must be finite and between 0 and 1; a supplied `failure` column must agree.
- `failure` and `retained` accept `true/false` or `1/0`, ignoring letter case and surrounding whitespace.
- Identities are nonempty strings, trimmed before duplicate detection. The unique key is `(dataset, model, run, clip, frame)`.
- One row is one frame–run observation. Two predictions of the same image from different training runs count as two observations.

```sh
python generic_report.py --input generic_example/inputs.csv --output-dir results/generic_demo_dice --dice-threshold 0.3 --expected-run repeat-red --expected-run repeat-blue --synthetic-demo
```

For your own measurements, replace the input path and omit `--synthetic-demo`. `--expected-run` is optional and repeatable; when given, every observed dataset/model/clip/frame must contain exactly the listed runs. This common run list applies to every model in the input. Entirely absent clips or frames require a separate expected-frame list; the reporter cannot infer them.

Malformed CSV rows, duplicate keys or headers, missing values, illegal Booleans, inconsistent Dice/failure values, and empty inputs stop before writing a report. Extra unused columns are not interpreted. Inputs are never modified, and the output directory must be new or empty.

## Outputs

`per_clip.csv` pools runs within each `(dataset, model, clip)`. Let N be all supplied observations, B their failures, A retained observations, and E retained failures.

| Columns | Meaning |
|---|---|
| `total_count`, `failure_count` | N and B |
| `retained_count`, `retained_failure_count` | A and E |
| `risk_before`, `risk_after`, `retention` | B/N, E/A, A/N |
| `oracle_min_retained_failures` | max(0, B − (N − A)) |
| `fixed_clip_count_risk_lower_bound` | max(0, B − (N − A))/A |
| `status` | `retained` or `zero_retained` |

The bound is the fewest failures possible while keeping exactly A observations from that clip's existing N predictions. It is a counting bound, not a label-free selection algorithm. Actual risk can exceed it. Changing the number kept in a clip changes its bound.

Zero-retained clips remain in the CSV with blank risk/bound fields and JSON `null` values. `summary.json` provides dataset/model counts and observation-weighted pooled risks. Its pooled lower bound conditions on every clip's retained count, rather than only total retention. The reporter does not assign a clinical cutoff or calculate confidence intervals.

`receipt.json` records input, source-code, and output hashes. `summary.json` also stores the resolved local input path. Review generated metadata before sharing a report.

## Synthetic example and tests

`generic_example/inputs.csv` contains eight fabricated rows for two models, two runs, and three clips. It includes zero retention and Dice exactly equal to 0.3. It contains no study measurements.

```sh
python -m unittest test_generic_report -v
```

The tests cover an imperfect selector whose risk exceeds the count bound, zero retention, duplicates, missing runs, invalid values, explicit Dice semantics, output hashes, and overwrite protection.
