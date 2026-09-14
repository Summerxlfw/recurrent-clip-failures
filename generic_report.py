#!/usr/bin/env python3
"""Describe selected frame-run outputs without fitting a selector or claiming control."""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path

KEY = ('dataset', 'model', 'run', 'clip', 'frame')
FIELDS = ('dataset', 'model', 'clip', 'run_count', 'total_count', 'failure_count',
          'retained_count', 'retained_failure_count', 'risk_before', 'risk_after',
          'retention', 'oracle_min_retained_failures', 'fixed_clip_count_risk_lower_bound',
          'status')


def boolean(value, field, line):
    if value is None or value.strip().lower() not in {'true', 'false', '1', '0'}:
        raise ValueError(f'row {line}: {field} must be true/false or 1/0')
    return value.strip().lower() in {'true', '1'}


def report_csv(path, dice_threshold=None, expected_runs=None, synthetic_demo=False):
    if dice_threshold is not None and (not math.isfinite(dice_threshold) or not 0 <= dice_threshold <= 1):
        raise ValueError('Dice threshold must be finite and in [0, 1]')
    if expected_runs is not None:
        if not expected_runs or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in expected_runs):
            raise ValueError('expected runs must be nonempty, whitespace-free labels')
        if len(set(expected_runs)) != len(expected_runs):
            raise ValueError('duplicate expected run labels')
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    columns = reader.fieldnames
    required = set(KEY) | {'retained'} | ({'dice'} if dice_threshold is not None else {'failure'})
    if columns is None or len(columns) != len(set(columns)) or any(not x for x in columns):
        raise ValueError('CSV needs unique, nonempty column names')
    if not required.issubset(columns):
        raise ValueError(f'missing columns: {sorted(required - set(columns))}')
    seen = set()
    groups = defaultdict(list)
    frame_runs = defaultdict(set)
    for row in reader:
        line = reader.line_num
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'row {line}: wrong number of fields')
        identity = tuple(row[k].strip() for k in KEY)
        if not all(identity):
            raise ValueError(f'row {line}: empty identity')
        if identity in seen:
            raise ValueError(f'row {line}: duplicate frame-run key {identity}')
        seen.add(identity)
        dataset, model, run, clip, frame = identity
        retained = boolean(row['retained'], 'retained', line)
        if dice_threshold is None:
            failure = boolean(row['failure'], 'failure', line)
        else:
            try:
                dice = float(row['dice'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'row {line}: invalid Dice') from exc
            if not math.isfinite(dice) or not 0 <= dice <= 1:
                raise ValueError(f'row {line}: Dice must be finite and in [0, 1]')
            failure = dice < dice_threshold
            if 'failure' in columns and boolean(row['failure'], 'failure', line) != failure:
                raise ValueError(f'row {line}: failure disagrees with explicit Dice threshold')
        groups[(dataset, model, clip)].append((run, failure, retained))
        frame_runs[(dataset, model, clip, frame)].add(run)
    if not seen:
        raise ValueError('CSV contains no observations')
    if expected_runs is not None:
        expected = set(expected_runs)
        for identity, actual in sorted(frame_runs.items()):
            if actual != expected:
                raise ValueError(f'run coverage mismatch for {identity}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    clip_rows = []
    summaries = defaultdict(list)
    for (dataset, model, clip), observations in sorted(groups.items()):
        n = len(observations)
        b = sum(x[1] for x in observations)
        a = sum(x[2] for x in observations)
        e = sum(x[1] and x[2] for x in observations)
        oracle_errors = max(0, b - (n - a))
        row = dict(zip(FIELDS, (dataset, model, clip, len({x[0] for x in observations}),
                               n, b, a, e, b/n, e/a if a else None, a/n,
                               oracle_errors, oracle_errors/a if a else None,
                               'retained' if a else 'zero_retained')))
        clip_rows.append(row)
        summaries[(dataset, model)].append(row)
    result_groups = []
    for (dataset, model), rows in sorted(summaries.items()):
        totals = {field: sum(r[field] for r in rows) for field in
                  ('total_count', 'failure_count', 'retained_count', 'retained_failure_count', 'oracle_min_retained_failures')}
        n, a = totals['total_count'], totals['retained_count']
        result_groups.append(dict(dataset=dataset, model=model, clip_count=len(rows),
            admitted_clip_count=sum(r['retained_count'] > 0 for r in rows),
            zero_retained_clip_count=sum(r['retained_count'] == 0 for r in rows),
            **totals, pooled_risk_before=totals['failure_count']/n,
            pooled_risk_after=totals['retained_failure_count']/a if a else None,
            retention=a/n,
            pooled_lower_bound_given_each_clip_retained_count=totals['oracle_min_retained_failures']/a if a else None))
    summary = dict(schema_version=1, unit='frame-run observation', synthetic_demo=synthetic_demo,
        scope='Descriptive report of supplied selected outputs; no selector fitting, calibration guarantee, or scientific validation.',
        source=dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(data).hexdigest(), rows=len(seen)),
        failure_rule=('provided boolean' if dice_threshold is None else f'dice < {dice_threshold!r}'),
        dice_threshold=dice_threshold, expected_runs=expected_runs,
        expected_run_check='Every observed dataset/model/clip/frame must appear in exactly all expected runs; absent entire clips/frames require a separate manifest.' if expected_runs is not None else 'Not requested; missing runs cannot be inferred.',
        groups=result_groups)
    return clip_rows, summary


def write_report(output_dir, rows, summary):
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('output directory must be absent or empty')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'per_clip.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    receipt = dict(source_sha256=summary['source']['sha256'],
                   reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   output_sha256=outputs, synthetic_demo=summary['synthetic_demo'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dice-threshold', type=float, help='Explicit rule: failure iff dice < threshold; otherwise use failure boolean.')
    parser.add_argument('--expected-run', action='append', dest='expected_runs', help='Repeat once per expected run; applies to every observed dataset/model/clip/frame.')
    parser.add_argument('--synthetic-demo', action='store_true', help='Mark synthetic demonstration outputs, never scientific evidence.')
    args = parser.parse_args()
    try:
        rows, summary = report_csv(args.input, args.dice_threshold, args.expected_runs, args.synthetic_demo)
        write_report(args.output_dir, rows, summary)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(f"{summary['source']['rows']} frame-run observations; {len(rows)} dataset/model/clip rows. No calibration or prospective guarantee.")
    for group in summary['groups']:
        risk = group['pooled_risk_after']
        print(f"{group['dataset']} / {group['model']}: retained={group['retained_count']}/{group['total_count']}; pooled risk={risk}; zero-retained clips={group['zero_retained_clip_count']}")


if __name__ == '__main__':
    main()
