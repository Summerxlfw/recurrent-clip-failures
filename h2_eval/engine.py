"""CPU-only post-hoc analysis. No model fitting or threshold search on evaluation data."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

class ValidationError(ValueError):
    pass

def require(test, message):
    if not test:
        raise ValidationError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strict_json(path):
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            require(k not in obj, f'duplicate JSON key {k}: {path}')
            obj[k] = v
        return obj
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValidationError(f'nonfinite JSON {s}')))

def validate_config(config):
    require(config.get('schema') == 'h2_posthoc_protocol/1', 'unsupported protocol schema')
    require(config['families'] and len(set(config['families'])) == len(config['families']), 'duplicate/empty families')
    require(config['seeds'] and len(set(config['seeds'])) == len(config['seeds']) and all(type(s) is int for s in config['seeds']), 'duplicate/illegal seeds')
    for key in ('failure_threshold', 'target_risk', 'clip_criterion', 'reference_failure_rate'):
        require(isinstance(config[key], (int, float)) and math.isfinite(config[key]) and 0 <= config[key] <= 1, f'illegal protocol {key}')
    require(type(config['bootstrap_replicates']) is int and config['bootstrap_replicates'] > 0, 'invalid bootstrap count')
    require(type(config['bootstrap_seed']) is int, 'invalid bootstrap seed')
    require(0 <= config['replay_tolerance'] <= 1e-12, 'replay tolerance too loose')
    for dataset in ('sun', 'pg'):
        spec = config['expected'][dataset]
        for key in ('frames_per_run', 'clips_per_run'):
            require(type(spec[key]) is int and spec[key] > 0, f'invalid expected {dataset}/{key}')
    require(0 < config['expected']['pg']['positive_clips'] <= config['expected']['pg']['clips_per_run'], 'invalid positive PG count')

def load_dataset(folder, name, config):
    expected = {(f, s) for f in config['families'] for s in config['seeds']}
    runs, paths, reference_gt = {}, [], None
    for path in sorted(Path(folder).glob('*.json')):
        d = strict_json(path)
        meta = d if name == 'sun' else d['summary']
        ident = (meta['family'], meta['seed'])
        require(ident in expected, f'unexpected run {ident}')
        require(ident not in runs, f'duplicate run {ident}')
        frames = d['per_frame'] if name == 'sun' else d['frames']
        require(isinstance(frames, list), f'{path}: frame payload is not list')
        records, gt_map = [], {}
        for row in frames:
            clip = row['clip'] if name == 'sun' else row['clip_id']
            frame = row['frame']
            require(isinstance(clip, str) and isinstance(frame, str), 'non-string identity')
            key = (clip, frame)
            require(key not in gt_map, f'duplicate frame {ident}/{key}')
            for field in ('dice', 'max_prob'):
                value = row[field]
                require(isinstance(value, (float, int)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= 1, f'illegal {field} {ident}/{key}')
            gt, pred = row['gt_pixels'], row['predicted_pixels']
            require(type(gt) is int and gt >= 0 and type(pred) is int and pred >= 0, 'illegal pixel count')
            case = 'both_empty' if gt == pred == 0 else 'one_empty' if min(gt, pred) == 0 else 'both_nonempty'
            if case == 'both_empty':
                require(row['dice'] == 1 and row['case'] == case, f'both-empty rule {key}')
            elif case == 'one_empty':
                require(row['dice'] == 0 and row['case'] == case, f'one-empty rule {key}')
            else:
                require(row['case'] not in ('both_empty', 'one_empty'), f'case/pixel conflict {key}')
            gt_map[key] = gt
            records.append((clip, frame, ident[0], ident[1], float(row['dice']), 1.0-float(row['max_prob']), gt))
        spec = config['expected'][name]
        require(len(gt_map) == spec['frames_per_run'], f'{ident}: incorrect frame count')
        require(len({k[0] for k in gt_map}) == spec['clips_per_run'], f'{ident}: incorrect clip count')
        if reference_gt is None:
            reference_gt = gt_map
        else:
            require(gt_map == reference_gt, f'frame identity or GT mismatch: {ident}')
        runs[ident] = records
        paths.append(path)
    require(set(runs) == expected, f'missing runs: {sorted(expected-set(runs))}')
    all_rows = [r for ident in sorted(runs) for r in runs[ident]]
    positive = {r[0] for r in all_rows if r[6] > 0}
    if name == 'pg':
        require(len(positive) == config['expected']['pg']['positive_clips'], 'PG positive clip count')
        all_rows = [r for r in all_rows if r[6] > 0]
    return all_rows, {'dataset': name, 'runs': len(runs), 'frames_per_run': len(reference_gt),
                      'all_clips': len({k[0] for k in reference_gt}), 'positive_clips': len(positive),
                      'analyzed_records': len(all_rows), 'identity_and_gt_aligned': True,
                      'files': [{'name': p.name, 'sha256': sha(p)} for p in paths]}

def split_clips(ids, reverse=False):
    ids = sorted(ids)
    cal, ev = (ids[1::2], ids[::2]) if reverse else (ids[::2], ids[1::2])
    require(not set(cal) & set(ev), 'calibration/evaluation overlap')
    require(cal and ev, 'empty partition')
    return cal, ev

def choose_tau(rows, target, failure_threshold, rule):
    require(rule in ('prefix', 'full_tie_groups'), 'invalid threshold rule')
    ordered = sorted(rows, key=lambda r: (r[5], r[0], r[1], r[2], r[3]))
    errors = 0
    tau, selected_n, selected_errors = None, 0, 0
    for i, row in enumerate(ordered):
        errors += row[4] < failure_threshold
        endpoint = i+1 == len(ordered) or ordered[i+1][5] != row[5]
        if (rule == 'prefix' or endpoint) and errors/(i+1) <= target:
            tau, selected_n, selected_errors = row[5], i+1, errors
    admitted = [r for r in ordered if tau is not None and r[5] <= tau]
    actual_errors = sum(r[4] < failure_threshold for r in admitted)
    return tau, {'selected_prefix_count': selected_n, 'selected_prefix_errors': selected_errors,
                 'selected_prefix_risk': selected_errors/selected_n if selected_n else None,
                 'actual_calibration_retained': len(admitted), 'actual_calibration_errors': actual_errors,
                 'actual_calibration_risk': actual_errors/len(admitted) if admitted else None,
                 'tie_expansion_count': len(admitted)-selected_n,
                 'actual_calibration_meets_target': actual_errors/len(admitted) <= target if admitted else None}

def retained_rows(rows, ids, tau, failure_threshold, criterion, **labels):
    by_clip = {c: [] for c in ids}
    for row in rows:
        if row[0] in by_clip:
            by_clip[row[0]].append(row)
    out = []
    for clip in sorted(ids):
        raw = by_clip[clip]
        kept = [r for r in raw if tau is not None and r[5] <= tau]
        errors = sum(r[4] < failure_threshold for r in kept)
        before = sum(r[4] < failure_threshold for r in raw)
        risk = errors/len(kept) if kept else None
        out.append(dict(labels, clip=clip, tau=tau, total_count=len(raw), retained_count=len(kept),
                        retained_crashes=errors, retained_risk=risk,
                        status='retained' if kept else 'zero_retained',
                        compliant=risk <= criterion if risk is not None else None,
                        retention=len(kept)/len(raw) if raw else None,
                        before_crashes=before, before_risk=before/len(raw) if raw else None))
    return out

def aggregate(rows):
    n = sum(r['retained_count'] for r in rows)
    e = sum(r['retained_crashes'] for r in rows)
    admitted = [r for r in rows if r['retained_count'] > 0]
    compliant = sum(bool(r['compliant']) for r in admitted)
    total = sum(r['total_count'] for r in rows)
    return {'clips': len(rows), 'admitted_clips': len(admitted), 'zero_retained_clips': len(rows)-len(admitted),
            'total_count': total, 'retained_count': n, 'retained_crashes': e,
            'pooled_risk': e/n if n else None, 'retention': n/total if total else None,
            'compliant_clips': compliant, 'compliance_fraction': compliant/len(admitted) if admitted else None}

def ranks(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind='stable')
    rank = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i+1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank[order[i:j]] = (i+j-1)/2
        i = j
    return rank

def spearman(x, y):
    if len(x) < 2:
        return None
    a, b = ranks(x), ranks(y)
    a, b = a-a.mean(), b-b.mean()
    den = float(np.sqrt(np.dot(a, a)*np.dot(b, b)))
    return float(np.dot(a, b)/den) if den else None

def percentile(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None

def bootstrap(rows, config, stratified=False, correlation=False):
    rng = np.random.default_rng(config['bootstrap_seed'])
    groups = {}
    for row in rows:
        groups.setdefault(row.get('fold', 'all') if stratified else 'all', []).append(row)
    risks, compliance, correlations = [], [], []
    for _ in range(config['bootstrap_replicates']):
        sampled = [group[i] for group in groups.values() for i in rng.integers(0, len(group), len(group))]
        result = aggregate(sampled)
        if result['pooled_risk'] is not None:
            risks.append(result['pooled_risk'])
            compliance.append(result['compliance_fraction'])
        if correlation:
            kept = [r for r in sampled if r['retained_risk'] is not None]
            value = spearman([r['before_risk'] for r in kept], [r['retained_risk'] for r in kept])
            if value is not None:
                correlations.append(value)
    result = {'replicates': config['bootstrap_replicates'], 'seed': config['bootstrap_seed'],
              'unit': 'evaluation clip', 'stratified_by_fold': stratified,
              'interpretation': 'conditional on trained runs and fixed calibration thresholds; descriptive',
              'pooled_risk_ci95': percentile(risks), 'compliance_fraction_ci95': percentile(compliance),
              'valid_risk_replicates': len(risks), 'valid_compliance_replicates': len(compliance),
              'zero_admitted_replicates': config['bootstrap_replicates']-len(risks)}
    if correlation:
        result.update(spearman_ci95=percentile(correlations), valid_spearman_replicates=len(correlations), undefined_spearman_replicates=config['bootstrap_replicates']-len(correlations))
    return result

def verify_sun(rows, path, summary, host, tolerance):
    with Path(path).open() as handle:
        refs = [r for r in csv.DictReader(handle) if r['host'] == host and r['arm'] == 'baseline']
    require(len(refs) == len(rows), f'SUN reference row count {host}')
    ref = {r['clip']: r for r in refs}
    require(len(ref) == len(refs) and set(ref) == {r['clip'] for r in rows}, 'SUN reference clips')
    for row in rows:
        r = ref[row['clip']]
        for k in ('retained_count', 'retained_crashes', 'total_count'):
            require(row[k] == int(r[k]), f'SUN replay {host}/{row["clip"]}/{k}: {row[k]} != {r[k]}')
        for k in ('tau', 'retained_risk'):
            if row[k] is None:
                require(r[k] in ('', 'null', 'None'), f'SUN replay missing {host}/{row["clip"]}/{k}')
            else:
                require(abs(row[k]-float(r[k])) <= tolerance, f'SUN replay {host}/{row["clip"]}/{k}')
    actual = aggregate(rows)['pooled_risk']
    expected = summary[host]['K6']['retained_crash_rate']
    require(actual is None and expected is None or actual is not None and expected is not None and abs(actual-expected) <= tolerance, f'SUN historical pooled risk {host}')
    return {'status': 'PASS', 'rows': len(rows), 'count_tolerance': 0, 'risk_tolerance': tolerance,
            'pooled_risk': actual, 'tau': summary[host]['K6']['tau']}

def analyze_sun(rows, config, ref_dir):
    historical = strict_json(ref_dir/'sun_summary.json')
    ids = sorted({r[0] for r in rows})
    cal, ev = split_clips(ids)
    threshold, criterion = config['failure_threshold'], config['clip_criterion']
    combined = retained_rows(rows, ids, None, threshold, criterion)
    reference = {r['clip'] for r in combined if r['before_risk'] >= config['reference_failure_rate']}
    results, all_output = {}, []
    for host in config['families'] + ['community_pooled']:
        subset = rows if host == 'community_pooled' else [r for r in rows if r[2] == host]
        tau = historical[host]['K6']['tau']
        out = retained_rows(subset, ev, tau, threshold, criterion, host=host, arm='baseline')
        replay = verify_sun(out, ref_dir/'sun_retained.csv', historical, host, config['replay_tolerance'])
        for r in out:
            r['reference_failure'] = r['clip'] in reference
            r['after_high_risk'] = r['retained_risk'] > criterion if r['retained_risk'] is not None else None
        valid = [r for r in out if r['retained_risk'] is not None]
        results[host] = dict(aggregate(out), tau=tau, replay=replay,
                            spearman=spearman([r['before_risk'] for r in valid], [r['retained_risk'] for r in valid]),
                            spearman_n=len(valid), reference_failure_clips=sum(r['reference_failure'] for r in out),
                            after_high_risk_clips=sum(r['after_high_risk'] is True for r in out),
                            intersection_clips=sum(r['reference_failure'] and r['after_high_risk'] is True for r in out),
                            reference_zero_retained=sum(r['reference_failure'] and r['retained_count']==0 for r in out),
                            uncertainty=bootstrap(out, config, correlation=True))
        group = results[host]
        group['intersection_fraction_of_reference'] = group['intersection_clips']/group['reference_failure_clips'] if group['reference_failure_clips'] else None
        group['intersection_fraction_of_after_high_risk'] = group['intersection_clips']/group['after_high_risk_clips'] if group['after_high_risk_clips'] else None
        all_output.extend(out)
    return {'calibration_clips': cal, 'evaluation_clips': ev,
            'reference_definition': 'combined 16-run pre-selection clip failure rate >= 0.5',
            'reference_set_all_clips': sorted(reference), 'groups': results}, all_output

def analyze_pg(rows, config):
    ids = sorted({r[0] for r in rows})
    results, all_output = {}, []
    for host in config['families'] + ['community_pooled']:
        subset = rows if host == 'community_pooled' else [r for r in rows if r[2] == host]
        host_result = {}
        for rule in ('prefix', 'full_tie_groups'):
            folds, cross_rows = {}, []
            for reverse in (False, True):
                fold = 'reverse' if reverse else 'primary'
                cal, ev = split_clips(ids, reverse)
                calibration = [r for r in subset if r[0] in set(cal)]
                tau, bookkeeping = choose_tau(calibration, config['target_risk'], config['failure_threshold'], rule)
                out = retained_rows(subset, ev, tau, config['failure_threshold'], config['clip_criterion'], host=host, rule=rule, fold=fold)
                folds[fold] = dict(aggregate(out), tau=tau, calibration_clips=cal, evaluation_clips=ev,
                                  calibration=bookkeeping, uncertainty=bootstrap(out, config))
                cross_rows.extend(out)
            require(len(cross_rows) == len(ids) and len({r['clip'] for r in cross_rows}) == len(ids), 'cross-fold duplication')
            host_result[rule] = {'folds': folds, 'crossfold': dict(aggregate(cross_rows),
                threshold_collection={f: d['tau'] for f, d in folds.items()},
                interpretation='each sequence evaluated once using its opposite-fold threshold; no single-threshold or calibration guarantee',
                uncertainty=bootstrap(cross_rows, config, stratified=True))}
            all_output.extend(cross_rows)
        results[host] = host_result
    return {'positive_clips': ids, 'groups': results}, all_output

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-demo', action='store_true')
    args = parser.parse_args()
    cfg = strict_json(args.config)
    validate_config(cfg)
    require(not cfg.get('synthetic_demo', False) or args.allow_demo, 'synthetic demo requires --allow-demo')
    require(args.output_dir.resolve() != args.input_dir.resolve(), 'input/output must differ')
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), 'output directory must be empty')
    started = datetime.now(timezone.utc).isoformat()
    sun, sun_validation = load_dataset(args.input_dir/'sun', 'sun', cfg)
    pg, pg_validation = load_dataset(args.input_dir/'pg', 'pg', cfg)
    sun_result, sun_csv = analyze_sun(sun, cfg, args.input_dir/'reference')
    pg_result, pg_csv = analyze_pg(pg, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/'summary.json', {'schema': 'h2_cpu_extension/1', 'analysis_status': 'post_hoc_descriptive',
                                               'synthetic_demo': cfg.get('synthetic_demo', False), 'sun': sun_result, 'pg': pg_result})
    write_csv(args.output_dir/'sun_before_after.csv', sun_csv)
    write_csv(args.output_dir/'pg_retained.csv', pg_csv)
    write_json(args.output_dir/'input_validation.json', {'status': 'PASS', 'sun': sun_validation, 'pg': pg_validation})
    try:
        from .plots import render_all
    except ModuleNotFoundError as exc:
        if exc.name != 'h2_eval.plots':
            raise
    else:
        render_all(args.output_dir)
    inputs = {str(p.relative_to(args.input_dir)): sha(p) for p in sorted(args.input_dir.rglob('*')) if p.is_file()}
    code = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    outputs = {p.name: sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    write_json(args.output_dir/'run_receipt.json', {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'numpy': np.__version__, 'config_sha256': sha(args.config),
            'code_sha256': code, 'input_sha256': inputs, 'output_sha256': outputs,
            'no_training': True, 'no_network': True, 'bootstrap_seed': cfg['bootstrap_seed']})
    print(json.dumps({'status': 'PASS', 'output_dir': str(args.output_dir), 'synthetic_demo': cfg.get('synthetic_demo', False)}))
