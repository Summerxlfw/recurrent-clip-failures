"""Generate synthetic records only. These are never study evidence."""
import csv, json
from pathlib import Path

root=Path(__file__).parent/'synthetic_inputs'
for name in ('sun','pg','reference'):(root/name).mkdir(parents=True,exist_ok=True)
families=['uac_base','pranet_plain'];seeds=[44,45]
for family in families:
    for seed in seeds:
        sun=[];pg=[]
        for i in range(6):
            # Even IDs calibrate; odd IDs evaluate. One error and two successes per side.
            dice=0.0 if i in (0,1) else 0.8
            row={'clip':f'clip{i}','frame':'f1','dice':dice,'max_prob':0.8,'gt_pixels':10,
                 'predicted_pixels':10,'case':'both_nonempty'}
            sun.append(row)
            pgrow=dict(row);pgrow['clip_id']=pgrow.pop('clip')
            if i>=4:pgrow.update(dice=1.0,gt_pixels=0,predicted_pixels=0,case='both_empty')
            pg.append(pgrow)
        (root/'sun'/f'{family}_{seed}.json').write_text(json.dumps({'family':family,'seed':seed,'per_frame':sun}))
        (root/'pg'/f'{family}_{seed}.json').write_text(json.dumps({'summary':{'family':family,'seed':seed},'frames':pg}))
summary={h:{'K6':{'tau':.4,'retained_crash_rate':1/3}} for h in families+['community_pooled']}
(root/'reference'/'sun_summary.json').write_text(json.dumps(summary))
with (root/'reference'/'sun_retained.csv').open('w',newline='') as f:
    fields=['host','arm','clip','retained_count','retained_crashes','total_count','retained_risk','compliant','tau']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for host in families+['community_pooled']:
        n=4 if host=='community_pooled' else 2
        for i in (1,3,5):w.writerow(dict(host=host,arm='baseline',clip=f'clip{i}',retained_count=n,
            retained_crashes=n if i==1 else 0,total_count=n,retained_risk=1.0 if i==1 else 0.0,compliant=int(i!=1),tau=.4))
config={'schema':'h2_posthoc_protocol/1','synthetic_demo':True,'families':families,'seeds':seeds,
        'expected':{'sun':{'frames_per_run':6,'clips_per_run':6},'pg':{'frames_per_run':6,'clips_per_run':6,'positive_clips':4}},
        'failure_threshold':.3,'target_risk':.1,'clip_criterion':.15,'reference_failure_rate':.5,
        'replay_tolerance':1e-12,'bootstrap_replicates':40,'bootstrap_seed':20260913}
(Path(__file__).parent/'demo_protocol.json').write_text(json.dumps(config,indent=2)+'\n')
print('Synthetic-only input directory:',root)
