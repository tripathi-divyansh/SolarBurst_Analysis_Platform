"""Controlled detection checks, not real-XSM sensitivity/false-alarm validation."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from xsm_burst.synthetic import synthetic_observation
from xsm_burst.pipeline import analyze
from xsm_burst.evaluation import match_events
from xsm_burst.schema import PipelineConfig
from xsm_burst.detection import estimate_background,proposals

recovered=0;total=0;extra=0
for seed in range(100,116):
    lc,truth=synthetic_observation(seed)
    out=analyze(lc,group_id=str(seed));truth['group_id']=str(seed)
    match=match_events(out['catalog'],truth,35)
    recovered+=match['tp'];total+=len(truth);extra+=match['fp']
# Independent curve shapes and seeds: a trend with no injected events.
null_counts=[]
for seed in range(20):
    rng=np.random.default_rng(2000+seed);t=np.arange(800.)+.5
    y=800+0.04*t+30*((t-400)/400)**2+rng.normal(0,28,len(t))
    err=np.full(len(t),28.)
    base,noise,_=estimate_background(t,y,err,301)
    found,_=proposals(t,y,err,base,noise,PipelineConfig())
    null_counts.append(len(found))
report={'controlled_injected_events':total,'recovered_within_35s':recovered,
        'other_proposals_in_signal_controls':extra,
        'null_trials':len(null_counts),'null_seconds':16000,
        'null_proposals':sum(null_counts),'null_counts':null_counts,
        'scope':'Software/simulation checks only; not real-data classification accuracy. Signal controls include known simulated artifacts.'}
print(json.dumps(report,indent=2))
