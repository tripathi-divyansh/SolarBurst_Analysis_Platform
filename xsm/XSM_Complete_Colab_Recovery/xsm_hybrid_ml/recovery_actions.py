"""Recovery operations; run with the isolated recovery Python, not notebook Python."""
import sys, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import log_loss
from xsm_burst.features import FEATURES
from xsm_classifier_demo import run_experiment, _event_metrics, _numeric, score_real_batch
RESULTS=ROOT/'runs/synthetic_classifier_v1'
SPLITS={'20260716':'train','20260913':'train','20260916':'train','20260917':'train',
        '20260919':'validation','20260920':'test'}

def baseline():
    frame=pd.read_csv(ROOT/'data/shared_tables/training_candidates.csv',dtype={'group_id':str})
    frame['split']=frame.group_id.map(SPLITS)
    if frame.split.isna().any():
        raise ValueError('Unassigned observation dates: define their split before using them.')
    x=_numeric(frame,FEATURES)
    train=frame.split.eq('train')
    columns=x.columns[x.loc[train].notna().any()].tolist()
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),
        IsolationForest(n_estimators=300,contamination='auto',random_state=42,n_jobs=2))
    model.fit(x.loc[train,columns])
    threshold=float(np.quantile(-model.score_samples(x.loc[train,columns]),.95))
    return frame,model,columns,threshold

def restore():
    frame,model,columns,threshold=baseline()
    dest=ROOT/'runs/recovered_baseline'
    dest.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(frame=frame,model=model,columns=columns,threshold=threshold),dest/'isolation_forest.joblib')
    frame.to_csv(dest/'candidates_with_fixed_splits.csv',index=False)
    print('Restored candidate counts:',frame.groupby('split').size().to_dict())
    print('Isolation Forest training-only threshold:',threshold)
    # These are bundled model files from this recovery package, not arbitrary uploaded pickles.
    bundle=joblib.load(RESULTS/'xgboost.joblib')
    model=bundle['model']
    history=pd.read_csv(RESULTS/'xgboost_loss.csv')
    minimum=int(history.loc[history.validation_log_loss.idxmin(),'round'])
    assert minimum==model.best_iteration+1
    assert len(history)-minimum==model.get_params()['early_stopping_rounds']
    print(f'XGBoost: selected round {minimum}; stopped at {len(history)}; patience 30 confirmed.')
    print('Saved RF and XGBoost models are available. No classifier retraining was needed.')
    print('Labels on real data remain unchanged. Simulation results are not real-XSM accuracy.')

def evaluate():
    report=json.loads((RESULTS/'report.json').read_text())
    protocol=json.loads((RESULTS/'protocol.json').read_text())
    candidates=pd.read_csv(RESULTS/'test_predictions.csv')
    truth=pd.read_csv(RESULTS/'test/injected_truth.csv')
    observations=pd.read_csv(RESULTS/'test/observations.csv')
    selections={'Extraction only':np.ones(len(candidates),dtype=bool),
      'Isolation Forest':candidates.isolation_forest_score>=protocol['isolation_threshold'],
      'Random Forest':candidates.random_forest_score>=protocol['supervised_threshold'],
      'XGBoost':candidates.xgboost_score>=protocol['supervised_threshold']}
    rows=[]; strengths=[]
    for name,mask in selections.items():
        metrics,bands=_event_metrics(candidates.loc[mask],truth,observations)
        rows.append(dict(model=name,**metrics))
        strengths.extend(dict(model=name,**r) for r in bands)
    target=ROOT/'runs/recovery_evaluation'
    target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'evaluation.csv',index=False)
    pd.DataFrame(strengths).to_csv(target/'recovery_by_strength.csv',index=False)
    print('SIMULATION ONLY. Re-display of the existing test; NOT a new independent test.')
    print('Original validation selection:',report['selected_model'])
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nBurst recovery by strength:')
    print(pd.DataFrame(strengths).to_string(index=False))
    print('\nOnly two weak bursts occur in this saved test; weak-burst performance is uncertain.')

def regularization():
    from xgboost import XGBClassifier
    bundle=joblib.load(RESULTS/'xgboost.joblib')
    train=pd.read_csv(RESULTS/'train/candidates.csv')
    val=pd.read_csv(RESULTS/'validation/candidates.csv')
    truth=pd.read_csv(RESULTS/'validation/injected_truth.csv')
    observations=pd.read_csv(RESULTS/'validation/observations.csv')
    cols=bundle['metadata']['feature_columns']
    xt=bundle['imputer'].transform(_numeric(train,cols))
    xv=bundle['imputer'].transform(_numeric(val,cols))
    variants={'Original':None,
      'Mild':dict(max_depth=3,min_child_weight=5,reg_alpha=.3,reg_lambda=10),
      'Shallow':dict(max_depth=2,min_child_weight=3,reg_alpha=.1,reg_lambda=10),
      'Strong':dict(max_depth=2,min_child_weight=5,reg_alpha=.5,reg_lambda=10)}
    rows=[]
    for name,params in variants.items():
        if params is None:model=bundle['model']
        else:
            model=XGBClassifier(n_estimators=800,learning_rate=.03,subsample=.8,
                colsample_bytree=.8,objective='binary:logistic',eval_metric='logloss',
                early_stopping_rounds=30,tree_method='hist',n_jobs=2,random_state=42,**params)
            model.fit(xt,train.label.astype(int),
                eval_set=[(xt,train.label.astype(int)),(xv,val.label.astype(int))],verbose=False)
        pt=model.predict_proba(xt)[:,1]; pv=model.predict_proba(xv)[:,1]
        metrics,_=_event_metrics(val.loc[pv>=.5],truth,observations)
        rows.append(dict(model=name,best_round=model.best_iteration+1,
            train_loss=log_loss(train.label,pt,labels=[0,1]),
            validation_loss=log_loss(val.label,pv,labels=[0,1]),**metrics))
    target=ROOT/'runs/regularization_validation'
    target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Original selected model preserved. Test data were not read by this comparison.')

def retrain():
    from datetime import datetime,timezone
    frame,model,columns,threshold=baseline()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    result=run_experiment(frame,model,columns,threshold,ROOT/'runs'/('reproduction_'+stamp))
    print('REPRODUCTION using original seeds: not a new independent test.')
    print('Original selected model preserved. New output:',result)

def score_july():
    from datetime import datetime,timezone
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'runs'/('july_exploratory_scores_'+stamp+'.csv')
    score_real_batch(RESULTS/'selected_model.joblib',ROOT/'results/july16_v02',output,trusted=True)
    print('Unvalidated real-data review scores saved to:',output)
    print('These are review priorities, not confirmed events or calibrated real probabilities.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['restore','evaluate','regularization','retrain','score-july'])
    action=parser.parse_args().action
    {'restore':restore,'evaluate':evaluate,'regularization':regularization,
     'retrain':retrain,'score-july':score_july}[action]()
