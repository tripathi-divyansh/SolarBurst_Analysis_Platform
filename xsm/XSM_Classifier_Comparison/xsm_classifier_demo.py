"""Supervised simulation experiment for the existing XSM v0.2 notebook.

Models learn simulation labels, never guessed labels on real ISRO candidates.
The real-data Isolation Forest is a frozen baseline. All model selection uses
validation simulations; test simulations are generated after selection freezes.
"""
from pathlib import Path
import json
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, average_precision_score
from sklearn.utils.validation import check_is_fitted
import joblib
from xsm_burst.schema import LightCurve, PipelineConfig
from xsm_burst.pipeline import analyze
from xsm_burst.features import FEATURES, FEATURE_VERSION
from xsm_burst.physics import fred_bin_mean
from xsm_burst.evaluation import match_events

VERSION = "xsm-synthetic-classifier-v1"
CONTEXT_FEATURES = ["ctx_peak_fraction", "ctx_roughness", "ctx_flat_top_fraction",
                    "ctx_longest_above2_s", "ctx_positive_area_over_peak_s"]
# Observable features; exclude physical fit estimates absent in real inputs.
CLASSIFIER_FEATURES = FEATURES[:22] + CONTEXT_FEATURES
EVENT_COLUMNS = ["group_id", "observed_start_s", "observed_peak_s", "observed_end_s"]
THRESHOLD = 0.5


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _concat(tables):
    nonempty = [x for x in tables if len(x)]
    return pd.concat(nonempty, ignore_index=True) if nonempty else tables[0].iloc[:0].copy()


def add_context_features(result):
    """Identical raw-context feature calculation for simulations and real data."""
    frame = result["catalog"].copy()
    for col in CONTEXT_FEATURES:
        frame[col] = np.nan
    lc = result["lightcurve"]
    background = result["baseline"]
    rate, error = lc.rates()
    segments = lc.segments()
    for index, row in frame.iterrows():
        segment = next((s for s in segments if lc.time[s[0]] <= row.observed_peak_s <= lc.time[s[-1]]), None)
        if segment is None:
            continue
        ids = segment[np.abs(lc.time[segment]-row.observed_peak_s) <= 45]
        if len(ids) < 3:
            continue
        signal = rate[ids]-background[ids]
        noise = float(np.median(error[ids]))
        if not np.isfinite(signal).all() or not np.isfinite(noise) or noise <= 0:
            continue
        positive = np.maximum(signal, 0)
        peak = max(float(positive.max()), noise)
        excess = np.maximum(signal-noise, 0)
        active = positive >= .2*peak
        run, longest = 0., 0.
        for above, width in zip(signal > 2*noise, lc.bin_width[ids]):
            run = run+width if above else 0.
            longest = max(longest, run)
        values = [
            float(excess.max()/max(excess.sum(), noise)),
            float(np.mean(np.abs(np.diff(signal)))/peak),
            float(np.sum(positive >= .8*peak)/max(1, active.sum())),
            longest,
            float(np.sum(positive*lc.bin_width[ids])/peak),
        ]
        frame.loc[index, CONTEXT_FEATURES] = values
    return frame


def _observation(seed, index, split, scales):
    rng = np.random.default_rng(seed)
    kind = ["burst", "burst", "burst", "burst", "noise", "artifacts"][index % 6]
    row = scales.iloc[int(rng.integers(len(scales)))]
    level, sigma = float(row.background), float(row.noise)
    n = 900
    t = np.arange(n, dtype=float)+.5
    rho = float(rng.uniform(0,.6)) if rng.random() < .5 else 0.
    raw = rng.normal(0,sigma*np.sqrt(1-rho*rho),n+100)
    for i in range(1,len(raw)):
        raw[i] += rho*raw[i-1]
    drift = rng.uniform(-1,1)*sigma*(t/n-.5)
    wave = rng.uniform(0,1.5)*sigma*np.sin(2*np.pi*t/rng.uniform(600,1800)+rng.uniform(0,2*np.pi))
    y = level+drift+wave+raw[100:]
    group = f"{split}_{seed}"
    truth = pd.DataFrame(columns=EVENT_COLUMNS+["strength", "shape", "strength_band"])
    if kind == "burst":
        strength = float(rng.uniform(2.5,12))
        center = float(rng.uniform(200,350))
        shape = str(rng.choice(["fred", "gaussian", "triangle", "double"]))
        def gaussian(c, width):
            return width*np.sqrt(np.pi/2)*(erf((t+.5-c)/(np.sqrt(2)*width))-erf((t-.5-c)/(np.sqrt(2)*width)))
        if shape == "fred":
            profile = fred_bin_mean(t-.5,t+.5,1,center,rng.uniform(1.5,12),rng.uniform(5,35))
        elif shape == "gaussian":
            profile = gaussian(center,rng.uniform(1.5,18))
        elif shape == "triangle":
            left, right = rng.uniform(3,25), rng.uniform(5,50)
            profile = np.where(t<center,np.clip(1+(t-center)/left,0,1),np.clip(1-(t-center)/right,0,1))
        else:
            profile = gaussian(center,rng.uniform(3,12)) + rng.uniform(.5,1)*gaussian(center+rng.uniform(10,30),rng.uniform(3,12))
        profile *= strength*sigma/profile.max()
        y += profile
        active = np.flatnonzero(profile >= .05*profile.max())
        truth = pd.DataFrame([dict(group_id=group, observed_start_s=float(t[active[0]]-.5),
                                  observed_peak_s=float(t[np.argmax(profile)]), observed_end_s=float(t[active[-1]]+.5),
                                  strength=strength, shape=shape,
                                  strength_band="Weak (2.5–4)" if strength <= 4 else "Medium (4–8)" if strength <= 8 else "Strong (8–12)")])
    artifact_count = 0 if kind == "noise" else 2 if kind == "artifacts" else int(rng.integers(0,4))
    occupied = []
    for j in range(artifact_count):
        choices = np.arange(70,n-70)
        if len(truth):
            start,end = truth.iloc[0][["observed_start_s","observed_end_s"]]
            choices = choices[(choices<start-70)|(choices>end+70)]
        for previous in occupied:
            choices = choices[np.abs(choices-previous)>50]
        pos = int(rng.choice(choices)); occupied.append(pos)
        amplitude = float(rng.uniform(3,13)*sigma)
        if j % 2 == 0:
            width = int(rng.integers(1,4))
            y[pos:pos+width] += amplitude
        else:
            box = np.zeros(n);box[pos:pos+int(rng.integers(4,22))] = amplitude
            y += gaussian_filter1d(box,float(rng.uniform(.2,1.5)))
    lc = LightCurve(t,y,np.full(n,sigma),np.ones(n),np.ones(n),np.zeros(n,int),
                    dict(quantity="count_rate",unit="count / s",epoch="relative",time_scale="relative",synthetic=True,generator_seed=seed))
    return lc, truth, dict(group_id=group,split=split,kind=kind,seed=seed,noise_correlation=rho,artifacts=artifact_count)


def _generate(split, count, seed_start, scales, directory):
    tables, truths, observations = [], [], []
    directory.mkdir()
    for i in range(count):
        lc, truth, obs = _observation(seed_start+i,i,split,scales)
        result = analyze(lc,PipelineConfig(),group_id=obs["group_id"])
        table = add_context_features(result)
        match = match_events(table,truth,35)
        labels = np.zeros(len(table),int)
        for pi, _ in match["pairs"]:
            labels[pi] = 1
        table["label"] = labels
        table["split"] = split
        table["label_origin"] = "SIMULATION_TRUTH_ONLY"
        obs.update(processed_seconds=float(result["manifest"]["processed_exposure_s"]),injected=len(truth))
        tables.append(table);truths.append(truth);observations.append(obs)
        if (i+1)%6 == 0 or i+1 == count:
            print(f"{split}: {i+1}/{count} simulated observations",flush=True)
    data, references, exposure = _concat(tables), _concat(truths), pd.DataFrame(observations)
    data.to_csv(directory/"candidates.csv",index=False)
    references.to_csv(directory/"injected_truth.csv",index=False)
    exposure.to_csv(directory/"observations.csv",index=False)
    return data,references,exposure


def _event_metrics(selected, truth, observations):
    match = match_events(selected,truth,35)
    tp,fp,fn = match["tp"],match["fp"],match["fn"]
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    null = observations[observations.injected == 0]
    null_false = int(selected.group_id.isin(null.group_id).sum())
    metrics = dict(recovered=tp,missed=fn,false_triggers=fp,event_precision=precision,event_recall=recall,
                   event_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                   null_false_triggers=null_false,
                   null_false_per_simulated_hour=null_false/(float(null.processed_seconds.sum())/3600),
                   null_simulated_hours=float(null.processed_seconds.sum())/3600)
    recovery = []
    recovered_ids = {ri for _,ri in match["pairs"]}
    for band in ["Weak (2.5–4)", "Medium (4–8)", "Strong (8–12)"]:
        ids = set(np.flatnonzero(truth.strength_band.to_numpy() == band))
        hits = len(ids & recovered_ids)
        recovery.append(dict(strength_band=band,injected=len(ids),recovered=hits,missed=len(ids)-hits,
                             recovery_percent=100*hits/len(ids) if ids else np.nan))
    return metrics,recovery


def _numeric(frame,columns):
    return frame[columns].apply(pd.to_numeric,errors="raise").replace([np.inf,-np.inf],np.nan)


def run_experiment(frame, isolation_model, isolation_columns, isolation_threshold, output_dir,
                   counts=(96,36,60), seed_base=20261009):
    """Train on 96 simulations, select on 36, then compare on 60 held-out ones.

Returns output directory. Existing outputs cannot be overwritten. More runs
with the same seeds are repetitions, not new independent evaluations.
"""
    from xgboost import XGBClassifier
    import xgboost
    check_is_fitted(isolation_model)
    if len(counts)!=3 or any(n<6 or n%6 for n in counts):
        raise ValueError("Use three observation counts, each a positive multiple of six (minimum six).")
    if not np.isfinite(isolation_threshold):
        raise ValueError("A finite frozen Isolation Forest threshold is required.")
    train_real = frame[frame.split.eq("train")].copy()
    if not len(train_real):raise ValueError("Missing original training-day candidates")
    if set(train_real.pipeline_hash)!={PipelineConfig().fingerprint()}:
        raise ValueError("Default v0.2 extraction fingerprint required")
    if set(train_real.quantity)!={"count_rate"} or set(train_real.unit)!={"count / s"}:
        raise ValueError("Count-rate inputs in count / s required")
    if set(train_real.feature_version)!={FEATURE_VERSION} or not np.allclose(train_real.cadence_s,1,atol=1e-4):
        raise ValueError("Expected v0.2 features with one-second cadence")
    if list(isolation_model.feature_names_in_)!=list(isolation_columns):
        raise ValueError("Isolation Forest feature ordering mismatch")
    scales = pd.DataFrame(dict(background=train_real.background_at_peak,
                               noise=train_real.net_peak_observed/train_real.peak_snr.replace(0,np.nan)))
    scales = scales.replace([np.inf,-np.inf],np.nan).dropna()
    scales = scales[(scales.background>0)&(scales.noise>0)]
    if not len(scales):raise ValueError("No valid training-only scale estimates")
    out = Path(output_dir)
    if out.exists():raise FileExistsError(f"Results already exist at {out}. Display the saved graphs/report; do not repeatedly tune on the test set.")
    out.mkdir(parents=True)
    protocol = dict(version=VERSION,scope="SYNTHETIC_ONLY_UNVALIDATED_ON_REAL_XSM",counts=list(counts),
                    seed_starts=dict(train=seed_base,validation=seed_base+100000,test=seed_base+200000),
                    pipeline_hash=PipelineConfig().fingerprint(),feature_version=FEATURE_VERSION,
                    feature_candidates=CLASSIFIER_FEATURES,matching_tolerance_s=35,context_half_window_s=45,
                    supervised_threshold=THRESHOLD,isolation_threshold=float(isolation_threshold),
                    selection_rule="Highest validation end-to-end event F1 at fixed threshold 0.5; ties prefer Random Forest",
                    early_stopping="XGBoost validation log loss, patience 30 boosting rounds",
                    source_scale_groups=sorted(train_real.group_id.astype(str).unique().tolist()),
                    real_training_input_sha256=hashlib.sha256(train_real.to_csv(index=False).encode()).hexdigest(),
                    versions={"numpy":np.__version__,"pandas":pd.__version__,"xgboost":xgboost.__version__,
                              **{name:importlib.metadata.version(name) for name in ["scikit-learn","scipy","astropy","PyWavelets"]}},
                    limitations=["No real candidate labels are created or modified.",
                                 "Real training candidates supply approximate background/noise scales only; supervised classifiers train on simulation truth.",
                                 "Gaussian or AR(1) noise plus smooth drift is not a validated instrument noise model.",
                                 "Only four simulated burst shape families; short bursts and artifacts can overlap in shape.",
                                 "No gaps, saturation, energy dependence, or actual housekeeping flags are simulated.",
                                 "Fixed 35-second matching tolerance; duplicates count as false triggers.",
                                 "Simulation performance and classifier scores are not calibrated real-XSM accuracy/probabilities.",
                                 "All three simulation splits share a generator family; real-world transfer is unverified."])
    _write_json(out/"protocol.json",protocol)
    scales.to_csv(out/"training_scale_pool.csv",index=False)
    train,train_truth,train_obs = _generate("train",counts[0],seed_base,scales,out/"train")
    val,val_truth,val_obs = _generate("validation",counts[1],seed_base+100000,scales,out/"validation")
    for name,part in [("train",train),("validation",val)]:
        if set(part.label)!={0,1}:raise ValueError(f"{name} simulation lacks one candidate class; inspect outputs before changing the protocol.")
    if set(train.group_id)&set(val.group_id) or set(train.source_hash)&set(val.source_hash):
        raise ValueError("Simulation leakage across partitions")
    columns = [c for c in CLASSIFIER_FEATURES if train[c].notna().any()]
    imputer = SimpleImputer(strategy="median",add_indicator=True)
    xt = imputer.fit_transform(_numeric(train,columns))
    xv = imputer.transform(_numeric(val,columns))
    rf = RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=3,max_features="sqrt",random_state=42,n_jobs=2)
    xgb = XGBClassifier(n_estimators=600,max_depth=3,learning_rate=.05,min_child_weight=3,
                        subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=5,
                        objective="binary:logistic",eval_metric="logloss",early_stopping_rounds=30,
                        tree_method="hist",n_jobs=2,random_state=42)
    print("Fitting Random Forest and XGBoost on simulation labels...",flush=True)
    rf.fit(xt,train.label.astype(int))
    xgb.fit(xt,train.label.astype(int),eval_set=[(xt,train.label.astype(int)),(xv,val.label.astype(int))],verbose=False)
    models = {"Random Forest":rf,"XGBoost":xgb}
    validation = {}
    for name,model in models.items():
        score = model.predict_proba(xv)[:,1]
        metrics,_ = _event_metrics(val.loc[score>=THRESHOLD],val_truth,val_obs)
        metrics["candidate_log_loss"] = float(log_loss(val.label,score,labels=[0,1]))
        metrics["candidate_average_precision"] = float(average_precision_score(val.label,score))
        validation[name] = metrics
    selected = max(models,key=lambda k:validation[k]["event_f1"])
    history = xgb.evals_result()
    loss = pd.DataFrame(dict(round=np.arange(len(history["validation_0"]["logloss"]))+1,
                             train_log_loss=history["validation_0"]["logloss"],
                             validation_log_loss=history["validation_1"]["logloss"]))
    loss.to_csv(out/"xgboost_loss.csv",index=False)
    prevalence = float(train.label.mean())
    constant_loss = float(log_loss(val.label,np.full(len(val),prevalence),labels=[0,1]))
    frozen = dict(selected_model=selected,validation=validation,feature_columns=columns,
                  best_xgboost_round=int(xgb.best_iteration+1),constant_validation_log_loss=constant_loss,
                  threshold=THRESHOLD,scope=protocol["scope"],protocol_version=VERSION)
    _write_json(out/"frozen_selection.json",frozen)
    compatibility = {c:str(train_real[c].iloc[0]) for c in ["pipeline_hash","feature_version","quantity","unit"]}
    for name,model in models.items():
        filename = "random_forest.joblib" if name=="Random Forest" else "xgboost.joblib"
        joblib.dump(dict(model=model,imputer=imputer,metadata={**frozen,"model_name":name,"compatibility":compatibility}),out/filename)
    joblib.dump(dict(model=models[selected],imputer=imputer,metadata={**frozen,"model_name":selected,"compatibility":compatibility}),out/"selected_model.joblib")
    xgb.save_model(out/"xgboost_model.json")
    print(f"Frozen selection: {selected}. Now generating untouched test simulations.",flush=True)
    test,truth,observations = _generate("test",counts[2],seed_base+200000,scales,out/"test")
    if set(test.group_id)&(set(train.group_id)|set(val.group_id)) or set(test.source_hash)&(set(train.source_hash)|set(val.source_hash)):
        raise ValueError("Test simulation leakage")
    xx = imputer.transform(_numeric(test,columns))
    scores = {"Isolation Forest":-isolation_model.score_samples(_numeric(test,isolation_columns)),
              **{name:model.predict_proba(xx)[:,1] for name,model in models.items()}}
    accepted = {"Extraction only":test}
    for name,score in scores.items():
        threshold = isolation_threshold if name=="Isolation Forest" else THRESHOLD
        accepted[name] = test.loc[score>=threshold]
        test[name.replace(" ","_").lower()+"_score"] = score
    test.to_csv(out/"test_predictions.csv",index=False)
    comparisons,recoveries = [],[]
    for name,candidates in accepted.items():
        metrics,by_strength = _event_metrics(candidates,truth,observations)
        comparisons.append(dict(model=name,selected_on_validation=name==selected,**metrics))
        recoveries.extend(dict(model=name,**r) for r in by_strength)
    comparison = pd.DataFrame(comparisons)
    recovery = pd.DataFrame(recoveries)
    comparison.to_csv(out/"test_comparison.csv",index=False)
    recovery.to_csv(out/"test_recovery_by_strength.csv",index=False)
    report = dict(**frozen,test_comparison=comparisons,
                  test_note="All models are reported descriptively; the winner was frozen using validation before these tests.",
                  limitations=protocol["limitations"])
    _write_json(out/"report.json",report)
    _plots(out,loss,constant_loss,xgb.best_iteration+1,comparison,recovery)
    print("\nSIMULATION RESULTS ONLY:")
    print(comparison.to_string(index=False))
    print("\nSaved:",out)
    return out


def _plots(out,loss,constant_loss,best_round,comparison,recovery):
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(loss["round"],loss.train_log_loss,label="Training simulation",color="#2463b4")
    ax.plot(loss["round"],loss.validation_log_loss,label="Validation simulation",color="#d36c20")
    ax.axvline(best_round,color="#666666",ls="--",label=f"Selected round {best_round}")
    ax.axhline(constant_loss,color="#888888",ls=":",label="Constant-prediction validation baseline")
    ax.set(title="XGBoost training vs validation loss — simulations only",xlabel="Boosting round (not epoch)",ylabel="Binary log loss (lower is better)")
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/"training_validation_loss.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    names=list(comparison.model)
    colors=["#888888","#a044a4","#2463b4","#d36c20"]
    bands=["Weak (2.5–4)","Medium (4–8)","Strong (8–12)"]
    for name,color in zip(names,colors):
        sub=recovery[recovery.model==name].set_index("strength_band").reindex(bands)
        axes[0].plot(range(3),sub.recovery_percent,marker="o",color=color,label=name)
    axes[0].set(xticks=range(3),xticklabels=bands,ylim=(-5,105),ylabel="Injected bursts recovered (%)",xlabel="Injected peak / simulated noise",title="Held-out burst recovery")
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    bars=axes[1].bar(range(4),comparison.null_false_per_simulated_hour,color=colors)
    axes[1].bar_label(bars,fmt="%.1f",padding=3)
    axes[1].set(xticks=range(4),xticklabels=["Extraction","Isolation\nForest","Random\nForest","XGBoost"],ylabel="False triggers / simulated hour",title="Zero-burst controls (noise and artifacts)")
    axes[1].set_ylim(0,max(1,float(comparison.null_false_per_simulated_hour.max())*1.25))
    n_by_band=recovery[recovery.model==names[0]].set_index("strength_band").reindex(bands).injected.tolist()
    fig.suptitle("Independent test simulations — not real-XSM accuracy",fontsize=14)
    fig.text(.5,.025,f"Injected counts by strength: {n_by_band}. Final classifier was selected on validation before testing.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(out/"heldout_comparison.png",dpi=160);plt.close(fig)


def score_real_batch(model_path,batch_run,output_csv,*,trusted=False):
    """Optional exploratory inference. Never changes real review labels."""
    if not trusted:raise ValueError("Set trusted=True only for your own model; joblib files can execute code.")
    bundle=joblib.load(model_path)
    meta=bundle["metadata"]
    if meta["protocol_version"]!=VERSION:raise ValueError("Feature implementation version mismatch")
    output_csv=Path(output_csv)
    if output_csv.exists():raise FileExistsError("Choose a new output CSV; existing scores are protected")
    rows=[]
    for manifest_path in sorted(Path(batch_run).glob("*/manifest.json")):
        directory=manifest_path.parent
        manifest=json.loads(manifest_path.read_text())
        table=pd.read_csv(directory/"candidates.csv")
        for col,value in meta["compatibility"].items():
            if col not in table or (len(table) and not table[col].astype(str).eq(value).all()):
                raise ValueError(f"Real input {col} differs from the model schema")
        with np.load(directory/"input_arrays.npz",allow_pickle=False) as data, np.load(directory/"baseline.npz",allow_pickle=False) as base:
            if not np.array_equal(data["time"],base["time_s"]):raise ValueError("Saved curve and baseline times differ")
            lc=LightCurve(*(data[k].copy() for k in ["time","value","error","bin_width","exposure","quality"]),metadata=manifest["metadata"])
            enriched=add_context_features(dict(catalog=table,lightcurve=lc,baseline=base["background"].copy()))
        if len(enriched):
            x=bundle["imputer"].transform(_numeric(enriched,meta["feature_columns"]))
            enriched["synthetic_classifier_score"]=bundle["model"].predict_proba(x)[:,1]
        else:enriched["synthetic_classifier_score"]=pd.Series(dtype=float)
        enriched["review_priority"]=np.where(enriched.synthetic_classifier_score>=meta["threshold"],"high","standard")
        enriched["model_scope"]="SYNTHETIC_ONLY_UNVALIDATED_ON_REAL_XSM"
        rows.append(enriched)
    if not rows:raise FileNotFoundError("No saved analysis directories in BATCH_RUN")
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    result=_concat(rows);result.to_csv(output_csv,index=False)
    print("Saved exploratory real-data scores:",output_csv)
    print("Scores are not confirmed event labels or calibrated solar-burst probabilities.")
    return result
