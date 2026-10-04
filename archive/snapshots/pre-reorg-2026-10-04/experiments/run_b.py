"""B: fixed-model feasibility pilot; targets are weak labels, not clinical gold."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,log_loss,confusion_matrix
from .features import FEATURES,extract_participant
LABELS=['support','deny','unknown']

def cache_key(path,rows):
    stat=path.stat()
    payload={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'rows':rows,'extractor':hashlib.sha256(Path(__file__).with_name('features.py').read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def matrices(rows,features):
    """Construct every comparator from exactly the same matched QA records."""
    nuisance=np.array([[r['word_count'],r['duration'],r['position'],int(r['question_category']=='sleep')] for r in rows],float)
    raw=np.array([[f['features'][k] for k in FEATURES] for f in features],float)
    base=np.array([[f['baseline'][k] for k in FEATURES] for f in features],float)
    return {'nuisance':nuisance,'raw':np.column_stack([nuisance,raw]),'delta':np.column_stack([nuisance,raw-base]),'raw_and_baseline':np.column_stack([nuisance,raw,base])}

def metrics(y,pred,proba):
    return {'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)), 'log_loss':float(log_loss(y,proba,labels=LABELS)), 'confusion_matrix_order':LABELS,'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist()}

def bootstrap(y,preds,groups):
    """Paired cluster bootstrap: all QAs of a sampled person move together."""
    unique=sorted(set(groups));rng=np.random.default_rng(20260927)
    indices=[np.where(groups==g)[0] for g in unique]
    diffs=defaultdict(list)
    comparisons=[('raw','nuisance'),('delta','raw'),('raw_and_baseline','raw')]
    for _ in range(2000):
        ix=np.concatenate([indices[j] for j in rng.integers(0,len(indices),len(indices))])
        scores={name:f1_score(y[ix],prediction[ix],labels=LABELS,average='macro',zero_division=0) for name,prediction in preds.items()}
        for first,second in comparisons:
            diffs[first+' minus '+second].append(scores[first]-scores[second])
    return {k:{'comparison':k,'macro_f1_difference_ci95':np.quantile(v,[.025,.975]).tolist(),'bootstrap_repetitions':2000} for k,v in diffs.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--zip-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    data_root=Path(__file__).resolve().parents[1]/'Data'
    if not args.output_dir.resolve().is_relative_to(data_root.resolve()):
        raise ValueError('Participant-derived outputs must remain beneath repository Data/')
    args.output_dir.mkdir(parents=True,exist_ok=True);cache=args.output_dir/'private_features';cache.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (args.data_dir/'qa.jsonl').read_text().splitlines() if l.strip()]
    if any(r['split'] not in ('train','dev') for r in rows):raise ValueError('Test sealed: B input may contain train/dev only')
    if any(r['participant_id'] in (300,440,451,458,480) for r in rows):raise ValueError('Excluded participant in input')
    grouped=defaultdict(list)
    for r in rows:grouped[r['participant_id']].append(r)
    features=[];errors={};futures={};cached=0
    completed=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for pid,part in sorted(grouped.items()):
            archive=args.zip_dir/f'{pid}_P.zip';key=cache_key(archive,part);out=cache/f'{pid}_{key}.json'
            if out.exists():features.extend(json.loads(out.read_text()));cached+=1
            else:futures[pool.submit(extract_participant,archive,pid,part)]=(pid,out)
        for done in as_completed(futures):
            pid,out=futures[done]
            try:
                data=done.result();out.write_text(json.dumps(data,allow_nan=True));features.extend(data)
            except Exception as exc:errors[str(pid)]=repr(exc)
            completed+=1
            print(f'B feature extraction: {cached+completed}/{len(grouped)} participants',flush=True)
    byqa={f['qa_id']:f for f in features}
    symptom=[r for r in rows if r['domain'] in ('sleep','interest')]
    matched=[r for r in symptom if r['qa_id'] in byqa and all(byqa[r['qa_id']][k] for k in ['audio_ok','visual_ok','baseline_ok']) and not r.get('scrubbed',False)]
    fs=[byqa[r['qa_id']] for r in matched]
    report={'status':'WEAK_LABEL_AGREEMENT_ONLY','clinical_efficacy':'NOT_TESTED','independent_gold':'BLOCKED_PENDING_TWO_HUMAN_ANNOTATORS','test_used':False,'covarep_metadata':'DAICWOZDepression_Documentation_AVEC2017.pdf section 4: first two columns F0,VUV at 100 Hz; voiced F0 only','feature_columns':FEATURES,'cohort':{},'extraction_errors':errors,'baseline_rule':'All earlier rapport QA before first sleep/interest QA; >=30 actual participant seconds, >=30 available audio seconds and valid A/V; not clinically neutral','models':{},'limitations':['Rules supply weak outcome labels; this is not independent clinical validation.','Only earlier rapport exists for baseline; cohort selection may bias estimates.','COVAREP zero rows can mix unvoiced and scrubbed entries; transcript-flagged scrubbed QAs excluded.','Every modality needs at least 80 percent valid temporal coverage of selected participant spans.','Overlap-flagged and scrubbed QA are excluded from features and reference to avoid speaker contamination.','Actual timestamps only; incomplete video is never extrapolated.','All uncertainty intervals are exploratory dev-set paired participant bootstrap intervals.']}
    for split in ['train','dev']:
        src=[r for r in symptom if r['split']==split];selected=[r for r in matched if r['split']==split]
        report['cohort'][split]={'symptom_qa':len(src),'symptom_participants':len(set(r['participant_id'] for r in src)),'matched_qa':len(selected),'matched_participants':len(set(r['participant_id'] for r in selected)),'weak_label_counts':dict(Counter(r['weak_label'] for r in selected)),'failed_audio_qa':sum(not byqa[r['qa_id']]['audio_ok'] for r in src if r['qa_id'] in byqa),'failed_visual_qa':sum(not byqa[r['qa_id']]['visual_ok'] for r in src if r['qa_id'] in byqa),'failed_baseline_qa':sum(not byqa[r['qa_id']]['baseline_ok'] for r in src if r['qa_id'] in byqa)}
    prediction_rows=[]
    if matched:
        x=matrices(matched,fs);y=np.array([r['weak_label'] for r in matched]);train=np.array([r['split']=='train' for r in matched]);dev=~train
        if train.any() and dev.any() and len(set(y[train]))>=2:
            predictions={}
            for name,values in x.items():
                model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=20260927))
                model.fit(values[train],y[train]);pred=model.predict(values[dev]);predictions[name]=pred
                # sklearn log_loss expects alphabetically ordered columns when labels passed.
                classes=sorted(LABELS);prob=np.full((int(dev.sum()),len(classes)),1e-15)
                mp=model.predict_proba(values[dev])
                for j,c in enumerate(model[-1].classes_):prob[:,classes.index(c)]=mp[:,j]
                prob/=prob.sum(axis=1,keepdims=True)
                report['models'][name]=metrics(y[dev],pred,prob)
                report['models'][name]['by_domain']={d:{'n':int(sum(r['domain']==d for r in np.array(matched,dtype=object)[dev])),'macro_f1':float(f1_score(y[dev][np.array([r['domain']==d for r in np.array(matched,dtype=object)[dev]])],pred[np.array([r['domain']==d for r in np.array(matched,dtype=object)[dev]])],labels=LABELS,average='macro',zero_division=0))} for d in ['sleep','interest']}
                for r,true,guess in zip(np.array(matched,dtype=object)[dev],y[dev],pred):prediction_rows.append({'qa_id':r['qa_id'],'participant_id':r['participant_id'],'model':name,'weak_label':true,'prediction':guess})
            dev_groups=np.array([r['participant_id'] for r in matched])[dev]
            missing_train=sorted(set(LABELS)-set(y[train]))
            report['feasibility']={'missing_training_classes':missing_train,'dev_participants':len(set(dev_groups)),'model_results_scope':'DIAGNOSTIC_WEAK_LABEL_AGREEMENT_ONLY'}
            if missing_train:
                report['limitations'].append('Training matched cohort lacks classes '+str(missing_train)+'; these classes cannot be learned. Log loss uses epsilon probability for missing classes and is diagnostic only.')
            if len(set(dev_groups)) <= 2 or missing_train:
                report['paired_bootstrap']={'status':'NOT_INTERPRETABLE_NOT_COMPUTED','reason':'At most two dev participants or missing training outcome classes; no inferential claim justified.'}
            else:
                report['paired_bootstrap']=bootstrap(y[dev],predictions,dev_groups)
        else: report['model_blocker']='Insufficient matched train/dev rows or fewer than two training classes'
    else:report['model_blocker']='No matched cohort with prior rapport baseline and valid audio/video'
    pd.DataFrame(prediction_rows).to_csv(args.output_dir/'private_predictions.csv',index=False)
    pd.DataFrame([{'qa_id':r['qa_id'],'participant_id':r['participant_id'],'split':r['split'],'domain':r['domain'],'matched':r in matched,**{k:byqa.get(r['qa_id'],{}).get(k) for k in ['audio_ok','visual_ok','baseline_ok','baseline_seconds','audio_covered_seconds','audio_coverage_fraction','covarep_coverage_fraction','visual_coverage_fractions']}} for r in symptom]).to_csv(args.output_dir/'private_coverage.csv',index=False)
    report['data_sha256']=hashlib.sha256((args.data_dir/'qa.jsonl').read_bytes()).hexdigest()
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
