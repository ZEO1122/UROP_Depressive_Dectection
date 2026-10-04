"""Participant-span feature extraction, without unpacking archive members."""
from __future__ import annotations
import io
import wave
import zipfile
import numpy as np
import pandas as pd

VISUAL = {'CLNF_AUs.txt': ['AU04_r','AU06_r','AU12_r','AU15_r'],
          'CLNF_pose.txt': ['Rx','Ry','Rz'], 'CLNF_gaze.txt': ['x_h0','y_h0','z_h0']}
FEATURES = ['audio_rms_mean','audio_rms_std','f0_mean','f0_std','voiced_fraction'] + [f'{c}_{s}' for cols in VISUAL.values() for c in cols for s in ['mean','std']]

def span_mask(times, spans):
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in spans:
        if np.isfinite(start) and np.isfinite(stop) and stop > start:
            mask |= (times >= max(0, start)) & (times < stop)
    return mask

def visual_values(frame, columns, spans):
    """Use actual timestamps; sentinel and low-confidence frames never contribute."""
    values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = span_mask(frame['timestamp'].to_numpy(float), spans)
    valid &= (frame['success'].to_numpy(float) == 1) & (frame['confidence'].to_numpy(float) >= .8)
    valid &= np.isfinite(values).all(axis=1) & (values != -100).all(axis=1)
    return values[valid]

def baseline_rows(rows):
    """Reference uses only earlier rapport; no future symptom response is included."""
    symptoms = [r['start'] for r in rows if r['domain'] in ('sleep','interest')]
    if not symptoms:
        return []
    first = min(symptoms)
    candidates = [r for r in rows if r['question_category'] == 'rapport' and r['stop'] <= first and not r.get('scrubbed',False) and not r.get('quality',{}).get('overlap',False)]
    return sorted(candidates, key=lambda r:r['start'])

def union_spans(spans):
    merged=[]
    for a,b in sorted((max(0,float(a)),float(b)) for a,b in spans if b>a):
        if merged and a <= merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
        else: merged.append([a,b])
    return merged

def extract_participant(zip_path, participant_id, rows):
    with zipfile.ZipFile(zip_path) as archive:
        # Read exact top-level members only. Nested archives cannot become participants.
        with archive.open(f'{participant_id}_AUDIO.wav') as source:
            with wave.open(io.BytesIO(source.read()),'rb') as wav:
                sr,channels,width=wav.getframerate(),wav.getnchannels(),wav.getsampwidth()
                if width != 2: raise ValueError(f'Unsupported PCM width: {width}')
                audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
                audio=audio.reshape(-1,channels).mean(axis=1)
        with archive.open(f'{participant_id}_COVAREP.csv') as source:
            covarep=pd.read_csv(source,header=None,usecols=[0,1]).to_numpy(float)
        cov_times=np.arange(len(covarep))*.01
        tables={}
        for suffix,cols in VISUAL.items():
            with archive.open(f'{participant_id}_{suffix}') as source:
                frame=pd.read_csv(source,skipinitialspace=True,usecols=['timestamp','confidence','success']+cols)
                frame.columns=frame.columns.str.strip()
                tables[suffix]=frame
    audio_end=len(audio)/sr
    def summary(spans):
        spans=union_spans(spans)
        expected_seconds=sum(b-a for a,b in spans)
        result={c:np.nan for c in FEATURES}
        rms=[]; covered=0.
        for start,stop in spans:
            start=max(0,start);stop=min(audio_end,stop)
            if stop <= start:continue
            samples=audio[int(start*sr):int(stop*sr)]
            covered += len(samples)/sr
            window=max(1,int(.02*sr))
            n=len(samples)//window
            if n:rms.extend(np.sqrt(np.mean(samples[:n*window].reshape(n,window)**2,axis=1)).tolist())
        rms=np.asarray(rms)
        nonzero=int((rms>1e-5).sum())
        audio_fraction=covered/expected_seconds if expected_seconds else 0.
        audio_ok=covered>=1 and nonzero>=10 and audio_fraction>=.8
        if audio_ok: result.update(audio_rms_mean=float(rms.mean()),audio_rms_std=float(rms.std()))
        cov_valid=span_mask(cov_times,spans) & (cov_times < audio_end) & np.isfinite(covarep).all(axis=1)
        voiced=cov_valid & (covarep[:,1] == 1) & (covarep[:,0] > 0)
        cov_fraction=min(1.,cov_valid.sum()*.01/expected_seconds) if expected_seconds else 0.
        if cov_valid.sum() >= 100 and voiced.sum() >= 10 and cov_fraction>=.8:
            result.update(f0_mean=float(covarep[voiced,0].mean()),f0_std=float(covarep[voiced,0].std()),voiced_fraction=float(voiced.sum()/cov_valid.sum()))
        else: audio_ok=False
        visual_ok=True; counts={}; fractions={}
        for suffix,cols in VISUAL.items():
            values=visual_values(tables[suffix],cols,spans)
            counts[suffix]=len(values)
            steps=np.diff(tables[suffix]['timestamp'].to_numpy(float))
            positive_steps=steps[np.isfinite(steps) & (steps>0)]
            frame_seconds=float(np.median(positive_steps)) if len(positive_steps) else 0.
            fraction=min(1.,len(values)*frame_seconds/expected_seconds) if expected_seconds else 0.
            fractions[suffix]=fraction
            ok=len(values)>=15 and fraction>=.8
            visual_ok &= ok
            if ok:
                for i,c in enumerate(cols):result[c+'_mean']=float(values[:,i].mean());result[c+'_std']=float(values[:,i].std())
        return dict(features=result,audio_ok=bool(audio_ok),visual_ok=bool(visual_ok),audio_covered_seconds=covered,audio_coverage_fraction=audio_fraction,covarep_coverage_fraction=cov_fraction,visual_coverage_fractions=fractions,visual_frame_counts=counts)
    eligible=baseline_rows(rows)
    spans=union_spans([p for r in eligible for p in r['participant_spans']])
    baseline_seconds=sum(b-a for a,b in spans)
    baseline=summary(spans)
    baseline_ok=baseline_seconds>=30 and baseline['audio_covered_seconds']>=30 and baseline['audio_ok'] and baseline['visual_ok']
    output=[]
    for r in rows:
        if r['domain'] not in ('sleep','interest'):continue
        s=summary([] if r.get('scrubbed',False) or r.get('quality',{}).get('overlap',False) else r['participant_spans'])
        output.append(dict(qa_id=r['qa_id'],participant_id=participant_id,**s,baseline=baseline['features'],baseline_ok=bool(baseline_ok),baseline_seconds=baseline_seconds,baseline_qa_ids=[r['qa_id'] for r in eligible]))
    return output
