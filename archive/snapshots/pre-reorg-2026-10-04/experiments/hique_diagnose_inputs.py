"""Read-only train input audit for HiQuE reproduction; no model fitting or labels."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .prepare import split_ids
from .hique2_audit import overlap


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def duration(intervals):
    return sum(b-a for a,b in union(intervals))


def intersect(a, b):
    b = union(b)
    return sum(overlap(interval, b) for interval in union(a))


def main():
    base = Path('Data/hique_reproduction_excluding440')
    out = Path('Data/hique_diagnosis_2026-10-03')
    out.mkdir(exist_ok=True)
    root = Path('Data/DAIC-WOZ')
    ids = sorted(split_ids(root)['train'])
    records = {r['participant_id']: r for r in map(json.loads, (base/'segments.jsonl').read_text().splitlines())}
    stats = []
    for pid in ids:
        asr = json.loads((base/'asr'/f'{pid}.json').read_text())
        with zipfile.ZipFile(root/f'{pid}_P.zip') as archive:
            rows = list(csv.DictReader(io.StringIO(archive.read(f'{pid}_TRANSCRIPT.csv').decode('utf-8-sig')), delimiter='\t'))
        ellie = union([[float(r['start_time']), float(r['stop_time'])] for r in rows if r['speaker']=='Ellie'])
        participant = union([[float(r['start_time']), float(r['stop_time'])] for r in rows if r['speaker']=='Participant'])
        pseudo = [[r['start'],r['end']] for r in asr['rows'] if r['speaker']=='Ellie']
        assigned = [e for e in records[pid]['events'] if 'text' in e]
        retained = records[pid]['slots']
        answers = [span for slot in retained for span in slot['spans']]
        dominant = {'ellie':0,'participant':0,'tie':0,'neither':0}
        for span in pseudo:
            e,p = overlap(span, ellie), overlap(span, participant)
            dominant['ellie' if e>p else 'participant' if p>e else 'tie' if e>0 else 'neither'] += 1
        stats.append({'participant_id':pid,'pseudo_rows':len(pseudo),
                      'pseudo_seconds':duration(pseudo), 'manual_ellie_seconds':duration(ellie),
                      'pseudo_overlap_ellie':intersect(pseudo,ellie),
                      'pseudo_overlap_participant':intersect(pseudo,participant),
                      'pseudo_overlap_either':intersect(pseudo,union(ellie+participant)),
                      'response_seconds':duration(answers),'response_overlap_ellie':intersect(answers,ellie),
                      'response_overlap_participant':intersect(answers,participant),
                      'dominant':dominant,'answer_events':len(assigned),'retained_slots':len(retained),
                      'answer_event_words':sum(len(e['text'].split()) for e in assigned),
                      'retained_words':sum(len(e['text'].split()) for e in retained),
                      'retained_chained_followups':sum(e['followup_depth']>1 for e in retained),
                      'retained_parent_diff_previous':sum(e['type']=='Follow-up' and not e['orphan_fallback'] and e['parent_topic_id']!=e['previous_question_id'] for e in retained)})
    aggregate = {key:sum(s[key] for s in stats) for key in stats[0] if key not in ['participant_id','dominant']}
    aggregate['dominant'] = {k:sum(s['dominant'][k] for s in stats) for k in stats[0]['dominant']}
    aggregate['pseudo_time_fraction_on_ellie'] = aggregate['pseudo_overlap_ellie']/aggregate['pseudo_seconds']
    aggregate['pseudo_time_fraction_on_participant'] = aggregate['pseudo_overlap_participant']/aggregate['pseudo_seconds']
    aggregate['pseudo_time_fraction_on_ellie_among_either'] = aggregate['pseudo_overlap_ellie']/aggregate['pseudo_overlap_either']
    aggregate['manual_ellie_time_covered_by_pseudo'] = aggregate['pseudo_overlap_ellie']/aggregate['manual_ellie_seconds']
    aggregate['response_time_fraction_on_ellie'] = aggregate['response_overlap_ellie']/aggregate['response_seconds']
    aggregate['response_time_fraction_on_participant'] = aggregate['response_overlap_participant']/aggregate['response_seconds']
    aggregate['overwrite_word_loss_fraction'] = 1-aggregate['retained_words']/aggregate['answer_event_words']
    aggregate['overwrite_event_loss_fraction'] = 1-aggregate['retained_slots']/aggregate['answer_events']
    aggregate['participants_pseudo_majority_on_participant'] = sum(s['pseudo_overlap_participant']>s['pseudo_overlap_ellie'] for s in stats)
    aggregate['participants_zero_ellie_overlap'] = sum(s['pseudo_overlap_ellie']==0 for s in stats)
    aggregate['retained_slot_median'] = float(np.median([s['retained_slots'] for s in stats]))
    result = {'scope':'all107officialtrain; no outcomes loaded; timestamp overlap is diagnostic not exact diarization accuracy',
              'aggregate':aggregate,'participants':stats,
              'limitations':['Provided transcript timestamps can have known offsets; overlap statistics include silence and ASR boundary differences.',
                             'Overwrite loss describes this inferred pipeline, not total original semantic information or a causal F1 decrement.']}
    (out/'input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
