"""Train/dev-only event-preserving manual transcript preparation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import wave
import zipfile
from urop.data.questions import normalize
from urop.io import require_private_output, write_json as write
from urop.paths import DEFAULT_STUDY




def cohort(root):
    result = {}
    for split in ('train', 'dev'):
        with (root / f'{split}_split_Depression_AVEC2017.csv').open() as stream:
            ids = [int(r['Participant_ID']) for r in csv.DictReader(stream) if r['Participant_ID'].strip()]
        for pid in ids:
            if split == 'dev' and pid in (440,451,458):
                continue
            if pid in result:
                raise ValueError('duplicate split ID')
            result[pid] = split
    if Counter(result.values()) != {'train':107,'dev':32}:
        raise ValueError('unexpected cohort')
    return result


def read_source(root, pid):
    with zipfile.ZipFile(root / f'{pid}_P.zip') as archive:
        blob = archive.read(f'{pid}_TRANSCRIPT.csv')
        with archive.open(f'{pid}_AUDIO.wav') as stream:
            with wave.open(stream) as audio:
                duration = audio.getnframes()/audio.getframerate()
    rows = list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig')), delimiter='\t'))
    return rows, duration, hashlib.sha256(blob).hexdigest()


def inventory(root, base):
    collected = {}
    for pid, split in sorted(cohort(root).items()):
        if split != 'train':
            continue
        rows, _, _ = read_source(root,pid)
        for index,row in enumerate(rows):
            if row['speaker'] != 'Ellie':
                continue
            key = normalize(row['value'])
            entry = collected.setdefault(key, {'normalized':key,'frequency':0,'raw_variants':{},'source_refs':[]})
            entry['frequency'] += 1
            raw = row['value']
            entry['raw_variants'][raw] = entry['raw_variants'].get(raw,0)+1
            entry['source_refs'].append({'participant_id':pid,'row_id':index})
    write(base/'train_utterance_inventory.json', {'normalization':'urop.data.questions.normalize','entries':[collected[k] for k in sorted(collected)]})


def union(spans):
    result = []
    for start,end in sorted(spans):
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            raise ValueError('invalid interval')
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def clipped_span(row,duration):
    try:
        start,end = float(row['start_time']),float(row['stop_time'])
    except (ValueError,TypeError):
        return [],'invalid_time'
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return [],'invalid_time'
    clipped = [max(0.,start),min(duration,end)]
    if clipped[1] <= clipped[0]:
        return [],'outside_audio'
    return [clipped], 'clipped' if clipped != [start,end] else 'valid'


def clean_answer(text):
    # Only marker tokens are removed; surrounding lexical content is retained.
    return ' '.join(re.sub(r'(?i)[\[<(]?\b(?:scrubbed_entry|xxx)\b[\])>]?',' ',text).split())


def build_record(pid,split,rows,duration,mapping,questions):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid audio duration')
    events, source = [], []
    current = None
    previous = None
    root = None
    depth = 0
    for index,row in enumerate(rows):
        spans,time_status = clipped_span(row,duration)
        item = {'row_id':index,'raw':row,'time_status':time_status,'clipped_spans':spans,'event_id':None}
        source.append(item)
        if row['speaker'] == 'Ellie':
            key = normalize(row['value'])
            mapped = mapping.get(key,{'status':'unknown','slot':None,'reason':'not in frozen TRAIN mapping'})
            if mapped['status'] == 'ack':
                item['classification'] = 'ack'
                if current is not None:
                    item['event_id'] = events[current]['event_id']
                    events[current]['ack_row_ids'].append(index)
                    events[current]['source_row_ids'].append(index)
                continue
            slot = mapped.get('slot')
            if mapped['status'] not in ('mapped','unknown') or (mapped['status']=='mapped' and (not isinstance(slot,int) or not 0 <= slot < len(questions))):
                raise ValueError('invalid mapping entry')
            if mapped['status'] == 'unknown':
                slot = None
            event_id = f'{pid}:{index}'
            kind = questions[slot]['type'] if slot is not None else 'Unknown'
            follow = kind.lower().startswith('follow')
            if slot is None:
                root,depth = None,0
            elif not follow:
                root,depth = event_id,0
            else:
                depth += 1
            event = {'event_id':event_id,'participant_id':pid,'split':split,'source_row_ids':[index],
                     'question_row_id':index,'raw_question':row['value'],'normalized_question':key,
                     'canonical_question_id':slot,'slot':slot,'question_type':kind,'type':kind,
                     'mapping_status':mapped['status'],'mapping_reason':mapped.get('reason',''),
                     'preceding_question_event_id':previous,'root_primary_event_id':root,
                     'followup_depth':depth,'parent_unknown':bool(follow and root is None),
                     'answer_row_ids':[],'answer_parts':[],'answer_intervals':[],'ack_row_ids':[],
                     'question_spans':spans}
            events.append(event)
            current = len(events)-1
            previous = event_id
            item.update(classification='question_'+mapped['status'],event_id=event_id)
        elif row['speaker'] == 'Participant':
            text = clean_answer(row['value'])
            if current is not None:
                event = events[current]
                item['event_id'] = event['event_id']
                event['source_row_ids'].append(index)
            if not text:
                item['classification'] = 'empty_or_scrubbed'
            elif not spans:
                item['classification'] = 'excluded_invalid_time'
            elif current is None:
                item['classification'] = 'prequestion'
            else:
                event = events[current]
                item['classification'] = 'answer_'+event['mapping_status']
                event['answer_row_ids'].append(index)
                event['answer_parts'].append({'row_id':index,'text':text,'spans':spans})
                event['answer_intervals'].extend(spans)
        else:
            item['classification'] = 'excluded_unknown_speaker'
    for event in events:
        event['answer_parts'].sort(key=lambda p:(p['spans'][0][0],p['row_id']))
        event['answer_text'] = ' '.join(p['text'] for p in event['answer_parts'])
        event['answer_intervals'] = union(event['answer_intervals'])
    counts = Counter(r['classification'] for r in source)
    return {'participant_id':pid,'split':split,'audio_duration':duration,'events':events,'source_rows':source,'audit':dict(counts)}


def select_record(record,policy):
    if policy not in ('first','last','all'):
        raise ValueError('unknown response policy')
    groups = defaultdict(list)
    for event in record['events']:
        if event['slot'] is not None and event['answer_text'] and event['answer_intervals']:
            groups[event['slot']].append(event)
    slots = []
    for slot,events in sorted(groups.items()):
        selected = events[:1] if policy=='first' else events[-1:] if policy=='last' else events
        parts = sorted([p for e in selected for p in e['answer_parts']],key=lambda p:(p['spans'][0][0],p['row_id']))
        slots.append({'slot':slot,'type':selected[0]['type'],'text':' '.join(p['text'] for p in parts),
                      'spans':union([s for p in parts for s in p['spans']]),
                      'event_ids':[e['event_id'] for e in selected],'answer_row_ids':[p['row_id'] for p in parts]})
    return {'participant_id':record['participant_id'],'split':record['split'],'slots':slots,
            'question_presence':[s in groups for s in range(85)],'position_ids':list(range(85))}


def packet_item(event,record,stratum):
    related_ids = {event['preceding_question_event_id'],event['root_primary_event_id']}
    context_events = [e for e in record['events'] if e['event_id'] in related_ids]
    row_ids = set(event['source_row_ids'])
    # Neighboring rows show the question boundary and any next question explicitly.
    lo,hi = max(0,min(row_ids)-2),min(len(record['source_rows']),max(row_ids)+3)
    return {'stratum':stratum,'event':event,'relation_context_events':context_events,
            'source_rows':[r for r in record['source_rows'] if r['row_id'] in row_ids],
            'neighbor_context_rows':record['source_rows'][lo:hi]}


def review_packet(records):
    pool = []
    for record in records:
        if record['split'] != 'train':
            continue
        frequencies = Counter(e['slot'] for e in record['events'] if e['slot'] is not None)
        for event in record['events']:
            tags = ['general']
            if event['slot'] is not None and frequencies[event['slot']] > 1: tags.append('repeated')
            if event['followup_depth'] > 1: tags.append('chain')
            if event['ack_row_ids'] or event['slot'] is None: tags.append('ackunknown')
            pool.append((event,record,tags))
    rng = random.Random(20261003)
    rng.shuffle(pool)
    used,packets = set(),[]
    # Reserve narrow strata first so general cannot consume them.
    for stratum,count in [('chain',40),('repeated',60),('ackunknown',20),('general',80)]:
        candidates = [p for p in pool if stratum in p[2] and p[0]['event_id'] not in used]
        for event,record,_ in candidates[:count]:
            used.add(event['event_id'])
            packets.append(packet_item(event,record,stratum))
    for event,record,_ in pool:
        if len(packets)>=200: break
        if event['event_id'] not in used:
            used.add(event['event_id'])
            packets.append(packet_item(event,record,'shortfall_backfill'))
    return {'seed':20261003,'status':'AI-review draft; no independent human gold','counts':dict(Counter(p['stratum'] for p in packets)),'packets':packets}


def prepare(root,base,mapping_path,questions_path):
    payload = json.loads(mapping_path.read_text())
    mapping = payload.get('mapping',payload)
    questions = json.loads(questions_path.read_text())
    records,manifest = [],[]
    for pid,split in sorted(cohort(root).items()):
        rows,duration,sha = read_source(root,pid)
        record = build_record(pid,split,rows,duration,mapping,questions)
        record['transcript_sha256'] = sha
        records.append(record)
        manifest.append({'participant_id':pid,'split':split,'transcript_sha256':sha,'audio_duration':duration})
    (base/'events.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
    for condition,policy in [('E1','last'),('E2','first'),('E3','all')]:
        selected = [select_record(r,policy) for r in records]
        if any(not r['slots'] for r in selected):
            raise ValueError('participant without usable mapped response')
        (base/f'segments_{condition}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in selected))
    write(base/'manifest.json',{'participants':manifest,'mapping_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write(base/'input_review_packet.json',review_packet(records))
    counts = Counter()
    times = Counter()
    for record in records:
        counts.update(record['audit'])
        times.update(r['time_status'] for r in record['source_rows'])
    write(base/'preparation_summary.json',{'participants':len(records),'split_counts':dict(Counter(r['split'] for r in records)), 'source_row_classifications':dict(counts),'source_time_status':dict(times),'all_rows_accounted':sum(counts.values())==sum(len(r['source_rows']) for r in records),'test_accessed':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['inventory','prepare'],required=True)
    parser.add_argument('--base',type=Path,default=DEFAULT_STUDY)
    parser.add_argument('--zip-dir',type=Path,default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--mapping',type=Path)
    parser.add_argument('--questions',type=Path,default=Path('Data/hique_reproduction/questions.json'))
    args=parser.parse_args()
    args.base=require_private_output(args.base)
    args.base.mkdir(parents=True,exist_ok=True)
    if args.stage=='inventory': inventory(args.zip_dir,args.base)
    else: prepare(args.zip_dir,args.base,args.mapping or args.base/'mapping.json',args.questions)


if __name__=='__main__': main()
