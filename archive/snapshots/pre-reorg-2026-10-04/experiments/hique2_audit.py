"""Label-free audit of punctuation-inferred speaker intervals against train transcripts."""
import csv
import io
import json
from pathlib import Path
import zipfile

from .prepare import split_ids
from .hique2_data import write


def overlap(a, b):
    intervals = sorted((max(a[0], x), min(a[1], y)) for x, y in b if min(a[1], y) > max(a[0], x))
    total = 0.
    end = -1.
    for left, right in intervals:
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def main():
    base = Path('Data/hique_reproduction_excluding440')
    root = Path('Data/DAIC-WOZ')
    # Predetermined first ten official train IDs; no outcomes parsed.
    selected = sorted(split_ids(root)['train'])[:10]
    records = []
    for pid in selected:
        file = base / 'asr' / f'{pid}.json'
        if not file.exists():
            raise ValueError(f'Wait for ASR train audit participant {pid}')
        data = json.loads(file.read_text())
        with zipfile.ZipFile(root / f'{pid}_P.zip') as z:
            transcript = list(csv.DictReader(io.StringIO(z.read(f'{pid}_TRANSCRIPT.csv').decode('utf-8-sig')), delimiter='\t'))
        ellie = [[float(r['start_time']), float(r['stop_time'])] for r in transcript if r['speaker'] == 'Ellie']
        participant = [[float(r['start_time']), float(r['stop_time'])] for r in transcript if r['speaker'] == 'Participant']
        inferred = [[float(r['start']), float(r['end'])] for r in data['rows'] if r['speaker'] == 'Ellie']
        duration = sum(b-a for a,b in inferred)
        e_overlap = sum(overlap(a, ellie) for a in inferred)
        p_overlap = sum(overlap(a, participant) for a in inferred)
        records.append({'participant_id': pid, 'pseudo_question_rows': len(inferred),
                        'pseudo_question_seconds': duration, 'overlap_manual_ellie_seconds': e_overlap,
                        'overlap_manual_participant_seconds': p_overlap,
                        'fraction_pseudo_duration_overlapping_ellie': e_overlap/duration if duration else None,
                        'fraction_pseudo_duration_overlapping_participant': p_overlap/duration if duration else None})
    duration = sum(r['pseudo_question_seconds'] for r in records)
    summary = {'selection': 'first ten official train IDs; fixed before inspection', 'records': records,
               'pseudo_duration_seconds': duration,
               'fraction_overlapping_manual_ellie': sum(r['overlap_manual_ellie_seconds'] for r in records)/duration if duration else None,
               'fraction_overlapping_manual_participant': sum(r['overlap_manual_participant_seconds'] for r in records)/duration if duration else None,
               'interpretation': 'Descriptive time overlap, not verified diarization accuracy; ASR/annotation boundaries differ and speaker overlap can occur. No outcome or mapping changes from this audit.'}
    write(base / 'speaker_audit.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
