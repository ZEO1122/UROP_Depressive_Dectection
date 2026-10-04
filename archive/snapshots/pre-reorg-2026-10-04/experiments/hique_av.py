"""API-free HiQuE acoustic and paper CLNF features, without reading labels.

Visual upstream VGG extraction is replaced with paper-described 68 XY landmark
mean/variance at 1 fps (272 dimensions). Masks/coverage are explicit deviations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

SLOTS = 85


def validate_spans(spans):
    values = np.asarray(spans, dtype=float).reshape(-1, 2)
    if (not np.isfinite(values).all() or np.any(values[:, 0] < 0)
            or np.any(values[:, 1] <= values[:, 0])
            or np.any(values[1:, 0] < values[:-1, 1])):
        raise ValueError('spans must be finite, positive, sorted and nonoverlapping')
    return values


def visual_features(timestamps, xy, valid, spans):
    """First valid frame in each wallclock-second bin; population variance.

    Coverage integrates valid frame cells, capped at the median frame interval,
    so gaps and truncated streams never count as observed time.
    """
    spans = validate_spans(spans)
    result = np.zeros(272, dtype=np.float32)
    duration = float(np.sum(spans[:, 1] - spans[:, 0]))
    times = np.asarray(timestamps, dtype=float)
    xy = np.asarray(xy, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if xy.shape != (len(times), 136) or valid.shape != times.shape:
        raise ValueError('CLNF requires 136 XY values per timestamp')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('CLNF timestamps must be finite and strictly increasing')
    if not duration or len(times) < 2:
        return result, False, 0., 0
    step = float(np.median(np.diff(times)))
    ends = np.minimum(times + step, np.r_[times[1:], times[-1] + step])
    inside = np.zeros(len(times), dtype=bool)
    observed = 0.
    valid = valid & np.isfinite(xy).all(axis=1)
    for start, stop in spans:
        inside |= (times >= start) & (times < stop)
        overlap = np.maximum(0., np.minimum(ends, stop) - np.maximum(times, start))
        observed += float(overlap[valid].sum())
    indices = np.flatnonzero(inside & valid)
    if len(indices):
        _, first = np.unique(np.floor(times[indices]), return_index=True)
        indices = indices[first]
    coverage = min(1., observed / duration)
    usable = bool(len(indices) >= 2 and coverage >= .5)
    if usable:
        selected = xy[indices]
        result = np.r_[selected.mean(axis=0), selected.var(axis=0)].astype(np.float32)
        usable = bool(np.isfinite(result).all())
        if not usable:
            result[:] = 0
    return result, usable, coverage, len(indices)


def audio_samples(signal, sample_rate, spans):
    spans = validate_spans(spans)
    duration = float(np.sum(spans[:, 1] - spans[:, 0]))
    pieces = []
    for start, stop in spans:
        a = min(len(signal), int(np.ceil(start * sample_rate)))
        b = min(len(signal), int(np.ceil(stop * sample_rate)))
        if b > a:
            pieces.append(signal[a:b])
    combined = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    coverage = min(1., len(combined) / sample_rate / duration) if duration else 0.
    usable = bool(coverage >= .8 and len(combined) / sample_rate >= .5
                  and np.isfinite(combined).all())
    return combined, usable, coverage


def extract_one(task):
    import opensmile
    import pandas as pd
    import soundfile as sf

    record, zip_dir, output_dir = task
    pid = int(record['participant_id'])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(zip_dir) / f'{pid}_P.zip'
    npz_path = output / f'{pid}.npz'
    meta_path = output / f'{pid}.json'
    stat = archive.stat()
    fingerprint = hashlib.sha256(json.dumps({
        'segments': record, 'zip_size': stat.st_size, 'zip_mtime_ns': stat.st_mtime_ns,
        'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'opensmile': opensmile.__version__, 'numpy': np.__version__,
    }, sort_keys=True).encode()).hexdigest()
    if npz_path.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached.get('fingerprint') == fingerprint:
            return {'participant_id': pid, 'cached': True, **cached['summary']}
    arrays = {'A': np.zeros((SLOTS, 88), dtype=np.float32),
              'V': np.zeros((SLOTS, 272), dtype=np.float32),
              'mask_A': np.zeros(SLOTS, dtype=bool), 'mask_V': np.zeros(SLOTS, dtype=bool),
              'duration': np.zeros(SLOTS, dtype=np.float32),
              'audio_coverage': np.zeros(SLOTS, dtype=np.float32),
              'visual_coverage': np.zeros(SLOTS, dtype=np.float32),
              'visual_sample_count': np.zeros(SLOTS, dtype=np.int32)}
    errors = []
    slots = [int(slot['slot']) for slot in record['slots']]
    if len(set(slots)) != len(slots) or any(s < 0 or s >= SLOTS for s in slots):
        raise ValueError(f'{pid}: duplicate or out of range slots')
    with zipfile.ZipFile(archive) as z:
        try:
            signal, rate = sf.read(io.BytesIO(z.read(f'{pid}_AUDIO.wav')), dtype='float32')
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                    feature_level=opensmile.FeatureLevel.Functionals)
        except Exception as exc:
            signal = None
            errors.append({'modality': 'A', 'reason': repr(exc)})
        try:
            with z.open(f'{pid}_CLNF_features.txt') as stream:
                table = pd.read_csv(stream, skipinitialspace=True)
            table.columns = table.columns.str.strip()
            times = table['timestamp'].to_numpy()
            xy = table[[f'{axis}{i}' for axis in 'xy' for i in range(68)]].to_numpy()
            valid = (table['success'].to_numpy() == 1) & (table['confidence'].to_numpy() >= .8)
        except Exception as exc:
            table = None
            errors.append({'modality': 'V', 'reason': repr(exc)})
        for slot in record['slots']:
            s = int(slot['slot'])
            try:
                spans = validate_spans(slot['spans'])
            except ValueError as exc:
                errors.append({'slot': s, 'modality': 'AV', 'reason': str(exc)})
                continue
            arrays['duration'][s] = np.sum(spans[:, 1] - spans[:, 0])
            if signal is not None:
                try:
                    samples, usable, coverage = audio_samples(signal, rate, spans)
                    arrays['audio_coverage'][s] = coverage
                    if usable:
                        values = smile.process_signal(samples, rate).to_numpy().reshape(-1)
                        if values.shape != (88,) or not np.isfinite(values).all():
                            raise ValueError('nonfinite or invalid eGeMAPS shape')
                        arrays['A'][s] = values
                        arrays['mask_A'][s] = True
                    else:
                        errors.append({'slot': s, 'modality': 'A', 'reason': 'short/nonfinite audio or coverage below .8'})
                except Exception as exc:
                    errors.append({'slot': s, 'modality': 'A', 'reason': repr(exc)})
            if table is not None:
                try:
                    values, usable, coverage, count = visual_features(times, xy, valid, spans)
                    arrays['V'][s] = values
                    arrays['mask_V'][s] = usable
                    arrays['visual_coverage'][s] = coverage
                    arrays['visual_sample_count'][s] = count
                    if not usable:
                        errors.append({'slot': s, 'modality': 'V', 'reason': 'fewer than 2 frames or coverage below .5'})
                except Exception as exc:
                    errors.append({'slot': s, 'modality': 'V', 'reason': repr(exc)})
    summary = {'A_slots': int(arrays['mask_A'].sum()), 'V_slots': int(arrays['mask_V'].sum()),
               'error_count': len(errors)}
    metadata = {'participant_id': pid, 'split': record['split'], 'fingerprint': fingerprint,
                'summary': summary, 'errors': errors, 'visual_order': 'xmean68,ymean68,xvar68,yvar68',
                'deviation': 'CLNF paper representation replaces upstream VGG; explicit coverage masks; first valid frame per wallclock second'}
    temporary = output / f'{pid}.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(npz_path)
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return {'participant_id': pid, 'cached': False, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', type=Path, required=True)
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.segments.read_text().splitlines() if line.strip()]
    if len({r['participant_id'] for r in records}) != len(records):
        raise ValueError('duplicate participant records')
    tasks = [(record, str(args.zip_dir), str(args.output_dir)) for record in records]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = []
        for summary in pool.map(extract_one, tasks):
            print(json.dumps(summary), flush=True)
            summaries.append(summary)
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
