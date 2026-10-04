"""HiQuE independent A/V features; coverage is audited, never thresholded.

Paper CLNF 68 XY mean/population variance replaces upstream VGG. Sampling
uses the first successful finite frame in each absolute second. No labels read.
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

from urop.data.intervals import validate_spans
from urop.io import require_private_output

SLOTS = 85


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
    if not duration or not len(times):
        return result, False, 0., 0
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    # A lone timestamp has no measurable frame duration; still yields features.
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
    usable = bool(len(indices) >= 1)
    if usable:
        selected = xy[indices]
        result = np.r_[selected.mean(axis=0), selected.var(axis=0)].astype(np.float32)
        usable = bool(np.isfinite(result).all())
        if not usable:
            result[:] = 0
    return result, usable, coverage, len(indices)


def audio_samples(signal, sample_rate, spans):
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
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
    usable = bool(len(combined) and np.isfinite(combined).all())
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
    with zipfile.ZipFile(archive) as z:
        members = {name: {'crc32': z.getinfo(name).CRC, 'size': z.getinfo(name).file_size}
                   for name in (f'{pid}_AUDIO.wav', f'{pid}_CLNF_features.txt')
                   if name in z.namelist()}
    fingerprint = hashlib.sha256(json.dumps({
        'segments': record, 'member_metadata': members, 'zip_size': stat.st_size, 'zip_mtime_ns': stat.st_mtime_ns,
        'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'opensmile': opensmile.__version__, 'numpy': np.__version__,
        'pandas': pd.__version__, 'soundfile': sf.__version__,
        'libsndfile': sf.__libsndfile_version__,
        'span_validator': hashlib.sha256(Path(validate_spans.__code__.co_filename).read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    if npz_path.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if (cached.get('fingerprint') == fingerprint
                and cached.get('npz_sha256') == hashlib.sha256(npz_path.read_bytes()).hexdigest()):
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
            valid = table['success'].to_numpy() == 1
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
                        errors.append({'slot': s, 'modality': 'A', 'reason': 'empty or nonfinite audio samples'})
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
                        errors.append({'slot': s, 'modality': 'V', 'reason': 'no successful finite sampled frame'})
                except Exception as exc:
                    errors.append({'slot': s, 'modality': 'V', 'reason': repr(exc)})
    summary = {'A_slots': int(arrays['mask_A'].sum()), 'V_slots': int(arrays['mask_V'].sum()),
               'error_count': len(errors)}
    metadata = {'participant_id': pid, 'split': record['split'], 'fingerprint': fingerprint,
                'summary': summary, 'errors': errors, 'visual_order': 'xmean68,ymean68,xvar68,yvar68',
                'deviation': 'CLNF paper representation replaces upstream VGG; first successful finite frame per absolute second; independent modality masks; coverage descriptive only'}
    temporary = output / f'{pid}.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(npz_path)
    metadata['npz_sha256'] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    temporary_meta = output / f'{pid}.tmp.json'
    temporary_meta.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary_meta.replace(meta_path)
    return {'participant_id': pid, 'cached': False, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', type=Path, required=True)
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir = require_private_output(args.output_dir)
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
