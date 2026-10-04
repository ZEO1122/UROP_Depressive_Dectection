"""Local GPU Whisper-base transcription following HiQuE's released preprocessing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tag_and_merge(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match upstream literal punctuation tagging and chronological row merging."""
    tagged: list[dict] = []
    rows: list[dict] = []
    for segment in segments:
        speaker = 'Ellie' if segment['text'].endswith('?') else 'Participant'
        tagged.append(dict(segment, speaker=speaker))
        row = {key: segment[key] for key in ('start', 'end', 'text')}
        row['speaker'] = speaker
        if rows and rows[-1]['speaker'] == speaker:
            rows[-1]['text'] += ' ' + row['text']
            rows[-1]['end'] = row['end']
        else:
            rows.append(row)
    return tagged, rows


def official_ids(root: Path) -> list[int]:
    ids: list[int] = []
    for split in ('train', 'dev', 'test'):
        with (root / f'{split}_split_Depression_AVEC2017.csv').open() as handle:
            reader = csv.DictReader(handle)
            id_key = next(key for key in reader.fieldnames or [] if key.lower() == 'participant_id')
            ids.extend(int(row[id_key]) for row in reader if row[id_key].strip())
    if len(ids) != 189 or len(set(ids)) != 189:
        raise ValueError('Official roster must contain 189 unique participants')
    return sorted(set(ids) - {440})


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip-dir', type=Path, default=Path('Data/DAIC-WOZ'))
    parser.add_argument('--output-dir', type=Path, default=Path('Data/hique_reproduction_excluding440/asr'))
    parser.add_argument('--model-dir', type=Path, default=Path('.tmp/hique_weights/whisper'))
    parser.add_argument('--ids', type=int, nargs='+')
    args = parser.parse_args()
    import torch
    import whisper
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; run in GPU-accessible execution environment')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model('base', device='cuda', download_root=str(args.model_dir))
    model_path = args.model_dir / 'base.pt'
    model_hash = sha256(model_path)
    expected_hash = whisper._MODELS['base'].split('/')[-2]
    if model_hash != expected_hash:
        raise RuntimeError('Whisper weight integrity check failed')
    config = {'model': 'base', 'model_sha256': model_hash,
              'openai_whisper_version': importlib.metadata.version('openai-whisper'),
              'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'transcribe_options': {}, 'device': 'cuda', 'torch_threads': 4, 'seed': 42,
              'speaker_rule': "text.endswith('?')", 'merge': 'consecutive same speaker; one added space',
              'source_sha256': sha256(Path(__file__))}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = args.output_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing ASR configuration differs; use a new output directory')
    atomic_json(config_path, config)
    roster = official_ids(args.zip_dir)
    selected = args.ids or roster
    if set(selected) - set(roster):
        raise ValueError('Requested IDs outside official roster excluding 440')
    for pid in selected:
        target = args.output_dir / f'{pid}.json'
        archive = args.zip_dir / f'{pid}_P.zip'
        archive_stat = archive.stat()
        with zipfile.ZipFile(archive) as zf:
            member = zf.getinfo(f'{pid}_AUDIO.wav')
            identity = {'archive_size': archive_stat.st_size, 'archive_mtime_ns': archive_stat.st_mtime_ns,
                        'audio_zip_crc32': member.CRC, 'audio_size': member.file_size}
            if target.exists():
                existing = json.loads(target.read_text())
                if (existing['provenance']['config_sha256'] != config_hash
                        or existing['provenance']['input_identity'] != identity):
                    raise RuntimeError(f'{pid}: cached provenance mismatch')
                print(f'{pid}: cached', flush=True)
                continue
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix=f'hique-asr-{pid}-') as tmp:
                audio = Path(tmp) / f'{pid}_AUDIO.wav'
                with zf.open(member) as source, audio.open('wb') as sink:
                    shutil.copyfileobj(source, sink)
                audio_hash = sha256(audio)
                torch.manual_seed(42)
                result = model.transcribe(str(audio))
            tagged, rows = tag_and_merge(result['segments'])
            elapsed = time.monotonic() - started
            payload = {'participant_id': pid, 'language': result['language'], 'segments': tagged,
                       'rows': rows, 'provenance': {'config_sha256': config_hash,
                       'input_identity': identity, 'audio_sha256': audio_hash,
                       'model_sha256': model_hash, 'source_sha256': config['source_sha256'],
                       'seconds': elapsed}}
            atomic_json(target, payload)
            print(f'{pid}: {len(tagged)} segments, {sum(r["speaker"] == "Ellie" for r in rows)} Ellie rows, {elapsed:.1f}s', flush=True)
    completed = [pid for pid in roster if (args.output_dir / f'{pid}.json').exists()]
    atomic_json(args.output_dir / 'summary.json', {'completed_ids': completed,
                'completed_count': len(completed), 'expected_count': len(roster), 'config_sha256': config_hash})


if __name__ == '__main__':
    main()
