import io
import json
import zipfile

import numpy as np
import pytest

from urop.features.av import audio_samples, extract_one, visual_features


def test_one_frame_has_zero_population_variance():
    values, usable, coverage, count = visual_features([2.], np.ones((1, 136)), [True], [[2., 3.]])
    assert usable and count == 1
    assert coverage == 0  # One timestamp cannot establish a duration.
    np.testing.assert_array_equal(values[:136], 1)
    np.testing.assert_array_equal(values[136:], 0)


def test_low_coverage_and_short_audio_retained():
    samples, usable, coverage = audio_samples(np.ones(1600), 16000, [[0, 2]])
    assert usable and len(samples) == 1600
    assert coverage == pytest.approx(.05)
    _, usable, coverage, count = visual_features([0., .1, .2], np.ones((3, 136)),
                                                [True, False, False], [[0, 2]])
    assert usable and count == 1 and coverage == pytest.approx(.05)


def test_truncated_visual_no_extrapolation_402():
    times = np.arange(30) / 30 + 832
    _, usable, coverage, count = visual_features(times, np.ones((30, 136)),
                                                np.ones(30, bool), [[832, 955]])
    assert usable and count == 1 and coverage == pytest.approx(1 / 123)
    values, usable, coverage, count = visual_features(times, np.ones((30, 136)),
                                                     np.ones(30, bool), [[834, 955]])
    assert not usable and coverage == 0 and count == 0 and not values.any()


def test_first_successful_finite_frame_each_absolute_second():
    xy = np.repeat(np.arange(5)[:, None], 136, axis=1).astype(float)
    xy[1] = np.nan
    values, usable, _, count = visual_features([.1, .2, .3, 1.1, 1.2], xy,
                                              [False, True, True, True, True], [[0, 2]])
    assert usable and count == 2
    np.testing.assert_array_equal(values[:136], 2.5)
    np.testing.assert_array_equal(values[136:], .25)


def test_independent_modalities_and_cache(tmp_path):
    pytest.importorskip('opensmile')
    sf = pytest.importorskip('soundfile')
    audio = io.BytesIO()
    sf.write(audio, np.sin(np.arange(1600) * .05), 16000, format='WAV')
    header = ','.join(['timestamp', 'success', 'confidence'] + [f'{axis}{i}' for axis in 'xy' for i in range(68)])
    visual = header + '\n' + ','.join(['2', '1', '0.1'] + ['1'] * 136) + '\n'
    with zipfile.ZipFile(tmp_path / '301_P.zip', 'w') as z:
        z.writestr('301_AUDIO.wav', audio.getvalue())
        z.writestr('301_CLNF_features.txt', visual)
    record = {'participant_id': 301, 'split': 'train', 'slots': [
        {'slot': 0, 'text': 'a', 'spans': [[0, .1]]},
        {'slot': 1, 'text': 'b', 'spans': [[2, 3]]}]}
    output = tmp_path / 'out'
    task = (record, str(tmp_path), str(output))
    assert not extract_one(task)['cached']
    with np.load(output / '301.npz') as data:
        assert data['mask_A'][0] and not data['mask_V'][0]
        assert data['mask_V'][1] and not data['mask_A'][1]
        assert np.isfinite(data['A']).all()
    assert extract_one(task)['cached']
    (output / '301.npz').write_bytes(b'corrupt')
    assert not extract_one(task)['cached']
    assert json.loads((output / '301.json').read_text())['npz_sha256']
