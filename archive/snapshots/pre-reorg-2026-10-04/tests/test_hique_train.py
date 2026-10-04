import csv
import numpy as np
import pytest
from experiments.hique_train import augment, fit_scaler, input_hashes, load_labels, metrics, transform


def test_scaler_uses_only_observed_training_slots():
    train = np.array([[[2., 5.], [4., 5.], [999., 999.]]])
    mean, std = fit_scaler(train, np.array([[True, True, False]]))
    np.testing.assert_allclose(mean, [3, 5])
    np.testing.assert_allclose(std, [1, 1])
    held = np.array([[[103., 5.], [777., 777.]]])
    result = transform(held, np.array([[True, False]]), mean, std)
    np.testing.assert_allclose(result, [[[100, 0], [0, 0]]])


def test_augmentation_tripled_positives_synchronized_and_reproducible():
    a = np.ones((3, 85, 2), dtype='float32')
    v = np.ones((3, 85, 4), dtype='float32') * 2
    labels = np.array([0, 1, 1])
    x, y, log = augment([a, v], labels, 13)
    assert y.tolist() == [0, 1, 1, 1, 1, 1, 1]
    assert len(log['removed_slots']) == 4
    np.testing.assert_array_equal(x[0][:3], a)
    for i in range(3, 7):
        assert (x[0][i, :, 0] == 0).sum() == 10
        np.testing.assert_array_equal(x[0][i, :, 0] == 0, x[1][i, :, 0] == 0)
    np.testing.assert_array_equal(x[0], augment([a, v], labels, 13)[0][0])
    assert a.sum() == 510


def test_released_binary_is_kept_and_test_outcomes_not_needed(tmp_path):
    for split, pid in [('train', 409), ('dev', 401), ('test', 301)]:
        path = tmp_path / f'{split}_split_Depression_AVEC2017.csv'
        with path.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score'])
            writer.writerow([pid, 1, 2])
    assert load_labels(tmp_path, 'train') == {409: 1, 401: 1}
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path, 'evaluate')
    (tmp_path / 'full_test_split.csv').write_text('Participant_ID,PHQ_Binary,PHQ_Score\n301,0,20\n,,\n999,1,20\n')
    assert load_labels(tmp_path, 'evaluate') == {301: 0}


def test_frozen_input_hashes_detect_outcome_and_protocol_changes(tmp_path):
    for name in ('train_split_Depression_AVEC2017.csv', 'dev_split_Depression_AVEC2017.csv',
                 'test_split_Depression_AVEC2017.csv', 'full_test_split.csv', 'protocol.json', 'manifest.json'):
        (tmp_path / name).write_text('unparsed original bytes')
    features = tmp_path / 'features.npz'
    before = input_hashes(features, tmp_path)
    for name in ('full_test_split.csv', 'protocol.json'):
        path = tmp_path / name
        path.write_text('changed bytes')
        after = input_hashes(features, tmp_path)
        assert after[str(path.resolve())] != before[str(path.resolve())]


def test_probability_metrics_and_half_threshold():
    result = metrics([0, 0, 1, 1], [.1, .5, .8, .3])
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert result['macro_f1'] == .5
    assert result['auroc'] == .75
    assert result['sensitivity'] == .5
    assert result['specificity'] == .5
