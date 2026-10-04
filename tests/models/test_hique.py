"""Checks of positional intervention, original flat parity, and training viability."""
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from urop.models.hique import build_model  # noqa: E402
from urop.models.upstream import DIMENSIONS, load_patched_upstream  # noqa: E402


def sample(condition):
    rng = np.random.default_rng(32)
    modalities = "T" if condition == "H_T" else "AVT"
    return [rng.normal(size=(2, 85, DIMENSIONS[m])).astype("float32")
            for m in modalities] + [np.tile(np.arange(85, dtype="int32"), (2, 1))]


@pytest.mark.parametrize("condition", ["H_AVT", "F_AVT", "H_T"])
def test_training_seed_and_weights(condition, tmp_path):
    model = build_model(condition, 23)
    inputs = sample(condition)
    initial = model(inputs, training=False).numpy()
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(np.eye(2), pred))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.isfinite(tf.convert_to_tensor(g).numpy()).all()
               for g in gradients)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    trained = model(inputs, training=False).numpy()
    assert np.isfinite(trained).all()
    assert not np.array_equal(initial, trained)
    np.testing.assert_allclose(trained.sum(axis=1), 1, atol=1e-6)
    path = tmp_path / "model.weights.h5"
    model.save_weights(path)
    rebuilt = build_model(condition, 23)
    np.testing.assert_array_equal(initial, rebuilt(inputs, training=False).numpy())
    rebuilt.load_weights(path)
    np.testing.assert_array_equal(trained, rebuilt(inputs, training=False).numpy())


def test_parent_intervention_and_capacity():
    hierarchy = build_model("H_AVT", 23)
    inputs = sample("H_AVT")
    flat_pred = hierarchy(inputs, training=False).numpy()
    changed = inputs[:-1] + [inputs[-1].copy()]
    changed[-1][:, 30:40] = 2
    assert np.max(np.abs(hierarchy(changed, training=False).numpy() - flat_pred)) > 1e-7
    flat = build_model("F_AVT", 23)
    assert flat.count_params() == hierarchy.count_params()
    np.testing.assert_array_equal(flat(inputs, training=False).numpy(), flat_pred)
    for a, b in zip(flat.get_weights(), hierarchy.get_weights()):
        np.testing.assert_array_equal(a, b)


def test_flat_matches_public_graph_with_equivalent_weights():
    model = build_model("F_AVT", 23)
    inputs = sample("F_AVT")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(23)
    public = load_patched_upstream().hique(85)
    # Match groups explicitly: the extra graph input may alter topological layer order.
    groups = (("FeatureEmbedding", "FeatureEmbedding"),
              ("SuppliedPositionEmbedding", "PositionEmbedding"),
              ("TransformerBlock", "TransformerBlock"),
              ("CrossTransformerBlock", "CrossTransformerBlock"), ("Dense", "Dense"))
    for new_type, old_type in groups:
        new = [x for x in model.layers if type(x).__name__ == new_type]
        old = [x for x in public.layers if type(x).__name__ == old_type]
        assert len(new) == len(old)
        for n, o in zip(new, old):
            n.set_weights(o.get_weights())
    assert model.count_params() == public.count_params()
    np.testing.assert_allclose(model(inputs, training=False).numpy(),
                               public(inputs[:-1], training=False).numpy(), atol=1e-7, rtol=1e-6)


def test_invalid_condition():
    with pytest.raises(ValueError):
        build_model("AVT", 1)


def test_upstream_adapter_has_no_filesystem_writes(monkeypatch):
    from pathlib import Path
    def unexpected(*args, **kwargs):
        raise AssertionError('Model construction must not write a manifest')
    monkeypatch.setattr(Path, 'write_text', unexpected)
    module = load_patched_upstream()
    assert module.patch_manifest['upstream_commit'] == '24c553bf2666b442ae5b0e3490b998a5d4493559'


def test_existing_checkpoint_keeps_recorded_dev_predictions():
    """Optional local artifact regression: never fit or write participant data."""
    from urop.paths import DATA_ROOT
    base = DATA_ROOT / 'hique_input_study_v3'
    checkpoint = base / 'runs/E2_seed42/best.weights.h5'
    if not checkpoint.exists():
        pytest.skip('Private saved checkpoint is not distributed with the repository')
    with np.load(base / 'features/E2.npz', allow_pickle=False) as data:
        indices = np.flatnonzero(data['split'] == 'dev')[:3]
        inputs = [data[m][indices].copy() for m in 'AVT']
        ids = data['participant_ids'][indices]
        for i, m in enumerate('AVT'):
            inputs[i][~data['mask_' + m][indices]] = 0
    inputs.append(np.tile(np.arange(85, dtype=np.int32), (3, 1)))
    with np.load(checkpoint.parent / 'dev_predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(ids, saved['participant_ids'][:3])
        expected = saved['probability'][:3].copy()
    model = build_model('F_AVT', 42)
    model.load_weights(str(checkpoint))
    actual = model(inputs, training=False).numpy()[:, 1]
    # Backend/CPU-vs-GPU kernels may differ slightly from the recorded run.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(actual > .5, expected > .5)
