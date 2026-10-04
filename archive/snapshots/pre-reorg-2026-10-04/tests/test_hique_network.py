"""Executable TensorFlow smoke checks; global Python skips optional TF dependency."""

import os

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from experiments.hique_network import (  # noqa: E402
    DIMENSIONS,
    SEQUENCE_LENGTH,
    build_model,
    canonical_modalities,
    load_patched_upstream,
)


def sample(modalities):
    rng = np.random.default_rng(32)
    return [
        rng.normal(size=(2, SEQUENCE_LENGTH, DIMENSIONS[m])).astype("float32")
        for m in canonical_modalities(modalities)
    ]


@pytest.mark.parametrize("modalities", ["A", "V", "T", "AV", "AT", "VT", "AVT"])
def test_train_and_weights_roundtrip(modalities, tmp_path):
    model = build_model(modalities, seed=23)
    inputs = sample(modalities)
    initial = model(inputs, training=False).numpy()
    weights_before = [w.copy() for w in model.get_weights()]
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(np.eye(2), pred))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(
        g is not None and np.isfinite(tf.convert_to_tensor(g).numpy()).all()
        for g in gradients
    )
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    assert any(
        not np.array_equal(a, b) for a, b in zip(weights_before, model.get_weights())
    )
    trained = model(inputs, training=False).numpy()
    assert np.isfinite(trained).all()
    np.testing.assert_allclose(trained.sum(axis=1), 1, atol=1e-6)
    assert model.count_params() > 0
    path = tmp_path / "model.weights.h5"
    model.save_weights(path)
    rebuilt = build_model(modalities, seed=23)
    np.testing.assert_array_equal(initial, rebuilt(inputs, training=False).numpy())
    rebuilt.load_weights(path)
    np.testing.assert_array_equal(trained, rebuilt(inputs, training=False).numpy())


def test_avt_public_graph_parity_and_input_sensitivity():
    adapter = build_model("AVT", 7)
    x = sample("AVT")
    expected = adapter(x, training=False).numpy()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)
    public = load_patched_upstream().hique(85)
    np.testing.assert_array_equal(expected, public(x, training=False).numpy())
    assert adapter.count_params() == public.count_params()
    for i in range(3):
        changed = [v.copy() for v in x]
        changed[i] = changed[i][::-1].copy()
        assert np.max(np.abs(public(changed, training=False).numpy() - expected)) > 1e-7
    for layer in public.layers:
        layer.get_config()


def test_invalid_modalities():
    for invalid in ("", "AA", "X", "avt"):
        with pytest.raises(ValueError):
            canonical_modalities(invalid)
    assert canonical_modalities("TVA") == "AVT"
