"""Paper-based positional reimplementation; not an exact author implementation.

H_AVT/F_AVT take [A, V, T, position_ids]; H_T takes [T, position_ids].
The trainer supplies hierarchical or flat IDs. The graphs do not infer parents.
Rebuild and load_weights for serialization, as with the upstream adapter.
"""
from __future__ import annotations

from .upstream import DIMENSIONS, SEQUENCE_LENGTH, load_patched_upstream

CONDITIONS = ("H_AVT", "F_AVT", "H_T")


def build_model(condition: str, seed: int, learning_rate: float = 2e-4,
                dropout: float = 0.5):
    """Build capacity-matched H/F models with 85-entry per-modality position tables."""
    import tensorflow as tf

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    upstream = load_patched_upstream()
    layers = tf.keras.layers

    class SuppliedPositionEmbedding(upstream.PositionEmbedding):  # type: ignore[name-defined]
        # The base class is extracted dynamically from the hash-pinned public source.
        def call(self, inputs):
            features, position_ids = inputs
            return features + self.pos_emb(position_ids)

    modalities = "T" if condition == "H_T" else "AVT"
    inputs = [layers.Input((SEQUENCE_LENGTH, DIMENSIONS[m]), name=m)
              for m in modalities]
    positions = layers.Input((SEQUENCE_LENGTH,), dtype="int32", name="position_ids")
    embeddings = []
    for features in inputs:
        features = upstream.FeatureEmbedding(4)(features)
        embeddings.append(SuppliedPositionEmbedding(SEQUENCE_LENGTH, 4)([features, positions]))
    encoded = []
    for features in embeddings:
        features = upstream.TransformerBlock(4, 1, 4)(features)
        encoded.append(upstream.TransformerBlock(4, 1, 4)(features))
    if len(encoded) == 1:
        pooled = layers.GlobalAveragePooling1D()(encoded[0])
    else:
        branches = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            crossed = upstream.CrossTransformerBlock(4, 1, 4)(encoded[a], encoded[b])
            branches.append(layers.GlobalAveragePooling1D()(crossed))
        pooled = tf.concat(branches, axis=1)
    pooled = layers.Dropout(dropout)(pooled)
    prediction = layers.Dense(2, activation="softmax", name="depression")(pooled)
    model = tf.keras.Model(inputs=inputs + [positions], outputs=[prediction])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
