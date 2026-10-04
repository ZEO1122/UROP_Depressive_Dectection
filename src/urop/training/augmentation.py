"""Synchronized positive-class slot masking."""
import numpy as np

def augment(inputs, labels, seed, slots_to_mask=10):
    """Two extra positive copies; synchronized slot removal across modalities."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    sources = np.tile(positive, 2)
    outputs = [np.concatenate([x, x[sources].copy()]) for x in inputs]
    removed = []
    for index in range(len(sources)):
        slots = rng.choice(inputs[0].shape[1], slots_to_mask, replace=False)
        removed.append(slots.tolist())
        for array in outputs:
            array[len(labels) + index, slots] = 0
    return outputs, np.concatenate([labels, labels[sources]]), {'seed': seed, 'source_train_indices': sources.tolist(), 'removed_slots': removed}

