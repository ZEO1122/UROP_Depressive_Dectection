"""Load the pinned public HiQuE graph with explicit, recorded compatibility repairs.

Inputs are lists in canonical AVT order, of shape (batch, 85, dimension).
AVT calls the original hique function; ablations reuse its backbone and head.
Only weights serialization is supported; rebuild the graph before load_weights.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import linecache
from pathlib import Path
from types import ModuleType

DIMENSIONS = {"A": 88, "V": 272, "T": 768}
SEQUENCE_LENGTH = 85
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data/hique_reproduction/upstream/code/fusionmodel.py"
SOURCE_SHA256 = "5a50b9fd19822adfb64ac4aad9be4f734f3d93f7b6bfa4460ddfb0aeed1a7d04"
COMMIT = "24c553bf2666b442ae5b0e3490b998a5d4493559"


def canonical_modalities(modalities: str) -> str:
    if (
        not modalities
        or len(set(modalities)) != len(modalities)
        or set(modalities) - set(DIMENSIONS)
    ):
        raise ValueError("modalities must be a nonempty, nonrepeated subset of AVT")
    return "".join(m for m in DIMENSIONS if m in modalities)


def load_patched_upstream() -> ModuleType:
    """Extract only required definitions and fail closed if upstream has changed."""
    import tensorflow as tf

    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Pinned fusionmodel.py hash mismatch; re-audit before execution"
        )
    names = {
        "CrossTransformerBlock",
        "TransformerBlock",
        "PositionEmbedding",
        "FeatureEmbedding",
        "get_embeddings",
        "hique",
    }
    tree = ast.parse(source.decode())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    patches = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            init = next(
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            )
            args = [a.arg for a in init.args.args[1:]]
            # Scalar constructor configuration replaces references to nonexistent attrs.
            config = (
                "self._hique_config = {" + ", ".join(f"{a!r}: {a}" for a in args) + "}"
            )
            init.body.append(ast.parse(config).body[0])
            node.body = [
                n
                for n in node.body
                if not (isinstance(n, ast.FunctionDef) and n.name == "get_config")
            ]
            node.body.append(
                ast.parse(
                    "def get_config(self):\n    return {**super().get_config(), **self._hique_config}\n"
                ).body[0]
            )
            patches.append(
                f"{node.name}.get_config: return scalar constructor configuration"
            )
            for fn in node.body:
                if (
                    isinstance(fn, ast.FunctionDef)
                    and fn.name == "call"
                    and fn.args.args[-1].arg == "training"
                ):
                    fn.args.defaults = [ast.Constant(value=None)]
                    patches.append(f"{node.name}.call: training=None")
            if node.name == "TransformerBlock":
                fn = next(
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "call"
                )
                ret = fn.body[-1]
                assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple)
                ret.value = ret.value.elts[0]
                patches.append(
                    "TransformerBlock.call: return tensor only, not (tensor, attention_scores)"
                )
        elif isinstance(node, ast.FunctionDef) and node.name == "get_embeddings":
            node.body = [
                n
                for n in node.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "print"
                )
            ]
            patches.append("get_embeddings: remove diagnostic print")
    for descendant in ast.walk(tree):
        if (
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id == "Adam"
        ):
            for kw in descendant.keywords:
                if kw.arg == "lr":
                    kw.arg = "learning_rate"
    patches.extend(
        [
            "Adam(lr=): use learning_rate=",
            "Extract required definitions only; omit unused PIL/private TensorFlow imports and other models",
        ]
    )
    ast.fix_missing_locations(tree)
    module = ModuleType("hique_patched_public")
    module.__dict__.update(
        tf=tf, keras=tf.keras, layers=tf.keras.layers, Adam=tf.keras.optimizers.Adam
    )
    patched_source = ast.unparse(tree) + "\n"
    filename = "<hique_patched_public>"
    # AutoGraph inspects source, so expose the patched text rather than original lines.
    linecache.cache[filename] = (
        len(patched_source),
        None,
        patched_source.splitlines(True),
        filename,
    )
    exec(compile(patched_source, filename, "exec"), module.__dict__)
    manifest = {
        "upstream_commit": COMMIT,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "patched_definitions_sha256": hashlib.sha256(
            ast.unparse(tree).encode()
        ).hexdigest(),
        "patches": patches,
        "preserved": [
            "Shared layernorm1 across both cross-attention directions",
            "Attention dropout remains disabled as in upstream",
            "No question mask; learned positional embeddings on all 85 slots",
            "AVT original hique function, 4-wide conv embeddings, two self-attention blocks per modality, three cross-modal branches",
        ],
        "ablation_deviations": [
            "Single modality: original 4-wide two-block backbone, pooled dropout softmax head",
            "Two modalities: same backbone and single original cross-modal block",
            "These controlled deletions differ from separately named upstream ablation functions (8-wide, variable depth/dropout)",
        ],
        "serialization": "save_weights/load_weights only; rebuild with build_model",
    }
    path = ROOT / "Data/hique_reproduction/model_patch_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return module


def build_model(
    modalities: str, seed: int, learning_rate: float = 2e-4, dropout: float = 0.5
):
    """Return a compiled Keras classifier. Target: two-column categorical labels."""
    import tensorflow as tf

    modalities = canonical_modalities(modalities)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    upstream = load_patched_upstream()
    if modalities == "AVT":
        return upstream.hique(SEQUENCE_LENGTH, lr=learning_rate, dropout=dropout)
    layers = tf.keras.layers
    inputs = [layers.Input(shape=(SEQUENCE_LENGTH, DIMENSIONS[m])) for m in modalities]
    embeddings = [upstream.get_embeddings(x, 4) for x in inputs]
    encoded = []
    for x in embeddings:
        x = upstream.TransformerBlock(4, 1, 4)(x)
        encoded.append(upstream.TransformerBlock(4, 1, 4)(x))
    if len(encoded) == 1:
        pooled = layers.GlobalAveragePooling1D()(encoded[0])
    else:
        branches = [
            layers.GlobalAveragePooling1D()(
                upstream.CrossTransformerBlock(4, 1, 4)(a, b)
            )
            for a, b in itertools.combinations(encoded, 2)
        ]
        pooled = branches[0]
    pooled = layers.Dropout(dropout)(pooled)
    prediction = layers.Dense(2, activation="softmax", name="depression")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=[prediction])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
