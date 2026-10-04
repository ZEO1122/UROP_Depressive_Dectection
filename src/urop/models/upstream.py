"""Load the pinned public HiQuE graph with explicit, recorded compatibility repairs.

Only the required public definitions are loaded; module.patch_manifest records repairs.
Loading never writes to the historical artifact directory.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
from urop.paths import ROOT, UPSTREAM_SOURCE
from types import ModuleType

DIMENSIONS = {"A": 88, "V": 272, "T": 768}
SEQUENCE_LENGTH = 85
SOURCE = UPSTREAM_SOURCE
SOURCE_SHA256 = "5a50b9fd19822adfb64ac4aad9be4f734f3d93f7b6bfa4460ddfb0aeed1a7d04"
COMMIT = "24c553bf2666b442ae5b0e3490b998a5d4493559"


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
    module.__dict__["patch_manifest"] = manifest
    return module


