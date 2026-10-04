from pathlib import Path
import numpy as np
import pytest
from urop.evaluation import test_features as extension
from urop.features import pipeline, text


def test_train_test_share_encoder():
    assert extension.extract_text is text.extract_text is pipeline.extract_text
    assert "full_test_split" not in Path(extension.__file__).read_text()


def test_fixed_cohort_arrays_and_missing_zero():
    arrays = {'participant_ids': np.arange(45), 'split': np.array(['test'] * 45)}
    for modality, width in [('A', 88), ('V', 272), ('T', 768)]:
        arrays[modality] = np.zeros((45, 85, width), dtype=np.float32)
        arrays['mask_' + modality] = np.zeros((45, 85), dtype=bool)
        arrays['mask_' + modality][:, 0] = True
    extension.validate_arrays(arrays)
    arrays['A'][0, 1, 0] = 1
    with pytest.raises(ValueError, match='nonzero missing'):
        extension.validate_arrays(arrays)


