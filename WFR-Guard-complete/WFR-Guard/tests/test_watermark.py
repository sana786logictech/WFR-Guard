import numpy as np
from PIL import Image

from wfr_guard.watermark import embed, verify


def test_valid_watermark_is_verified():
    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8))
    protected = embed(image, "test-key", "sample-1", "class-a", repetitions=3)
    result = verify(
        protected, "test-key", "sample-1", "class-a", repetitions=3, threshold=0.90
    )
    assert result.valid
    assert result.match_rate == 1.0


def test_changed_label_fails_verification():
    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8))
    protected = embed(image, "test-key", "sample-1", "class-a", repetitions=3)
    result = verify(
        protected, "test-key", "sample-1", "class-b", repetitions=3, threshold=0.80
    )
    assert not result.valid

