import numpy as np

from wfr_guard.feature_guard import FeatureGuard


def test_feature_guard_flags_distant_sample():
    rng = np.random.default_rng(7)
    class_zero_center = np.array([1.0] + [0.0] * 15)
    class_one_center = np.array([-1.0] + [0.0] * 15)
    class_zero = class_zero_center + rng.normal(0, 0.02, size=(30, 16))
    class_one = class_one_center + rng.normal(0, 0.02, size=(30, 16))
    features = np.vstack([class_zero, class_one])
    labels = np.array([0] * 30 + [1] * 30)

    guard = FeatureGuard("secret", input_dim=16, projection_dim=8).fit(
        features, labels
    )
    suspect = np.array([[0.0, 8.0] + [0.0] * 14])
    _, flagged = guard.score(suspect, np.array([0]))
    assert flagged[0]
