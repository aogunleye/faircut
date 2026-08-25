import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate

def test_disparate_impact_metric():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0])
    sensitive_features = np.array([1, 1, 1, 0, 0, 0])
    
    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    assert metric_frame.by_group.notnull().all()