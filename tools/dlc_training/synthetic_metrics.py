"""Dense synthetic evaluation in the geometry's image-edge coordinate frame."""

import numpy as np


def evaluate_predictions(
    predictions, projected_xy, projection_valid, visibility, feature_names
):
    values = np.asarray(predictions)
    truth = np.asarray(projected_xy)
    if values.shape != (*truth.shape[:-1], 3) or truth.shape[-1] != 2:
        raise ValueError(
            "Expected predictions [image, feature, x/y/confidence] matching geometry xy"
        )
    if (
        projection_valid.shape != truth.shape[:-1]
        or visibility.shape != projection_valid.shape
        or len(feature_names) != truth.shape[1]
    ):
        raise ValueError(
            "Feature identities and masks must match the dense coordinate axes"
        )
    if not np.isfinite(truth[projection_valid]).all():
        raise ValueError("Valid geometry must contain finite coordinates")
    edge_xy = values[..., :2].astype(np.float64) + 0.5
    finite_xy = np.isfinite(edge_xy).all(axis=-1)
    finite_prediction = np.isfinite(values).all(axis=-1)
    errors = np.linalg.norm(edge_xy - truth, axis=-1)
    errors[~projection_valid | ~finite_xy] = np.nan

    def summarize(mask):
        count = int(mask.sum())
        distances = errors[mask & finite_xy]
        return {
            "target_count": count,
            "finite_coordinate_count": int((mask & finite_xy).sum()),
            "nonfinite_prediction_count": int((mask & ~finite_prediction).sum()),
            "mean_error_px": float(distances.mean()) if distances.size else None,
            "euclidean_rmse_px": float(np.sqrt(np.mean(distances**2)))
            if distances.size
            else None,
            "median_error_px": float(np.median(distances)) if distances.size else None,
            "p95_error_px": float(np.percentile(distances, 95))
            if distances.size
            else None,
            "pck": {
                str(threshold): float(
                    (mask & finite_prediction & (errors <= threshold)).sum() / count
                )
                if count
                else None
                for threshold in (5, 10, 20)
            },
        }

    groups = {
        "all_valid": projection_valid,
        "visible": projection_valid & visibility,
        "occluded": projection_valid & ~visibility,
    }
    summary = {"groups": {name: summarize(mask) for name, mask in groups.items()}}
    # Slice all arrays together when computing individual feature statistics.
    per_feature = []
    for index, name in enumerate(feature_names):
        feature_mask = np.zeros_like(projection_valid)
        feature_mask[:, index] = True
        per_feature.append(
            {
                "feature_name": str(name),
                **{
                    group: summarize(mask & feature_mask)
                    for group, mask in groups.items()
                },
            }
        )
    summary["per_feature"] = per_feature
    summary["confidence_filter_applied"] = False
    summary["projection_invalid_count"] = int((~projection_valid).sum())
    confidence_selected = projection_valid & finite_prediction & (values[..., 2] >= 0.6)
    summary["confidence_diagnostic"] = {
        "threshold": 0.6,
        "comparison": "confidence >= threshold; diagnostic only, primary results remain unfiltered",
        "included_count": int(confidence_selected.sum()),
        "excluded_valid_count": int((projection_valid & ~confidence_selected).sum()),
        "coverage": float(confidence_selected.sum() / projection_valid.sum())
        if projection_valid.any()
        else None,
        "statistics": summarize(confidence_selected),
    }
    return edge_xy, errors, finite_prediction, summary
