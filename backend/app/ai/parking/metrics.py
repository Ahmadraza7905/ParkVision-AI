from typing import Any


def calculate_occupancy_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate binary occupancy classification metrics."""

    truth = {
        item["id"]: bool(item["occupied"])
        for item in ground_truth
    }

    predicted = {
        item["id"]: bool(item["occupied"])
        for item in predictions
    }

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for space_id, actual in truth.items():

        prediction = predicted.get(
            space_id,
            False,
        )

        if actual and prediction:
            true_positive += 1

        elif not actual and not prediction:
            true_negative += 1

        elif not actual and prediction:
            false_positive += 1

        elif actual and not prediction:
            false_negative += 1

    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
