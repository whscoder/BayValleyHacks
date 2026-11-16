"""Train and evaluate the production fact/opinion classifier."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import FeatureUnion, Pipeline


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fact_opinion"
    / "processed"
    / "fact_opinion.csv"
)
MODEL_PATH = DATA_PATH.with_name("fact_opinion_classifier.pkl")
METRICS_PATH = DATA_PATH.with_name("model_metrics.json")
LABELS = ["fact", "opinion", "mixed"]
CLASS_PRECISION_TARGETS = {"fact": 0.90, "opinion": 0.95, "mixed": 0.90}
MIN_MIXED_VALIDATION_ROWS = 100


def build_pipeline() -> Pipeline:
    """Build the unchanged production TF-IDF and logistic-regression pipeline."""
    tfidf = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", tfidf),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze legacy splits and honor the authoritative NewSSD partition."""
    newssd = frame[frame["source"].eq("newssd_eng")]
    legacy = frame[~frame["source"].eq("newssd_eng")]
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=2026)
    train_index, test_index = next(
        splitter.split(legacy["text"], legacy["label"], groups=legacy["group_id"])
    )
    train = legacy.iloc[train_index]
    test = legacy.iloc[test_index]

    validation_splitter = GroupShuffleSplit(
        test_size=0.2, n_splits=1, random_state=2027
    )
    fit_index, validation_index = next(
        validation_splitter.split(
            train["text"], train["label"], groups=train["group_id"]
        )
    )
    fit = train.iloc[fit_index]
    validation = train.iloc[validation_index]

    newssd_split = newssd["group_id"].str.extract(
        r"^newssd_eng:(train|validation|test):", expand=False
    )
    if newssd_split.isna().any():
        raise ValueError("NewSSD group_id must encode train, validation, or test split")
    return (
        pd.concat([fit, newssd.loc[newssd_split.eq("train")]]),
        pd.concat([validation, newssd.loc[newssd_split.eq("validation")]]),
        pd.concat([test, newssd.loc[newssd_split.eq("test")]]),
    )


def select_class_thresholds(
    expected: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Choose the lowest per-class confidence meeting its precision target."""
    evaluated_labels = [label for label in LABELS if label in set(classes)]
    if (
        "mixed" in evaluated_labels
        and int(expected.eq("mixed").sum()) < MIN_MIXED_VALIDATION_ROWS
    ):
        raise ValueError(
            f"mixed requires at least {MIN_MIXED_VALIDATION_ROWS} validation rows"
        )
    confidence = probabilities.max(axis=1)
    thresholds: dict[str, float] = {}
    for label in evaluated_labels:
        target = CLASS_PRECISION_TARGETS[label]
        # A threshold above the probability range safely disables local acceptance
        # when the validation precision requirement cannot be demonstrated.
        thresholds[label] = 1.01
        predicted_label = predicted == label
        for threshold in np.arange(0.50, 0.96, 0.01):
            accepted = predicted_label & (confidence >= threshold)
            if not accepted.any():
                continue
            precision = float(expected.loc[accepted].eq(label).mean())
            if precision >= target:
                thresholds[label] = float(round(threshold, 2))
                break
    return thresholds


def binary_baseline_report(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    classifier_c: float,
) -> dict:
    """Retrain the binary baseline on the identical groups for comparison."""
    map_label = lambda values: values.replace({"mixed": "fact"})
    model = build_pipeline().set_params(classifier__C=classifier_c)
    model.fit(fit["text"], map_label(fit["label"]))
    validation_probabilities = model.predict_proba(validation["text"])
    validation_predictions = model.classes_[validation_probabilities.argmax(axis=1)]
    thresholds = select_class_thresholds(
        map_label(validation["label"]),
        validation_predictions,
        validation_probabilities,
        model.classes_,
    )
    train = pd.concat([fit, validation]).sort_index()
    model.fit(train["text"], map_label(train["label"]))
    probabilities = model.predict_proba(test["text"])
    predictions = model.classes_[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    accepted = accepted_mask(predictions, confidence, thresholds)
    expected = map_label(test["label"])
    return {
        "class_thresholds": thresholds,
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(f1_score(expected, predictions, average="macro")),
        "fallback_rate": float((~accepted).mean()),
        "safety_mapped": safety_metrics(test["label"], predictions),
    }


def accepted_mask(
    predicted: np.ndarray, confidence: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    return np.asarray(
        [score >= thresholds[str(label)] for label, score in zip(predicted, confidence)]
    )


def _group_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    return len(set(left["group_id"]) & set(right["group_id"]))


def split_metrics(
    fit: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Describe split sizes and fail if related rows cross a split boundary."""
    overlaps = {
        "fit_validation": _group_overlap(fit, validation),
        "fit_test": _group_overlap(fit, test),
        "validation_test": _group_overlap(validation, test),
    }
    if any(overlaps.values()):
        raise AssertionError(f"group_id overlap detected: {overlaps}")
    return {
        "rows": {
            "fit": int(len(fit)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "non_test_total": int(len(fit) + len(validation)),
            "dataset_total": int(len(fit) + len(validation) + len(test)),
        },
        "groups": {
            "fit": int(fit["group_id"].nunique()),
            "validation": int(validation["group_id"].nunique()),
            "test": int(test["group_id"].nunique()),
        },
        "group_overlap_counts": overlaps,
        "group_isolation_verified": True,
    }


def evaluation_metrics(expected: pd.Series, predicted: np.ndarray) -> dict:
    """Return JSON-safe overall and per-label classification metrics."""
    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "rows": int(len(expected)),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=LABELS, average="macro")),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=LABELS
        ).astype(int).tolist(),
        "per_label": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def safety_metrics(expected: pd.Series, predicted: np.ndarray) -> dict:
    """Score the safety decision: research fact/mixed and exclude opinion."""
    expected_route = expected.replace(
        {"fact": "research", "mixed": "research", "opinion": "exclude"}
    )
    predicted_route = pd.Series(predicted, index=expected.index).replace(
        {"fact": "research", "mixed": "research", "opinion": "exclude"}
    )
    return {
        "macro_f1": float(f1_score(expected_route, predicted_route, average="macro")),
        "factual_exclusion_rows": int(
            (expected.isin(["fact", "mixed"]) & predicted_route.eq("exclude")).sum()
        ),
        "factual_exclusion_rate": float(
            (
                expected.isin(["fact", "mixed"]) & predicted_route.eq("exclude")
            ).sum()
            / max(1, expected.isin(["fact", "mixed"]).sum())
        ),
    }


def build_metrics_report(
    *,
    frame: pd.DataFrame,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_predictions: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    class_thresholds: dict[str, float],
    best_params: dict,
    binary_baseline: dict,
) -> dict:
    accepted = accepted_mask(predictions, confidence, class_thresholds)
    per_source = {}
    for source in sorted(test["source"].astype(str).unique()):
        source_mask = test["source"].astype(str).to_numpy() == source
        per_source[source] = evaluation_metrics(
            test.loc[source_mask, "label"], predictions[source_mask]
        )

    accepted_metrics = (
        evaluation_metrics(test.loc[accepted, "label"], predictions[accepted])
        if accepted.any()
        else None
    )
    return {
        "dataset": str(DATA_PATH),
        "model": str(MODEL_PATH),
        "dataset_rows": int(len(frame)),
        "dataset_label_counts": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "training_rows": int(len(fit) + len(validation)),
        "training_label_counts": {
            str(label): int(count)
            for label, count in pd.concat([fit, validation])["label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "split": split_metrics(fit, validation, test),
        "best_parameters": best_params,
        "binary_baseline": binary_baseline,
        "confidence_threshold": float(class_thresholds["opinion"]),
        "class_thresholds": class_thresholds,
        "validation": evaluation_metrics(
            validation["label"], validation_predictions
        ),
        "test": evaluation_metrics(test["label"], predictions),
        "safety_mapped_test": safety_metrics(test["label"], predictions),
        "per_source_test": per_source,
        "threshold_performance": {
            "accepted_rows": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "fallback_rows": int((~accepted).sum()),
            "fallback_rate": float((~accepted).mean()),
            "accepted": accepted_metrics,
            "factual_exclusion_rows": int(
                (
                    accepted
                    & (predictions == "opinion")
                    & test["label"].isin(["fact", "mixed"]).to_numpy()
                ).sum()
            ),
        },
    }


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Training requires approved fact, opinion, and mixed rows")
    fit, validation, test = split_dataset(frame)
    train = pd.concat([fit, validation]).sort_index()
    split_metrics(fit, validation, test)

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "classifier__C": [2.0, 4.0, 8.0],
        },
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
    )
    search.fit(fit["text"], fit["label"], groups=fit["group_id"])

    validation_probabilities = search.predict_proba(validation["text"])
    validation_predictions = search.classes_[validation_probabilities.argmax(axis=1)]
    class_thresholds = select_class_thresholds(
        validation["label"],
        validation_predictions,
        validation_probabilities,
        search.classes_,
    )

    model = clone(search.best_estimator_)
    model.fit(train["text"], train["label"])
    model.class_thresholds_ = class_thresholds
    model.confidence_threshold_ = class_thresholds["opinion"]

    test_probabilities = model.predict_proba(test["text"])
    test_predictions = model.classes_[test_probabilities.argmax(axis=1)]
    test_confidence = test_probabilities.max(axis=1)
    accepted = accepted_mask(test_predictions, test_confidence, class_thresholds)
    baseline = binary_baseline_report(
        fit,
        validation,
        test,
        classifier_c=float(search.best_params_["classifier__C"]),
    )

    report = build_metrics_report(
        frame=frame,
        fit=fit,
        validation=validation,
        test=test,
        validation_predictions=validation_predictions,
        predictions=test_predictions,
        confidence=test_confidence,
        class_thresholds=class_thresholds,
        best_params=search.best_params_,
        binary_baseline=baseline,
    )
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    METRICS_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("Best parameters:", search.best_params_)
    print("Class thresholds:", class_thresholds)
    print("OpenAI fallback rate:", f"{1 - accepted.mean():.1%}")
    print("\nAll test predictions")
    print(classification_report(test["label"], test_predictions, digits=3))
    print("Accepted local predictions")
    print(classification_report(test["label"][accepted], test_predictions[accepted], digits=3))
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
