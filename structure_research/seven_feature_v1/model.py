"""Seven-dimensional preprocessing and consensus HDBSCAN utilities.

This module deliberately contains no material names, acceptance targets, scoring
weights, PCA, UMAP, or imputation.  It implements only the preregistered
seven-feature model transform and density-clustering ensemble.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import PowerTransformer, RobustScaler

from .common import MODEL_FEATURES, validate_config


ArrayLike = np.ndarray | Sequence[Sequence[float]]
ClustererFactory = Callable[..., Any]
ApproximatePredict = Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]
ValidityFunction = Callable[..., float]


def _boolean_mask(mask: Sequence[bool] | np.ndarray, n_rows: int, name: str) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{name} must be a one-dimensional boolean mask of length {n_rows}")
    if not np.issubdtype(values.dtype, np.bool_):
        raise TypeError(f"{name} must have boolean dtype")
    return values.astype(bool, copy=True)


def _finite_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("seven-feature input must be a pandas DataFrame")
    missing_columns = [name for name in MODEL_FEATURES if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing required model feature columns: {missing_columns}")

    selected = frame.loc[:, list(MODEL_FEATURES)]
    try:
        values = selected.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("all seven model features must be numeric") from exc
    finite = np.isfinite(values)
    if not np.all(finite):
        bad_counts = {
            MODEL_FEATURES[column]: int(np.count_nonzero(~finite[:, column]))
            for column in range(len(MODEL_FEATURES))
            if np.any(~finite[:, column])
        }
        raise ValueError(
            "missing or non-finite seven-feature values are forbidden; "
            f"no imputation is performed: {bad_counts}"
        )
    return values


def _finite_7d_array(matrix: ArrayLike, name: str = "X") -> np.ndarray:
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric two-dimensional array") from exc
    if values.ndim != 2 or values.shape[1] != len(MODEL_FEATURES):
        raise ValueError(
            f"{name} must have shape (n_rows, {len(MODEL_FEATURES)}); "
            f"found {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains missing or non-finite values")
    return np.asarray(values, dtype=float)


def _preprocessing_settings(config: Mapping[str, Any]) -> tuple[float | None, tuple[float, float]]:
    validate_config(dict(config))
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("preprocessing configuration is required")
    if preprocessing.get("D_transform") != "robust_scale_only":
        raise ValueError("feature__D must use RobustScaler only")
    if preprocessing.get("continuous_transform") != "yeo_johnson_then_robust_scale":
        raise ValueError(
            "the six continuous features must use Yeo-Johnson then RobustScaler"
        )
    if preprocessing.get("power_transform_standardize") is not False:
        raise ValueError("PowerTransformer must use standardize=False")
    if preprocessing.get("feature_weighting") != "equal_per_feature":
        raise ValueError("seven-feature clustering requires equal per-feature weighting")
    if preprocessing.get("PCA_for_clustering") is not False:
        raise ValueError("PCA is forbidden for seven-feature clustering")

    clip_raw = preprocessing.get("robust_scaled_clip")
    if clip_raw is None:
        clip_value = None
    else:
        clip_value = float(clip_raw)
        if not np.isfinite(clip_value) or clip_value <= 0:
            raise ValueError("robust_scaled_clip must be null or a positive finite number")

    quantiles_raw = preprocessing.get("robust_quantile_range", (25.0, 75.0))
    if not isinstance(quantiles_raw, Sequence) or len(quantiles_raw) != 2:
        raise ValueError("robust_quantile_range must contain exactly two quantiles")
    quantile_range = (float(quantiles_raw[0]), float(quantiles_raw[1]))
    if not (0.0 <= quantile_range[0] < quantile_range[1] <= 100.0):
        raise ValueError("robust_quantile_range must satisfy 0 <= low < high <= 100")
    return clip_value, quantile_range


@dataclass
class SevenFeatureTransform:
    """Fitted, joblib-serializable transform for the fixed seven columns.

    ``feature__D`` is scaled directly.  The remaining six columns are first
    Yeo-Johnson transformed (without internal standardization) and then robustly
    scaled.  Clipping, when configured, is applied after both branches are put
    back into the original seven-column order.
    """

    feature_names: tuple[str, ...]
    d_scaler: RobustScaler
    continuous_power: PowerTransformer
    continuous_scaler: RobustScaler
    clip_value: float | None
    quantile_range: tuple[float, float]
    n_fit_rows: int

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        fit_mask: Sequence[bool] | np.ndarray,
        config: Mapping[str, Any],
    ) -> "SevenFeatureTransform":
        """Fit only on rows selected by ``fit_mask``; never impute values."""

        clip_value, quantile_range = _preprocessing_settings(config)
        values = _finite_feature_matrix(frame)
        mask = _boolean_mask(fit_mask, len(values), "fit_mask")
        if np.count_nonzero(mask) < 2:
            raise ValueError("at least two fit rows are required for Yeo-Johnson fitting")

        fit_values = values[mask]
        continuous = fit_values[:, 1:]
        constant = np.ptp(continuous, axis=0) == 0.0
        if np.any(constant):
            names = [MODEL_FEATURES[index + 1] for index in np.flatnonzero(constant)]
            raise ValueError(
                "Yeo-Johnson fit columns must vary within fit_mask; "
                f"constant columns: {names}"
            )

        d_scaler = RobustScaler(quantile_range=quantile_range)
        d_scaler.fit(fit_values[:, [0]])

        continuous_power = PowerTransformer(method="yeo-johnson", standardize=False)
        powered = continuous_power.fit_transform(continuous)
        continuous_scaler = RobustScaler(quantile_range=quantile_range)
        continuous_scaler.fit(powered)

        return cls(
            feature_names=tuple(MODEL_FEATURES),
            d_scaler=d_scaler,
            continuous_power=continuous_power,
            continuous_scaler=continuous_scaler,
            clip_value=clip_value,
            quantile_range=quantile_range,
            n_fit_rows=int(np.count_nonzero(mask)),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform complete rows into the ordered, equally weighted 7D space."""

        if tuple(self.feature_names) != tuple(MODEL_FEATURES):
            raise ValueError("fitted transform feature order does not match MODEL_FEATURES")
        values = _finite_feature_matrix(frame)
        d_values = self.d_scaler.transform(values[:, [0]])
        continuous = self.continuous_power.transform(values[:, 1:])
        continuous = self.continuous_scaler.transform(continuous)
        transformed = np.column_stack((d_values[:, 0], continuous))
        if self.clip_value is not None:
            transformed = np.clip(
                transformed, -float(self.clip_value), float(self.clip_value)
            )
        if transformed.shape != (len(frame), len(MODEL_FEATURES)):
            raise RuntimeError("seven-feature transform unexpectedly changed dimensionality")
        if not np.all(np.isfinite(transformed)):
            raise ValueError("seven-feature transform produced non-finite values")
        return transformed

    def get_feature_names_out(self) -> np.ndarray:
        """Return feature names in the exact transform output order."""

        return np.asarray(self.feature_names, dtype=object)


@dataclass
class HDBSCANEnsemble:
    """Fitted 30-run ensemble and fit-row consensus state."""

    fit_mask: np.ndarray
    labels: np.ndarray
    consensus: np.ndarray
    params: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    clusterers: tuple[Any, ...]
    consensus_threshold: float
    feature_names: tuple[str, ...] = tuple(MODEL_FEATURES)

    @property
    def fit_indices(self) -> np.ndarray:
        return np.flatnonzero(self.fit_mask)

    @property
    def fit_labels(self) -> np.ndarray:
        return self.labels[self.fit_mask]

    @property
    def n_runs(self) -> int:
        return len(self.params)

    def diagnostics_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(row) for row in self.diagnostics])


@dataclass
class HDBSCANPrediction:
    """Per-run approximate predictions for selected query rows."""

    labels: np.ndarray
    strengths: np.ndarray
    predict_mask: np.ndarray


def _hdbscan_grid(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    validate_config(dict(config))
    hcfg = config.get("hdbscan")
    if not isinstance(hcfg, Mapping):
        raise ValueError("hdbscan configuration is required")
    if hcfg.get("use_all_preregistered_runs_equally") is not True:
        raise ValueError("all preregistered HDBSCAN runs must be used equally")

    try:
        min_sizes = [int(value) for value in hcfg["min_cluster_size"]]
        min_samples_values = [int(value) for value in hcfg["min_samples"]]
        methods = [str(value) for value in hcfg["cluster_selection_method"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid HDBSCAN grid configuration") from exc
    metric = str(hcfg.get("metric", "euclidean"))
    if any(value < 2 for value in min_sizes):
        raise ValueError("min_cluster_size values must be at least 2")
    if any(value < 1 for value in min_samples_values):
        raise ValueError("min_samples values must be positive")
    if any(value not in {"eom", "leaf"} for value in methods):
        raise ValueError("cluster_selection_method must contain only eom or leaf")

    grid = tuple(
        {
            "run_index": run_index,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "metric": metric,
        }
        for run_index, (min_cluster_size, min_samples, method) in enumerate(
            itertools.product(min_sizes, min_samples_values, methods)
        )
    )
    unique_specs = {
        (
            item["min_cluster_size"],
            item["min_samples"],
            item["cluster_selection_method"],
            item["metric"],
        )
        for item in grid
    }
    if len(grid) != 30 or len(unique_specs) != 30:
        raise ValueError(
            "seven_feature_v1 requires exactly 30 distinct HDBSCAN grid runs"
        )
    return grid


def _default_clusterer_factory(**kwargs: Any) -> Any:
    import hdbscan

    return hdbscan.HDBSCAN(**kwargs)


def _default_approximate_predict(
    clusterer: Any, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import hdbscan

    return hdbscan.approximate_predict(clusterer, values)


def _optional_validity_function() -> ValidityFunction | None:
    try:
        from hdbscan.validity import validity_index
    except (ImportError, AttributeError):
        return None
    return validity_index


def _integer_labels(labels: Any, n_rows: int, name: str) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(f"{name} must contain one label per row")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer cluster labels") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain finite integer cluster labels")
    integer = numeric.astype(int)
    if np.any(integer < -1):
        raise ValueError(f"{name} contains labels below the HDBSCAN noise label -1")
    return integer


def consensus_matrix(fit_labels: np.ndarray) -> np.ndarray:
    """Return equal-run co-clustering frequencies for fitted rows.

    Two distinct noise rows never co-cluster.  The diagonal is set to one as the
    conventional self-similarity value, including for rows labelled noise in
    every run.
    """

    raw = np.asarray(fit_labels)
    if raw.ndim != 2:
        raise ValueError("fit_labels must have shape (n_fit_rows, n_runs)")
    n_rows, n_runs = raw.shape
    if n_runs < 1:
        raise ValueError("at least one clustering run is required")
    labels = np.column_stack(
        [_integer_labels(raw[:, run], n_rows, f"fit_labels[:, {run}]") for run in range(n_runs)]
    )
    counts = np.zeros((n_rows, n_rows), dtype=np.uint32)
    for run in range(n_runs):
        run_labels = labels[:, run]
        for cluster_id in np.unique(run_labels[run_labels >= 0]):
            members = np.flatnonzero(run_labels == cluster_id)
            counts[np.ix_(members, members)] += 1
    consensus = counts.astype(float) / float(n_runs)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def _cluster_diagnostics(
    labels: np.ndarray,
    matrix: np.ndarray,
    spec: Mapping[str, Any],
    validity_fn: ValidityFunction | None,
) -> dict[str, Any]:
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    n_clusters = int(len(cluster_ids))
    n_noise = int(np.count_nonzero(labels < 0))
    largest = int(np.max(counts)) if len(counts) else 0
    diagnostic: dict[str, Any] = {
        **dict(spec),
        "n_fit_rows": int(len(labels)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / len(labels)) if len(labels) else float("nan"),
        "largest_cluster_size": largest,
        "largest_cluster_fraction": (
            float(largest / len(labels)) if len(labels) else float("nan")
        ),
        "dbcv": float("nan"),
        "dbcv_status": "unavailable" if validity_fn is None else "not_applicable",
    }
    if validity_fn is not None and n_clusters >= 2:
        try:
            value = float(validity_fn(matrix, labels, metric=spec["metric"]))
            if not np.isfinite(value):
                raise ValueError("non-finite DBCV")
            diagnostic["dbcv"] = value
            diagnostic["dbcv_status"] = "ok"
        except Exception as exc:  # DBCV is diagnostic and must not abort a fitted run.
            diagnostic["dbcv_status"] = f"failed:{type(exc).__name__}"
    return diagnostic


def fit_hdbscan_ensemble(
    X: ArrayLike,
    fit_mask: Sequence[bool] | np.ndarray,
    config: Mapping[str, Any],
    *,
    clusterer_factory: ClustererFactory | None = None,
    validity_fn: ValidityFunction | None = None,
) -> HDBSCANEnsemble:
    """Fit all 30 preregistered HDBSCAN runs on ``fit_mask`` rows only."""

    values = _finite_7d_array(X)
    mask = _boolean_mask(fit_mask, len(values), "fit_mask")
    if not np.any(mask):
        raise ValueError("fit_mask selects no rows")
    fit_values = values[mask]
    grid = _hdbscan_grid(config)
    factory = clusterer_factory or _default_clusterer_factory
    resolved_validity = validity_fn if validity_fn is not None else _optional_validity_function()

    labels = np.full((len(values), len(grid)), -1, dtype=int)
    clusterers: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    for run, spec in enumerate(grid):
        estimator_kwargs = {
            "min_cluster_size": int(spec["min_cluster_size"]),
            "min_samples": int(spec["min_samples"]),
            "cluster_selection_method": str(spec["cluster_selection_method"]),
            "metric": str(spec["metric"]),
            "prediction_data": True,
            "core_dist_n_jobs": 1,
        }
        clusterer = factory(**estimator_kwargs)
        run_labels = _integer_labels(
            clusterer.fit_predict(fit_values), len(fit_values), f"run {run} labels"
        )
        labels[mask, run] = run_labels
        clusterers.append(clusterer)
        diagnostics.append(
            _cluster_diagnostics(run_labels, fit_values, spec, resolved_validity)
        )

    threshold = float(config["hdbscan"]["consensus_threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("consensus_threshold must be between zero and one")
    return HDBSCANEnsemble(
        fit_mask=mask,
        labels=labels,
        consensus=consensus_matrix(labels[mask]),
        params=tuple(dict(item) for item in grid),
        diagnostics=tuple(diagnostics),
        clusterers=tuple(clusterers),
        consensus_threshold=threshold,
    )


def predict_hdbscan_ensemble(
    ensemble: HDBSCANEnsemble,
    X: ArrayLike,
    predict_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    approximate_predict_fn: ApproximatePredict | None = None,
) -> HDBSCANPrediction:
    """Apply every frozen clusterer with ``approximate_predict`` only.

    Output matrices have one row per query row.  Unselected rows retain label
    ``-1`` and strength ``NaN`` so callers can merge predictions with the
    ensemble's stored fit-row labels without confusing unselected rows with a
    fitted result.
    """

    values = _finite_7d_array(X)
    if tuple(ensemble.feature_names) != tuple(MODEL_FEATURES):
        raise ValueError("ensemble feature order does not match MODEL_FEATURES")
    if len(ensemble.clusterers) != len(ensemble.params) or ensemble.n_runs != 30:
        raise ValueError("ensemble must contain all 30 fitted HDBSCAN runs")
    if predict_mask is None:
        mask = np.ones(len(values), dtype=bool)
    else:
        mask = _boolean_mask(predict_mask, len(values), "predict_mask")

    labels = np.full((len(values), ensemble.n_runs), -1, dtype=int)
    strengths = np.full((len(values), ensemble.n_runs), np.nan, dtype=float)
    if not np.any(mask):
        return HDBSCANPrediction(labels=labels, strengths=strengths, predict_mask=mask)

    predictor = approximate_predict_fn or _default_approximate_predict
    selected = values[mask]
    for run, clusterer in enumerate(ensemble.clusterers):
        predicted_raw, strengths_raw = predictor(clusterer, selected)
        predicted = _integer_labels(predicted_raw, len(selected), f"prediction run {run}")
        run_strengths = np.asarray(strengths_raw, dtype=float)
        if run_strengths.ndim != 1 or len(run_strengths) != len(selected):
            raise ValueError(f"prediction strengths for run {run} have invalid shape")
        if not np.all(np.isfinite(run_strengths)):
            raise ValueError(f"prediction strengths for run {run} are non-finite")
        if np.any((run_strengths < 0.0) | (run_strengths > 1.0)):
            raise ValueError(f"prediction strengths for run {run} must lie in [0, 1]")
        labels[mask, run] = predicted
        strengths[mask, run] = run_strengths
    return HDBSCANPrediction(labels=labels, strengths=strengths, predict_mask=mask)


def prototype_support(
    labels: np.ndarray,
    prototype_indices: Mapping[str, Sequence[int] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Frequency with which each row co-clusters with a prototype anchor.

    A run contributes support only when both the row and at least one anchor
    have the same non-noise cluster label.
    """

    raw = np.asarray(labels)
    if raw.ndim != 2:
        raise ValueError("labels must have shape (n_rows, n_runs)")
    n_rows, n_runs = raw.shape
    if n_runs < 1:
        raise ValueError("labels must contain at least one run")
    normalized = np.column_stack(
        [_integer_labels(raw[:, run], n_rows, f"labels[:, {run}]") for run in range(n_runs)]
    )

    result: dict[str, np.ndarray] = {}
    for prototype, raw_indices in prototype_indices.items():
        indices_array = np.asarray(raw_indices)
        if indices_array.ndim != 1 or len(indices_array) == 0:
            raise ValueError(f"prototype {prototype!r} requires at least one anchor index")
        if np.issubdtype(indices_array.dtype, np.bool_):
            raise TypeError(f"prototype {prototype!r} indices must be integer row indices")
        try:
            numeric_indices = indices_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prototype {prototype!r} indices must be integers") from exc
        if not np.all(np.isfinite(numeric_indices)) or not np.all(
            numeric_indices == np.floor(numeric_indices)
        ):
            raise ValueError(f"prototype {prototype!r} indices must be finite integers")
        indices = np.unique(numeric_indices.astype(int))
        if np.any((indices < 0) | (indices >= n_rows)):
            raise IndexError(f"prototype {prototype!r} anchor index is out of bounds")

        matches = np.zeros((n_rows, n_runs), dtype=bool)
        for run in range(n_runs):
            anchor_labels = np.unique(normalized[indices, run])
            anchor_labels = anchor_labels[anchor_labels >= 0]
            if len(anchor_labels):
                matches[:, run] = (normalized[:, run] >= 0) & np.isin(
                    normalized[:, run], anchor_labels
                )
        result[str(prototype)] = np.mean(matches, axis=1, dtype=float)
    return result


def prototype_cocluster_frequency(
    labels: np.ndarray,
    prototype_indices: Mapping[str, Sequence[int] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Descriptive alias for :func:`prototype_support`."""

    return prototype_support(labels, prototype_indices)


def stable_families(
    consensus: np.ndarray,
    threshold: float,
    min_family_size: int = 2,
) -> np.ndarray:
    """Cut complete-linkage clustering at ``1 - threshold``.

    Families smaller than ``min_family_size`` are labelled ``-1``.  Remaining
    family identifiers are zero-based and ordered by the first fit-row index.
    """

    similarities = np.asarray(consensus, dtype=float)
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("consensus must be a square matrix")
    if not np.all(np.isfinite(similarities)):
        raise ValueError("consensus contains non-finite values")
    if np.any((similarities < -1e-12) | (similarities > 1.0 + 1e-12)):
        raise ValueError("consensus values must lie in [0, 1]")
    if not np.allclose(similarities, similarities.T, rtol=0.0, atol=1e-12):
        raise ValueError("consensus must be symmetric")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if not isinstance(min_family_size, (int, np.integer)) or min_family_size < 1:
        raise ValueError("min_family_size must be a positive integer")

    n_rows = similarities.shape[0]
    family_labels = np.full(n_rows, -1, dtype=int)
    if n_rows == 0:
        return family_labels
    if n_rows == 1:
        if min_family_size == 1:
            family_labels[0] = 0
        return family_labels

    distance = np.clip(1.0 - similarities, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    raw_families = fcluster(
        tree, t=1.0 - threshold_value, criterion="distance"
    )
    groups = [
        np.flatnonzero(raw_families == raw_label)
        for raw_label in np.unique(raw_families)
    ]
    groups.sort(key=lambda members: int(members[0]))
    next_label = 0
    for members in groups:
        if len(members) < min_family_size:
            continue
        family_labels[members] = next_label
        next_label += 1
    return family_labels


__all__ = [
    "HDBSCANEnsemble",
    "HDBSCANPrediction",
    "SevenFeatureTransform",
    "consensus_matrix",
    "fit_hdbscan_ensemble",
    "predict_hdbscan_ensemble",
    "prototype_cocluster_frequency",
    "prototype_support",
    "stable_families",
]
