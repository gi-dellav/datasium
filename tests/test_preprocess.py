"""Tests for the pure preprocess helpers."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.preprocess import (
    PreprocessSpec,
    run_preprocess,
    PREPROCESS_METHODS,
    FS_METHODS,
    DECOMP_METHODS,
    COV_METHODS,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "c": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "cat": ["x", "y", "x", "y", "x", "y"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def test_empty_frame_raises():
    empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="no rows to process"):
        run_preprocess(empty, PreprocessSpec(category="preprocessing"))


def test_unknown_category_raises(df):
    with pytest.raises(ValueError, match="unknown category"):
        run_preprocess(df, PreprocessSpec(category="bogus"))


# --------------------------------------------------------------------------- #
# preprocessing – scalers
# --------------------------------------------------------------------------- #
def test_no_columns_raises(df):
    spec = PreprocessSpec(category="preprocessing", preprocess_columns=[])
    with pytest.raises(ValueError, match="select at least one"):
        run_preprocess(df, spec)


def test_missing_column_raises(df):
    spec = PreprocessSpec(
        category="preprocessing", preprocess_columns=["zzz"]
    )
    with pytest.raises(ValueError, match="not found: zzz"):
        run_preprocess(df, spec)


def test_non_numeric_column_raises(df):
    spec = PreprocessSpec(
        category="preprocessing", preprocess_columns=["cat"]
    )
    with pytest.raises(ValueError, match="not numeric: cat"):
        run_preprocess(df, spec)


@pytest.mark.parametrize(
    "method",
    [
        "standard_scaler",
        "minmax_scaler",
        "maxabs_scaler",
        "robust_scaler",
        "normalizer",
        "quantile_transformer",
        "power_transformer",
        "binarizer",
    ],
)
def test_scaler_replaces_values(df, method):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method=method,
        preprocess_columns=["a", "b"],
    )
    result = run_preprocess(df, spec)
    assert result.height == df.height
    assert set(result.columns) == set(df.columns)
    # values should differ from the originals (except binarizer on some data)
    assert result["a"].to_list() != df["a"].to_list() or method == "binarizer"


def test_standard_scaler_zero_mean(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="standard_scaler",
        preprocess_columns=["a"],
    )
    result = run_preprocess(df, spec)
    mean = result["a"].mean()
    assert abs(mean) < 1e-10


def test_minmax_scaler_range(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="minmax_scaler",
        preprocess_columns=["a"],
        feature_range_min=0.0,
        feature_range_max=1.0,
    )
    result = run_preprocess(df, spec)
    assert abs(result["a"].min()) < 1e-10
    assert abs(result["a"].max() - 1.0) < 1e-10


def test_binarizer_threshold(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="binarizer",
        preprocess_columns=["a"],
        binarize_threshold=3.5,
    )
    result = run_preprocess(df, spec)
    vals = result["a"].to_list()
    assert all(v in (0.0, 1.0) for v in vals)
    assert vals[0] == 0.0  # 1.0 <= 3.5
    assert vals[5] == 1.0  # 6.0 > 3.5


def test_unknown_preprocess_method_raises(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="bogus",
        preprocess_columns=["a"],
    )
    with pytest.raises(ValueError, match="unknown preprocessing method"):
        run_preprocess(df, spec)


# --------------------------------------------------------------------------- #
# preprocessing – encoders
# --------------------------------------------------------------------------- #
def test_label_encoder(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="label_encoder",
        preprocess_columns=["cat"],
    )
    result = run_preprocess(df, spec)
    assert result.height == df.height
    vals = result["cat"].to_list()
    assert set(vals) == {0, 1}


def test_ordinal_encoder(df):
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="ordinal_encoder",
        preprocess_columns=["cat"],
    )
    result = run_preprocess(df, spec)
    vals = result["cat"].to_list()
    assert set(vals) == {0.0, 1.0}


# --------------------------------------------------------------------------- #
# preprocessing – null handling
# --------------------------------------------------------------------------- #
def test_null_rows_preserved():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="standard_scaler",
        preprocess_columns=["a"],
    )
    result = run_preprocess(df, spec)
    assert result.height == 4
    assert result["a"][1] is None
    assert result["a"][0] is not None


def test_all_nulls_raises():
    df = pl.DataFrame({"a": pl.Series([None, None], dtype=pl.Float64)})
    spec = PreprocessSpec(
        category="preprocessing",
        preprocess_method="standard_scaler",
        preprocess_columns=["a"],
    )
    with pytest.raises(ValueError, match="all rows contain nulls"):
        run_preprocess(df, spec)


# --------------------------------------------------------------------------- #
# feature selection
# --------------------------------------------------------------------------- #
def test_variance_threshold_drops_constant():
    df = pl.DataFrame(
        {
            "const": [1.0, 1.0, 1.0, 1.0],
            "vary": [1.0, 2.0, 3.0, 4.0],
            "label": [0, 1, 0, 1],
        }
    )
    spec = PreprocessSpec(
        category="feature_selection",
        fs_method="variance_threshold",
        fs_columns=["const", "vary"],
        variance_threshold=0.0,
    )
    result = run_preprocess(df, spec)
    assert "const" not in result.columns
    assert "vary" in result.columns
    assert "label" in result.columns


def test_select_k_best():
    df = pl.DataFrame(
        {
            "noise": [1.0, 1.1, 0.9, 1.0, 1.1, 0.9],
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    spec = PreprocessSpec(
        category="feature_selection",
        fs_method="select_k_best",
        fs_columns=["noise", "signal"],
        fs_target="target",
        k_best=1,
        score_func="f_classif",
    )
    result = run_preprocess(df, spec)
    assert "signal" in result.columns
    assert "target" in result.columns
    assert result.height == df.height


def test_fs_needs_target():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    spec = PreprocessSpec(
        category="feature_selection",
        fs_method="select_k_best",
        fs_columns=["a", "b"],
        fs_target=None,
    )
    with pytest.raises(ValueError, match="target column"):
        run_preprocess(df, spec)


def test_fs_target_in_features_raises(df):
    spec = PreprocessSpec(
        category="feature_selection",
        fs_method="select_k_best",
        fs_columns=["a", "label"],
        fs_target="label",
    )
    with pytest.raises(ValueError, match="must not be in the feature"):
        run_preprocess(df, spec)


@pytest.mark.parametrize(
    "method",
    ["select_percentile", "select_fpr", "select_fdr", "select_fwe"],
)
def test_fs_supervised_methods_run(df, method):
    spec = PreprocessSpec(
        category="feature_selection",
        fs_method=method,
        fs_columns=["a", "b", "c"],
        fs_target="label",
        score_func="f_classif",
    )
    result = run_preprocess(df, spec)
    assert result.height == df.height
    assert "label" in result.columns


# --------------------------------------------------------------------------- #
# decomposition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method,prefix",
    [
        ("pca", "pca"),
        ("truncated_svd", "svd"),
        ("factor_analysis", "fa"),
        ("fast_ica", "ica"),
        ("incremental_pca", "ipca"),
    ],
)
def test_decomposition_adds_components(df, method, prefix):
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method=method,
        decomp_columns=["a", "b", "c"],
        n_components=2,
    )
    result = run_preprocess(df, spec)
    assert f"{prefix}_0" in result.columns
    assert f"{prefix}_1" in result.columns
    assert result.height == df.height
    # originals kept by default
    assert "a" in result.columns


def test_decomposition_drop_originals(df):
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method="pca",
        decomp_columns=["a", "b", "c"],
        n_components=2,
        keep_originals=False,
    )
    result = run_preprocess(df, spec)
    assert "a" not in result.columns
    assert "pca_0" in result.columns
    assert "cat" in result.columns  # non-selected columns kept


def test_nmf_negative_raises():
    df = pl.DataFrame({"a": [-1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method="nmf",
        decomp_columns=["a", "b"],
        n_components=1,
    )
    with pytest.raises(ValueError, match="non-negative"):
        run_preprocess(df, spec)


def test_nmf_positive(df):
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method="nmf",
        decomp_columns=["a", "b"],
        n_components=1,
    )
    result = run_preprocess(df, spec)
    assert "nmf_0" in result.columns


def test_kernel_pca(df):
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method="kernel_pca",
        decomp_columns=["a", "b", "c"],
        n_components=2,
        kernel="rbf",
    )
    result = run_preprocess(df, spec)
    assert "kpca_0" in result.columns


def test_decomposition_null_rows():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    spec = PreprocessSpec(
        category="decomposition",
        decomp_method="pca",
        decomp_columns=["a", "b"],
        n_components=1,
    )
    result = run_preprocess(df, spec)
    assert result.height == 4
    assert result["pca_0"][1] is None
    assert result["pca_0"][0] is not None


# --------------------------------------------------------------------------- #
# density estimation
# --------------------------------------------------------------------------- #
def test_kde_adds_column(df):
    spec = PreprocessSpec(
        category="density",
        density_columns=["a", "b"],
        bandwidth=1.0,
    )
    result = run_preprocess(df, spec)
    assert "log_density" in result.columns
    assert result.height == df.height
    assert result["log_density"].null_count() == 0


def test_kde_custom_output(df):
    spec = PreprocessSpec(
        category="density",
        density_columns=["a"],
        density_output="kde_score",
    )
    result = run_preprocess(df, spec)
    assert "kde_score" in result.columns


def test_kde_existing_column_raises(df):
    spec = PreprocessSpec(
        category="density",
        density_columns=["a"],
        density_output="a",
    )
    with pytest.raises(ValueError, match="already exists"):
        run_preprocess(df, spec)


def test_kde_null_rows():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    spec = PreprocessSpec(category="density", density_columns=["a"])
    result = run_preprocess(df, spec)
    assert result.height == 4
    assert result["log_density"][1] is None


# --------------------------------------------------------------------------- #
# covariance estimation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method",
    ["empirical", "shrunk", "ledoit_wolf", "oas", "min_cov_det", "elliptic_envelope"],
)
def test_covariance_adds_column(df, method):
    spec = PreprocessSpec(
        category="covariance",
        cov_method=method,
        cov_columns=["a", "b"],
    )
    result = run_preprocess(df, spec)
    assert "mahalanobis" in result.columns
    assert result.height == df.height
    assert result["mahalanobis"].null_count() == 0


def test_covariance_custom_output(df):
    spec = PreprocessSpec(
        category="covariance",
        cov_method="empirical",
        cov_columns=["a", "b"],
        cov_output="dist",
    )
    result = run_preprocess(df, spec)
    assert "dist" in result.columns


def test_covariance_existing_column_raises(df):
    spec = PreprocessSpec(
        category="covariance",
        cov_method="empirical",
        cov_columns=["a"],
        cov_output="a",
    )
    with pytest.raises(ValueError, match="already exists"):
        run_preprocess(df, spec)


def test_covariance_null_rows():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    spec = PreprocessSpec(
        category="covariance", cov_method="empirical", cov_columns=["a", "b"]
    )
    result = run_preprocess(df, spec)
    assert result.height == 4
    assert result["mahalanobis"][1] is None


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #
def test_preprocess_methods_registry():
    keys = {k for _, k in PREPROCESS_METHODS}
    assert "standard_scaler" in keys
    assert "label_encoder" in keys
    assert len(keys) == 10


def test_fs_methods_registry():
    keys = {k for _, k in FS_METHODS}
    assert "variance_threshold" in keys
    assert len(keys) == 6


def test_decomp_methods_registry():
    keys = {k for _, k in DECOMP_METHODS}
    assert "pca" in keys
    assert len(keys) == 7


def test_cov_methods_registry():
    keys = {k for _, k in COV_METHODS}
    assert "empirical" in keys
    assert len(keys) == 6
