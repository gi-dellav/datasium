"""Preprocess tab: scikit-learn preprocessing, feature selection,
decomposition, density estimation, and covariance estimation.

Each category is exposed as a ``ui.expansion`` section in the
:class:`PreprocessPanel`.  The pure helpers (:func:`run_preprocess` and its
per-category workers) are UI-free so they can be unit-tested.

Categories
----------
* **Preprocessing** – scalers, transformers, and encoders that modify column
  values in-place.
* **Feature Selection** – drop low-signal feature columns.
* **Decomposition** – project numeric columns onto lower-dimensional
  components (PCA, SVD, NMF, …).
* **Density Estimation** – add a log-density column via KDE.
* **Covariance Estimation** – add a Mahalanobis-distance column using a
  (robust) covariance estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from nicegui import ui

from datasium.calculate import _is_numeric

# --------------------------------------------------------------------------- #
# method registries  (label, key)
# --------------------------------------------------------------------------- #
PREPROCESS_METHODS: list[tuple[str, str]] = [
    ("Standard Scaler", "standard_scaler"),
    ("Min-Max Scaler", "minmax_scaler"),
    ("Max-Abs Scaler", "maxabs_scaler"),
    ("Robust Scaler", "robust_scaler"),
    ("Normalizer (row-wise)", "normalizer"),
    ("Quantile Transformer", "quantile_transformer"),
    ("Power Transformer", "power_transformer"),
    ("Binarizer", "binarizer"),
    ("Label Encoder", "label_encoder"),
    ("Ordinal Encoder", "ordinal_encoder"),
]

FS_METHODS: list[tuple[str, str]] = [
    ("Variance Threshold", "variance_threshold"),
    ("Select K Best", "select_k_best"),
    ("Select Percentile", "select_percentile"),
    ("Select FPR", "select_fpr"),
    ("Select FDR", "select_fdr"),
    ("Select FWE", "select_fwe"),
]

DECOMP_METHODS: list[tuple[str, str]] = [
    ("PCA", "pca"),
    ("Kernel PCA", "kernel_pca"),
    ("Truncated SVD", "truncated_svd"),
    ("NMF", "nmf"),
    ("Factor Analysis", "factor_analysis"),
    ("FastICA", "fast_ica"),
    ("Incremental PCA", "incremental_pca"),
]

COV_METHODS: list[tuple[str, str]] = [
    ("Empirical Covariance", "empirical"),
    ("Shrunk Covariance", "shrunk"),
    ("Ledoit-Wolf", "ledoit_wolf"),
    ("OAS", "oas"),
    ("Min Cov Det (robust)", "min_cov_det"),
    ("Elliptic Envelope", "elliptic_envelope"),
]

SCORE_FUNCS: dict[str, str] = {
    "f_classif": "ANOVA F (classification)",
    "f_regression": "F (regression)",
    "mutual_info_classif": "Mutual info (classification)",
    "mutual_info_regression": "Mutual info (regression)",
    "chi2": "Chi-squared",
}

KDE_KERNELS: dict[str, str] = {
    "gaussian": "Gaussian",
    "tophat": "Top-hat",
    "epanechnikov": "Epanechnikov",
    "exponential": "Exponential",
    "linear": "Linear",
    "cosine": "Cosine",
}

_ENCODERS = {"label_encoder", "ordinal_encoder"}


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessSpec:
    """Declarative description of a single preprocess operation."""

    category: str = "preprocessing"

    # -- preprocessing -------------------------------------------------------
    preprocess_method: str = "standard_scaler"
    preprocess_columns: list[str] = field(default_factory=list)
    feature_range_min: float = 0.0
    feature_range_max: float = 1.0
    quantile_low: float = 25.0
    quantile_high: float = 75.0
    norm: str = "l2"
    n_quantiles: int = 1000
    output_distribution: str = "uniform"
    power_method: str = "yeo-johnson"
    binarize_threshold: float = 0.0

    # -- feature selection ---------------------------------------------------
    fs_method: str = "variance_threshold"
    fs_columns: list[str] = field(default_factory=list)
    fs_target: str | None = None
    variance_threshold: float = 0.0
    k_best: int = 10
    percentile: float = 10.0
    alpha: float = 0.05
    score_func: str = "f_classif"

    # -- decomposition -------------------------------------------------------
    decomp_method: str = "pca"
    decomp_columns: list[str] = field(default_factory=list)
    n_components: int = 2
    whiten: bool = False
    svd_solver: str = "auto"
    kernel: str = "linear"
    gamma: float | None = None
    decomp_max_iter: int = 200
    keep_originals: bool = True

    # -- density estimation --------------------------------------------------
    density_columns: list[str] = field(default_factory=list)
    bandwidth: float = 1.0
    kde_kernel: str = "gaussian"
    density_output: str = "log_density"

    # -- covariance estimation -----------------------------------------------
    cov_method: str = "empirical"
    cov_columns: list[str] = field(default_factory=list)
    shrinkage: float | None = None
    contamination: float = 0.1
    cov_output: str = "mahalanobis"


# --------------------------------------------------------------------------- #
# pure helpers – shared utilities
# --------------------------------------------------------------------------- #
def _validate_columns(df: pl.DataFrame, cols: list[str], label: str) -> None:
    if not cols:
        raise ValueError(f"select at least one {label} column")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) not found: {', '.join(missing)}")


def _validate_numeric(df: pl.DataFrame, cols: list[str]) -> None:
    bad = [c for c in cols if not _is_numeric(df.schema[c])]
    if bad:
        raise ValueError(f"column(s) not numeric: {', '.join(bad)}")


def _null_safe_transform(
    df: pl.DataFrame,
    cols: list[str],
    transform_fn,
) -> pl.DataFrame:
    """Apply *transform_fn(X) -> X'* on the non-null rows of *cols* and
    re-join, preserving null rows with null output values."""
    import numpy as np

    df_idx = df.with_row_index("__ds_row")
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in cols)
    valid = df_idx.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    X = valid.select(cols).to_numpy().astype(np.float64)
    X_new = transform_fn(X)

    if X_new.shape[1] != len(cols):
        raise ValueError("column count mismatch in _null_safe_transform")

    new_cols = [pl.Series(c, X_new[:, i]) for i, c in enumerate(cols)]
    valid_out = valid.with_columns(new_cols)
    # drop originals before joining so no _right suffix appears
    result = df_idx.drop(cols).join(
        valid_out.select("__ds_row", *cols),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    return result


# --------------------------------------------------------------------------- #
# pure helpers – preprocessing
# --------------------------------------------------------------------------- #
def _apply_preprocessing(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    method = spec.preprocess_method
    cols = spec.preprocess_columns
    _validate_columns(df, cols, "preprocessing")

    if method in _ENCODERS:
        return _apply_encoder(df, spec)

    _validate_numeric(df, cols)

    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        Normalizer,
        QuantileTransformer,
        PowerTransformer,
        Binarizer,
    )

    if method == "standard_scaler":
        scaler = StandardScaler()
    elif method == "minmax_scaler":
        scaler = MinMaxScaler(
            feature_range=(spec.feature_range_min, spec.feature_range_max)
        )
    elif method == "maxabs_scaler":
        scaler = MaxAbsScaler()
    elif method == "robust_scaler":
        scaler = RobustScaler(
            quantile_range=(spec.quantile_low, spec.quantile_high)
        )
    elif method == "normalizer":
        scaler = Normalizer(norm=spec.norm)
    elif method == "quantile_transformer":
        scaler = QuantileTransformer(
            n_quantiles=min(spec.n_quantiles, df.height),
            output_distribution=spec.output_distribution,
        )
    elif method == "power_transformer":
        scaler = PowerTransformer(method=spec.power_method)
    elif method == "binarizer":
        scaler = Binarizer(threshold=spec.binarize_threshold)
    else:
        raise ValueError(f"unknown preprocessing method {method!r}")

    return _null_safe_transform(df, cols, lambda X: scaler.fit_transform(X))


def _apply_encoder(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    import numpy as np

    method = spec.preprocess_method
    cols = spec.preprocess_columns

    df_idx = df.with_row_index("__ds_row")
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in cols)
    valid = df_idx.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    new_cols = []
    for c in cols:
        vals = valid[c].to_list()
        if method == "label_encoder":
            le = LabelEncoder()
            encoded = le.fit_transform(vals).astype(np.int64)
        else:
            oe = OrdinalEncoder()
            encoded = oe.fit_transform([[v] for v in vals]).ravel().astype(np.int64)
        new_cols.append(pl.Series(c, encoded))

    valid_out = valid.with_columns(new_cols)
    result = df_idx.drop(cols).join(
        valid_out.select("__ds_row", *cols),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    return result


# --------------------------------------------------------------------------- #
# pure helpers – feature selection
# --------------------------------------------------------------------------- #
def _apply_feature_selection(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    method = spec.fs_method
    cols = spec.fs_columns
    _validate_columns(df, cols, "feature")
    _validate_numeric(df, cols)

    from sklearn.feature_selection import (
        VarianceThreshold,
        SelectKBest,
        SelectPercentile,
        SelectFpr,
        SelectFdr,
        SelectFwe,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
        chi2,
    )
    import numpy as np

    _score_map = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
        "chi2": chi2,
    }

    supervised = method != "variance_threshold"
    target = spec.fs_target
    if supervised:
        if not target:
            raise ValueError("select a target column for supervised feature selection")
        if target not in df.columns:
            raise ValueError(f"target column {target!r} not found")
        if target in cols:
            raise ValueError("target column must not be in the feature columns")

    # work on non-null rows
    check_cols = [*cols] + ([target] if supervised else [])
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in check_cols)
    valid = df.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    X = valid.select(cols).to_numpy().astype(np.float64)

    if method == "variance_threshold":
        selector = VarianceThreshold(threshold=spec.variance_threshold)
        selector.fit(X)
        keep_mask = selector.get_support()
    else:
        y = valid[target].to_numpy()
        score_fn = _score_map.get(spec.score_func)
        if score_fn is None:
            raise ValueError(f"unknown score function {spec.score_func!r}")
        if method == "select_k_best":
            k = min(spec.k_best, len(cols))
            selector = SelectKBest(score_func=score_fn, k=k)
        elif method == "select_percentile":
            selector = SelectPercentile(score_func=score_fn, percentile=spec.percentile)
        elif method == "select_fpr":
            selector = SelectFpr(score_func=score_fn, alpha=spec.alpha)
        elif method == "select_fdr":
            selector = SelectFdr(score_func=score_fn, alpha=spec.alpha)
        elif method == "select_fwe":
            selector = SelectFwe(score_func=score_fn, alpha=spec.alpha)
        else:
            raise ValueError(f"unknown feature selection method {method!r}")
        selector.fit(X, y)
        keep_mask = selector.get_support()

    kept = [c for c, keep in zip(cols, keep_mask) if keep]
    dropped = [c for c, keep in zip(cols, keep_mask) if not keep]
    if not dropped:
        return df  # nothing to remove
    if not kept and supervised:
        # keep at least the target
        pass
    return df.drop(dropped)


# --------------------------------------------------------------------------- #
# pure helpers – decomposition
# --------------------------------------------------------------------------- #
_DECOMP_PREFIX = {
    "pca": "pca",
    "kernel_pca": "kpca",
    "truncated_svd": "svd",
    "nmf": "nmf",
    "factor_analysis": "fa",
    "fast_ica": "ica",
    "incremental_pca": "ipca",
}


def _apply_decomposition(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    method = spec.decomp_method
    cols = spec.decomp_columns
    _validate_columns(df, cols, "decomposition")
    _validate_numeric(df, cols)

    from sklearn.decomposition import (
        PCA,
        KernelPCA,
        TruncatedSVD,
        NMF,
        FactorAnalysis,
        FastICA,
        IncrementalPCA,
    )
    import numpy as np

    n_comp = min(spec.n_components, len(cols))

    if method == "pca":
        model = PCA(n_components=n_comp, whiten=spec.whiten, svd_solver=spec.svd_solver)
    elif method == "kernel_pca":
        model = KernelPCA(
            n_components=n_comp, kernel=spec.kernel,
            gamma=spec.gamma, fit_inverse_transform=False,
        )
    elif method == "truncated_svd":
        model = TruncatedSVD(n_components=n_comp)
    elif method == "nmf":
        model = NMF(n_components=n_comp, max_iter=spec.decomp_max_iter)
    elif method == "factor_analysis":
        model = FactorAnalysis(n_components=n_comp)
    elif method == "fast_ica":
        model = FastICA(n_components=n_comp, max_iter=spec.decomp_max_iter)
    elif method == "incremental_pca":
        model = IncrementalPCA(n_components=n_comp)
    else:
        raise ValueError(f"unknown decomposition method {method!r}")

    # null-safe: fit on non-null rows, project them, null rows get null comps
    df_idx = df.with_row_index("__ds_row")
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in cols)
    valid = df_idx.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    X = valid.select(cols).to_numpy().astype(np.float64)

    if method == "nmf" and np.any(X < 0):
        raise ValueError("NMF requires non-negative data")

    components = model.fit_transform(X)
    prefix = _DECOMP_PREFIX.get(method, method)
    comp_names = [f"{prefix}_{i}" for i in range(components.shape[1])]

    comp_series = [
        pl.Series(name, components[:, i]) for i, name in enumerate(comp_names)
    ]
    valid_out = valid.with_columns(comp_series)

    result = df_idx.join(
        valid_out.select("__ds_row", *comp_names),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    if not spec.keep_originals:
        result = result.drop(cols)

    return result


# --------------------------------------------------------------------------- #
# pure helpers – density estimation
# --------------------------------------------------------------------------- #
def _apply_density(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    cols = spec.density_columns
    _validate_columns(df, cols, "density")
    _validate_numeric(df, cols)

    out_col = spec.density_output or "log_density"
    if out_col in df.columns:
        raise ValueError(f"column {out_col!r} already exists")

    from sklearn.neighbors import KernelDensity
    import numpy as np

    df_idx = df.with_row_index("__ds_row")
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in cols)
    valid = df_idx.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    X = valid.select(cols).to_numpy().astype(np.float64)
    kde = KernelDensity(bandwidth=spec.bandwidth, kernel=spec.kde_kernel)
    kde.fit(X)
    log_density = kde.score_samples(X)

    valid_out = valid.with_columns(pl.Series(out_col, log_density))
    result = df_idx.join(
        valid_out.select("__ds_row", out_col),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    return result


# --------------------------------------------------------------------------- #
# pure helpers – covariance estimation
# --------------------------------------------------------------------------- #
def _apply_covariance(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    method = spec.cov_method
    cols = spec.cov_columns
    _validate_columns(df, cols, "covariance")
    _validate_numeric(df, cols)

    out_col = spec.cov_output or "mahalanobis"
    if out_col in df.columns:
        raise ValueError(f"column {out_col!r} already exists")

    from sklearn.covariance import (
        EmpiricalCovariance,
        ShrunkCovariance,
        LedoitWolf,
        OAS,
        MinCovDet,
        EllipticEnvelope,
    )
    import numpy as np

    if method == "empirical":
        est = EmpiricalCovariance()
    elif method == "shrunk":
        est = ShrunkCovariance(
            shrinkage=spec.shrinkage if spec.shrinkage is not None else 0.1
        )
    elif method == "ledoit_wolf":
        est = LedoitWolf()
    elif method == "oas":
        est = OAS()
    elif method == "min_cov_det":
        est = MinCovDet()
    elif method == "elliptic_envelope":
        est = EllipticEnvelope(contamination=spec.contamination)
    else:
        raise ValueError(f"unknown covariance method {method!r}")

    df_idx = df.with_row_index("__ds_row")
    mask = pl.all_horizontal(pl.col(c).is_not_null() for c in cols)
    valid = df_idx.filter(mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    X = valid.select(cols).to_numpy().astype(np.float64)
    est.fit(X)
    mahal = est.mahalanobis(X)

    valid_out = valid.with_columns(pl.Series(out_col, mahal))
    result = df_idx.join(
        valid_out.select("__ds_row", out_col),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    return result


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def run_preprocess(df: pl.DataFrame, spec: PreprocessSpec) -> pl.DataFrame:
    """Dispatch to the appropriate category handler.

    Raises ``ValueError`` with a user-facing message on bad input.
    """
    if df.height == 0:
        raise ValueError("no rows to process")

    cat = spec.category
    if cat == "preprocessing":
        return _apply_preprocessing(df, spec)
    if cat == "feature_selection":
        return _apply_feature_selection(df, spec)
    if cat == "decomposition":
        return _apply_decomposition(df, spec)
    if cat == "density":
        return _apply_density(df, spec)
    if cat == "covariance":
        return _apply_covariance(df, spec)
    raise ValueError(f"unknown category {cat!r}")


# --------------------------------------------------------------------------- #
# UI panel
# --------------------------------------------------------------------------- #
class PreprocessPanel:
    """Five expansion sections, one per scikit-learn API category."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_run: Callable[[str], None],
        on_apply: Callable[[], None],
    ) -> None:
        self._columns = columns
        self._on_run = on_run
        self._on_apply = on_apply
        self._active_category: str = "preprocessing"

        all_names = [n for n, _ in columns]
        numeric = [n for n, d in columns if _is_numeric(d)]
        self._all_opts = {n: n for n in all_names} or {"—": "—"}
        self._num_opts = {n: n for n in numeric} or {"—": "—"}

        with parent:
            self._build_preprocessing_expansion()
            self._build_feature_selection_expansion()
            self._build_decomposition_expansion()
            self._build_density_expansion()
            self._build_covariance_expansion()

            ui.separator()
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.button(
                    "Apply",
                    icon="check",
                    on_click=lambda _=None: on_apply(),
                ).props("dense unelevated color=positive")
            self.meta = ui.label("").classes("text-xs opacity-50")
            self.preview_container = ui.column().classes("w-full")

    # ---- preprocessing expansion ------------------------------------------
    def _build_preprocessing_expansion(self) -> None:
        with ui.expansion("Preprocessing", icon="tune").classes("w-full"):
            ui.label(
                "Scale, transform, or encode column values in-place."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.pp_method = (
                    ui.select(
                        options={k: lbl for lbl, k in PREPROCESS_METHODS},
                        value="standard_scaler",
                        label="Method",
                        on_change=lambda _e: self._refresh_pp_params(),
                    )
                    .props("dense outlined")
                    .classes("w-52")
                )
                self.pp_cols = (
                    ui.select(
                        options=self._all_opts,
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Columns",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-64")
                )
            self.pp_param_container = ui.column().classes("w-full gap-1")
            self._refresh_pp_params()
            ui.button(
                "Run", icon="play_arrow",
                on_click=lambda _=None: self._on_run("preprocessing"),
            ).props("dense unelevated color=primary")

    def _refresh_pp_params(self) -> None:
        self.pp_param_container.clear()
        m = self.pp_method.value
        with self.pp_param_container:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                if m == "minmax_scaler":
                    self._pp_range_min = (
                        ui.number(value=0.0, label="range min")
                        .props("dense outlined").classes("w-28")
                    )
                    self._pp_range_max = (
                        ui.number(value=1.0, label="range max")
                        .props("dense outlined").classes("w-28")
                    )
                if m == "robust_scaler":
                    self._pp_q_low = (
                        ui.number(value=25.0, label="quantile low %")
                        .props("dense outlined").classes("w-32")
                    )
                    self._pp_q_high = (
                        ui.number(value=75.0, label="quantile high %")
                        .props("dense outlined").classes("w-32")
                    )
                if m == "normalizer":
                    self._pp_norm = (
                        ui.select(
                            {"l1": "L1", "l2": "L2", "max": "Max"},
                            value="l2", label="norm",
                        ).props("dense outlined").classes("w-24")
                    )
                if m == "quantile_transformer":
                    self._pp_n_quantiles = (
                        ui.number(value=1000, min=2, label="n_quantiles")
                        .props("dense outlined").classes("w-28")
                    )
                    self._pp_output_dist = (
                        ui.select(
                            {"uniform": "uniform", "normal": "normal"},
                            value="uniform", label="output_distribution",
                        ).props("dense outlined").classes("w-40")
                    )
                if m == "power_transformer":
                    self._pp_power_method = (
                        ui.select(
                            {"yeo-johnson": "Yeo-Johnson", "box-cox": "Box-Cox"},
                            value="yeo-johnson", label="method",
                        ).props("dense outlined").classes("w-36")
                    )
                if m == "binarizer":
                    self._pp_bin_thresh = (
                        ui.number(value=0.0, label="threshold")
                        .props("dense outlined").classes("w-28")
                    )

    # ---- feature selection expansion --------------------------------------
    def _build_feature_selection_expansion(self) -> None:
        with ui.expansion("Feature Selection", icon="filter_list").classes("w-full"):
            ui.label(
                "Remove low-signal feature columns. Supervised methods need a "
                "target column."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.fs_method = (
                    ui.select(
                        options={k: lbl for lbl, k in FS_METHODS},
                        value="variance_threshold",
                        label="Method",
                        on_change=lambda _e: self._refresh_fs_params(),
                    )
                    .props("dense outlined")
                    .classes("w-48")
                )
                self.fs_cols = (
                    ui.select(
                        options=self._num_opts,
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Feature columns",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-64")
                )
                self.fs_target = (
                    ui.select(
                        options=self._all_opts,
                        value=None,
                        clearable=True,
                        label="Target column",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
            self.fs_param_container = ui.column().classes("w-full gap-1")
            self._refresh_fs_params()
            ui.button(
                "Run", icon="play_arrow",
                on_click=lambda _=None: self._on_run("feature_selection"),
            ).props("dense unelevated color=primary")

    def _refresh_fs_params(self) -> None:
        self.fs_param_container.clear()
        m = self.fs_method.value
        supervised = m != "variance_threshold"
        self.fs_target.set_visibility(supervised)
        with self.fs_param_container:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                if m == "variance_threshold":
                    self._fs_var_thresh = (
                        ui.number(value=0.0, min=0.0, step=0.01, label="threshold")
                        .props("dense outlined").classes("w-28")
                    )
                if supervised:
                    self._fs_score_func = (
                        ui.select(
                            SCORE_FUNCS, value="f_classif", label="score_func",
                        ).props("dense outlined").classes("w-56")
                    )
                if m == "select_k_best":
                    self._fs_k = (
                        ui.number(value=10, min=1, label="k")
                        .props("dense outlined").classes("w-24")
                    )
                if m == "select_percentile":
                    self._fs_pct = (
                        ui.number(value=10.0, min=1.0, max=100.0, label="percentile")
                        .props("dense outlined").classes("w-28")
                    )
                if m in ("select_fpr", "select_fdr", "select_fwe"):
                    self._fs_alpha = (
                        ui.number(value=0.05, min=0.001, max=1.0, step=0.01, label="alpha")
                        .props("dense outlined").classes("w-28")
                    )

    # ---- decomposition expansion ------------------------------------------
    def _build_decomposition_expansion(self) -> None:
        with ui.expansion("Decomposition", icon="account_tree").classes("w-full"):
            ui.label(
                "Project numeric columns onto lower-dimensional components."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.dc_method = (
                    ui.select(
                        options={k: lbl for lbl, k in DECOMP_METHODS},
                        value="pca",
                        label="Method",
                        on_change=lambda _e: self._refresh_dc_params(),
                    )
                    .props("dense outlined")
                    .classes("w-44")
                )
                self.dc_cols = (
                    ui.select(
                        options=self._num_opts,
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Columns",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-64")
                )
                self.dc_n_comp = (
                    ui.number(value=2, min=1, label="n_components")
                    .props("dense outlined").classes("w-32")
                )
                self.dc_keep = ui.switch("Keep originals", value=True).props("dense")
            self.dc_param_container = ui.column().classes("w-full gap-1")
            self._refresh_dc_params()
            ui.button(
                "Run", icon="play_arrow",
                on_click=lambda _=None: self._on_run("decomposition"),
            ).props("dense unelevated color=primary")

    def _refresh_dc_params(self) -> None:
        self.dc_param_container.clear()
        m = self.dc_method.value
        with self.dc_param_container:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                if m == "pca":
                    self._dc_whiten = ui.switch("whiten", value=False).props("dense")
                    self._dc_solver = (
                        ui.select(
                            {"auto": "auto", "full": "full", "arpack": "arpack",
                             "randomized": "randomized"},
                            value="auto", label="svd_solver",
                        ).props("dense outlined").classes("w-32")
                    )
                if m == "kernel_pca":
                    self._dc_kernel = (
                        ui.select(
                            {"linear": "linear", "rbf": "rbf", "poly": "poly",
                             "sigmoid": "sigmoid", "cosine": "cosine"},
                            value="linear", label="kernel",
                        ).props("dense outlined").classes("w-28")
                    )
                    self._dc_gamma = (
                        ui.input(value="", label="gamma (blank=None)")
                        .props("dense outlined").classes("w-36")
                    )
                if m in ("nmf", "fast_ica"):
                    self._dc_max_iter = (
                        ui.number(value=200, min=1, label="max_iter")
                        .props("dense outlined").classes("w-28")
                    )

    # ---- density estimation expansion -------------------------------------
    def _build_density_expansion(self) -> None:
        with ui.expansion("Density Estimation", icon="blur_on").classes("w-full"):
            ui.label(
                "Add a log-density column via Kernel Density Estimation."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.kd_cols = (
                    ui.select(
                        options=self._num_opts,
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Columns",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-64")
                )
                self.kd_bandwidth = (
                    ui.number(value=1.0, min=0.01, step=0.1, label="bandwidth")
                    .props("dense outlined").classes("w-28")
                )
                self.kd_kernel = (
                    ui.select(
                        KDE_KERNELS, value="gaussian", label="kernel",
                    ).props("dense outlined").classes("w-32")
                )
                self.kd_out = (
                    ui.input(value="log_density", label="output column")
                    .props("dense outlined").classes("w-32")
                )
            ui.button(
                "Run", icon="play_arrow",
                on_click=lambda _=None: self._on_run("density"),
            ).props("dense unelevated color=primary")

    # ---- covariance estimation expansion ----------------------------------
    def _build_covariance_expansion(self) -> None:
        with ui.expansion("Covariance Estimation", icon="grid_on").classes("w-full"):
            ui.label(
                "Add a Mahalanobis-distance column using a (robust) covariance "
                "estimate."
            ).classes("text-xs opacity-50")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.cv_method = (
                    ui.select(
                        options={k: lbl for lbl, k in COV_METHODS},
                        value="empirical",
                        label="Method",
                        on_change=lambda _e: self._refresh_cv_params(),
                    )
                    .props("dense outlined")
                    .classes("w-48")
                )
                self.cv_cols = (
                    ui.select(
                        options=self._num_opts,
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Columns",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-64")
                )
                self.cv_out = (
                    ui.input(value="mahalanobis", label="output column")
                    .props("dense outlined").classes("w-32")
                )
            self.cv_param_container = ui.column().classes("w-full gap-1")
            self._refresh_cv_params()
            ui.button(
                "Run", icon="play_arrow",
                on_click=lambda _=None: self._on_run("covariance"),
            ).props("dense unelevated color=primary")

    def _refresh_cv_params(self) -> None:
        self.cv_param_container.clear()
        m = self.cv_method.value
        with self.cv_param_container:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                if m == "shrunk":
                    self._cv_shrinkage = (
                        ui.input(value="", label="shrinkage (blank=auto)")
                        .props("dense outlined").classes("w-40")
                    )
                if m == "elliptic_envelope":
                    self._cv_contam = (
                        ui.number(value=0.1, min=0.01, max=0.5, step=0.01,
                                  label="contamination")
                        .props("dense outlined").classes("w-32")
                    )

    # ---- spec accessor ----------------------------------------------------
    @property
    def active_category(self) -> str:
        return self._active_category

    def spec_for(self, category: str) -> PreprocessSpec:
        """Build a :class:`PreprocessSpec` from the widgets in *category*."""
        self._active_category = category
        spec = PreprocessSpec(category=category)

        if category == "preprocessing":
            spec.preprocess_method = self.pp_method.value or "standard_scaler"
            spec.preprocess_columns = list(self.pp_cols.value or [])
            spec.feature_range_min = float(getattr(self, "_pp_range_min", _V(0.0)).value or 0.0)
            spec.feature_range_max = float(getattr(self, "_pp_range_max", _V(1.0)).value or 1.0)
            spec.quantile_low = float(getattr(self, "_pp_q_low", _V(25.0)).value or 25.0)
            spec.quantile_high = float(getattr(self, "_pp_q_high", _V(75.0)).value or 75.0)
            spec.norm = getattr(self, "_pp_norm", _V("l2")).value or "l2"
            spec.n_quantiles = int(getattr(self, "_pp_n_quantiles", _V(1000)).value or 1000)
            spec.output_distribution = getattr(self, "_pp_output_dist", _V("uniform")).value or "uniform"
            spec.power_method = getattr(self, "_pp_power_method", _V("yeo-johnson")).value or "yeo-johnson"
            spec.binarize_threshold = float(getattr(self, "_pp_bin_thresh", _V(0.0)).value or 0.0)

        elif category == "feature_selection":
            spec.fs_method = self.fs_method.value or "variance_threshold"
            spec.fs_columns = list(self.fs_cols.value or [])
            spec.fs_target = self.fs_target.value if self.fs_target.visible else None
            spec.variance_threshold = float(getattr(self, "_fs_var_thresh", _V(0.0)).value or 0.0)
            spec.k_best = int(getattr(self, "_fs_k", _V(10)).value or 10)
            spec.percentile = float(getattr(self, "_fs_pct", _V(10.0)).value or 10.0)
            spec.alpha = float(getattr(self, "_fs_alpha", _V(0.05)).value or 0.05)
            spec.score_func = getattr(self, "_fs_score_func", _V("f_classif")).value or "f_classif"

        elif category == "decomposition":
            spec.decomp_method = self.dc_method.value or "pca"
            spec.decomp_columns = list(self.dc_cols.value or [])
            spec.n_components = int(self.dc_n_comp.value or 2)
            spec.keep_originals = bool(self.dc_keep.value)
            spec.whiten = bool(getattr(self, "_dc_whiten", _V(False)).value)
            spec.svd_solver = getattr(self, "_dc_solver", _V("auto")).value or "auto"
            spec.kernel = getattr(self, "_dc_kernel", _V("linear")).value or "linear"
            g = getattr(self, "_dc_gamma", _V("")).value
            spec.gamma = float(g) if g and str(g).strip() else None
            spec.decomp_max_iter = int(getattr(self, "_dc_max_iter", _V(200)).value or 200)

        elif category == "density":
            spec.density_columns = list(self.kd_cols.value or [])
            spec.bandwidth = float(self.kd_bandwidth.value or 1.0)
            spec.kde_kernel = self.kd_kernel.value or "gaussian"
            spec.density_output = (self.kd_out.value or "log_density").strip() or "log_density"

        elif category == "covariance":
            spec.cov_method = self.cv_method.value or "empirical"
            spec.cov_columns = list(self.cv_cols.value or [])
            spec.cov_output = (self.cv_out.value or "mahalanobis").strip() or "mahalanobis"
            s = getattr(self, "_cv_shrinkage", _V("")).value
            spec.shrinkage = float(s) if s and str(s).strip() else None
            spec.contamination = float(getattr(self, "_cv_contam", _V(0.1)).value or 0.1)

        return spec

    # ---- output slots -----------------------------------------------------
    def set_meta(self, text: str) -> None:
        self.meta.set_text(text)

    def render_preview(self, df: pl.DataFrame) -> None:
        self.preview_container.clear()
        with self.preview_container:
            if df.width == 0:
                ui.label("No columns remain.").classes("text-sm opacity-50")
                return
            columns = [
                {
                    "name": c,
                    "label": f"{c}\n{df.schema[c]}",
                    "field": c,
                    "align": "left",
                    "sortable": True,
                }
                for c in df.columns
            ]
            rows = df.head(200).rows(named=True)
            ui.table(columns=columns, rows=rows, row_key=df.columns[0]).props(
                "flat dense"
            ).classes("w-full")

    def render_error(self, msg: str) -> None:
        self.preview_container.clear()
        with self.preview_container:
            ui.label(msg).classes("text-sm text-negative")


class _V:
    """Tiny fallback value object so ``getattr(self, '_x', _V(dflt)).value``
    works when a dynamic widget was never created."""

    def __init__(self, v):
        self.value = v
