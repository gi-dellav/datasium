"""Cluster tab: run scikit-learn clustering algorithms on a dataset.

The :func:`run_clustering` helper takes a materialised :class:`polars.DataFrame`
plus a :class:`ClusterSpec` and returns a new DataFrame with an integer cluster
label column appended. It is UI-free so it can be unit-tested.

The :func:`build_cluster_figure` helper turns the clustered DataFrame into a
declarative Plotly ``dict`` (scatter of the first two feature columns coloured
by cluster label).

The :class:`ClusterPanel` widget wires algorithm selection, per-algorithm
parameter inputs, and column selection to the app's cluster callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from nicegui import ui

from datasium.calculate import _is_numeric

# (label, key) for the algorithm select.
ALGORITHMS: list[tuple[str, str]] = [
    ("K-Means", "kmeans"),
    ("Mini-Batch K-Means", "minibatch_kmeans"),
    ("Affinity Propagation", "affinity_propagation"),
    ("Mean Shift", "mean_shift"),
    ("Spectral Clustering", "spectral"),
    ("Agglomerative (Ward)", "ward"),
    ("Agglomerative Clustering", "agglomerative"),
    ("DBSCAN", "dbscan"),
    ("HDBSCAN", "hdbscan"),
    ("OPTICS", "optics"),
    ("Birch", "birch"),
    ("Gaussian Mixture", "gmm"),
]

_METRIC_OPTIONS = {
    "euclidean": "Euclidean",
    "manhattan": "Manhattan",
    "cosine": "Cosine",
    "l1": "L1",
    "l2": "L2",
}


@dataclass
class ClusterSpec:
    """Declarative description of a clustering operation."""

    algorithm: str = "kmeans"
    columns: list[str] = field(default_factory=list)
    scale: bool = True
    output_column: str = "cluster"

    # shared
    n_clusters: int = 3
    random_state: int = 42
    max_iter: int = 300

    # kmeans / minibatch
    init: str = "k-means++"
    n_init: int = 10

    # minibatch
    batch_size: int = 1024

    # affinity propagation
    damping: float = 0.5
    preference: float | None = None

    # mean shift
    bandwidth: float | None = None
    bin_seeding: bool = False

    # spectral
    affinity: str = "rbf"
    n_neighbors: int = 10
    assign_labels: str = "kmeans"

    # agglomerative / ward
    linkage: str = "ward"
    metric: str = "euclidean"
    distance_threshold: float | None = None

    # dbscan
    eps: float = 0.5
    min_samples: int = 5

    # hdbscan
    min_cluster_size: int = 5
    cluster_selection_method: str = "eom"

    # optics
    max_eps: float | None = None
    cluster_method: str = "dbscan"

    # birch
    threshold: float = 0.5
    branching_factor: int = 50

    # gmm
    covariance_type: str = "full"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def _build_model(spec: ClusterSpec):
    """Instantiate the scikit-learn estimator for ``spec.algorithm``."""
    from sklearn.cluster import (
        KMeans,
        MiniBatchKMeans,
        AffinityPropagation,
        MeanShift,
        SpectralClustering,
        AgglomerativeClustering,
        DBSCAN,
        HDBSCAN,
        OPTICS,
        Birch,
    )
    from sklearn.mixture import GaussianMixture

    algo = spec.algorithm

    if algo == "kmeans":
        return KMeans(
            n_clusters=spec.n_clusters,
            init=spec.init,
            n_init=spec.n_init,
            max_iter=spec.max_iter,
            random_state=spec.random_state,
        )

    if algo == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=spec.n_clusters,
            init=spec.init,
            n_init=spec.n_init,
            max_iter=spec.max_iter,
            batch_size=spec.batch_size,
            random_state=spec.random_state,
        )

    if algo == "affinity_propagation":
        return AffinityPropagation(
            damping=spec.damping,
            max_iter=spec.max_iter,
            preference=spec.preference,
            random_state=spec.random_state,
        )

    if algo == "mean_shift":
        return MeanShift(
            bandwidth=spec.bandwidth,
            bin_seeding=spec.bin_seeding,
        )

    if algo == "spectral":
        return SpectralClustering(
            n_clusters=spec.n_clusters,
            affinity=spec.affinity,
            n_neighbors=spec.n_neighbors,
            assign_labels=spec.assign_labels,
            random_state=spec.random_state,
        )

    if algo == "ward":
        return AgglomerativeClustering(
            n_clusters=spec.n_clusters,
            linkage="ward",
            distance_threshold=spec.distance_threshold,
        )

    if algo == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=spec.n_clusters,
            linkage=spec.linkage,
            metric=spec.metric if spec.linkage != "ward" else "euclidean",
            distance_threshold=spec.distance_threshold,
        )

    if algo == "dbscan":
        return DBSCAN(
            eps=spec.eps,
            min_samples=spec.min_samples,
            metric=spec.metric,
        )

    if algo == "hdbscan":
        return HDBSCAN(
            min_cluster_size=spec.min_cluster_size,
            min_samples=spec.min_samples,
            metric=spec.metric,
            cluster_selection_method=spec.cluster_selection_method,
        )

    if algo == "optics":
        return OPTICS(
            min_samples=spec.min_samples,
            max_eps=spec.max_eps if spec.max_eps is not None else float("inf"),
            metric=spec.metric,
            cluster_method=spec.cluster_method,
            eps=spec.eps if spec.cluster_method == "dbscan" else None,
        )

    if algo == "birch":
        return Birch(
            threshold=spec.threshold,
            branching_factor=spec.branching_factor,
            n_clusters=spec.n_clusters,
        )

    if algo == "gmm":
        return GaussianMixture(
            n_components=spec.n_clusters,
            covariance_type=spec.covariance_type,
            max_iter=spec.max_iter,
            n_init=spec.n_init,
            random_state=spec.random_state,
        )

    raise ValueError(f"unknown algorithm {algo!r}")


def run_clustering(df: pl.DataFrame, spec: ClusterSpec) -> pl.DataFrame:
    """Run the clustering algorithm described by ``spec`` on ``df``.

    Returns a new DataFrame with the cluster label column appended.  Rows that
    are null in any of the selected feature columns receive a null label.
    Density-based algorithms (DBSCAN, HDBSCAN, OPTICS) use ``-1`` for noise
    points.

    Raises ``ValueError`` with a user-facing message on bad input.
    """
    if not spec.columns:
        raise ValueError("select at least one numeric column")

    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(f"column(s) not found: {', '.join(missing)}")

    schema = df.schema
    non_numeric = [c for c in spec.columns if not _is_numeric(schema[c])]
    if non_numeric:
        raise ValueError(
            f"column(s) not numeric: {', '.join(non_numeric)}"
        )

    if df.height == 0:
        raise ValueError("no rows to cluster")

    out_col = spec.output_column or "cluster"
    if out_col in df.columns:
        raise ValueError(
            f"column {out_col!r} already exists — choose a different output name"
        )

    # rows with nulls in any feature column cannot be clustered; tag every
    # row with its positional index *before* filtering so we can re-join
    # labelled rows back onto the original frame.
    df_idx = df.with_row_index("__ds_row")
    not_null_mask = pl.all_horizontal(
        pl.col(c).is_not_null() for c in spec.columns
    )
    valid = df_idx.filter(not_null_mask)
    if valid.height == 0:
        raise ValueError("all rows contain nulls in the selected columns")

    import numpy as np

    X = valid.select(spec.columns).to_numpy().astype(np.float64)

    if spec.scale:
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)

    model = _build_model(spec)
    labels = model.fit_predict(X)

    label_series = pl.Series(out_col, labels, dtype=pl.Int32)
    valid_labeled = valid.with_columns(label_series)

    result = df_idx.join(
        valid_labeled.select("__ds_row", out_col),
        on="__ds_row",
        how="left",
    ).drop("__ds_row")

    return result


def build_cluster_figure(
    df: pl.DataFrame,
    spec: ClusterSpec,
) -> dict:
    """Build a Plotly scatter figure coloured by cluster label.

    Uses the first two feature columns for the X/Y axes.  When only one
    feature column is selected, a histogram coloured by cluster is returned
    instead.
    """
    out_col = spec.output_column or "cluster"
    if out_col not in df.columns:
        raise ValueError(f"column {out_col!r} not found — run clustering first")

    clustered = df.filter(pl.col(out_col).is_not_null())
    if clustered.height == 0:
        raise ValueError("no clustered rows to plot")

    cols = spec.columns
    if len(cols) >= 2:
        x_col, y_col = cols[0], cols[1]
        labels = clustered[out_col].to_list()
        unique_labels = sorted(set(labels))
        traces = []
        for lbl in unique_labels:
            mask = clustered.filter(pl.col(out_col) == lbl)
            name = f"noise" if lbl == -1 else f"cluster {lbl}"
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": name,
                    "x": mask[x_col].to_numpy(),
                    "y": mask[y_col].to_numpy(),
                    "marker": {"size": 5},
                }
            )
        layout: dict = {
            "title": {"text": f"Clusters ({spec.algorithm})"},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "legend": {"orientation": "h"},
            "xaxis": {"title": x_col},
            "yaxis": {"title": y_col},
        }
        return {"data": traces, "layout": layout}

    # single column → histogram
    x_col = cols[0]
    labels = clustered[out_col].to_list()
    unique_labels = sorted(set(labels))
    traces = []
    for lbl in unique_labels:
        mask = clustered.filter(pl.col(out_col) == lbl)
        name = "noise" if lbl == -1 else f"cluster {lbl}"
        traces.append(
            {
                "type": "histogram",
                "name": name,
                "x": mask[x_col].to_numpy(),
                "marker": {"opacity": 0.6},
            }
        )
    layout = {
        "title": {"text": f"Clusters ({spec.algorithm})"},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "legend": {"orientation": "h"},
        "xaxis": {"title": x_col},
        "yaxis": {"title": "count"},
        "barmode": "overlay",
    }
    return {"data": traces, "layout": layout}


# --------------------------------------------------------------------------- #
# UI panel
# --------------------------------------------------------------------------- #
class ClusterPanel:
    """Algorithm selector + per-algorithm parameter inputs + column picker."""

    def __init__(
        self,
        parent,
        columns: list[tuple[str, pl.DataType]],
        on_run: Callable[[], None],
        on_apply: Callable[[], None],
    ) -> None:
        self._columns = columns
        self._on_run = on_run
        self._on_apply = on_apply

        numeric = [n for n, d in columns if _is_numeric(d)]
        self._numeric_names = numeric

        with parent:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                self.algo_select = (
                    ui.select(
                        options={k: lbl for lbl, k in ALGORITHMS},
                        value="kmeans",
                        label="Algorithm",
                        on_change=lambda _e: self._refresh_params(),
                    )
                    .props("dense outlined")
                    .classes("w-56")
                )
                self.col_select = (
                    ui.select(
                        options={n: n for n in numeric} or {"—": "—"},
                        multiple=True,
                        value=[],
                        clearable=True,
                        label="Feature columns (numeric)",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-72")
                )
                self.scale_switch = ui.switch("Scale features", value=True).props(
                    "dense"
                )
                self.out_col_input = (
                    ui.input(value="cluster", label="Output column")
                    .props("dense outlined")
                    .classes("w-32")
                )

            self.param_container = ui.column().classes("w-full gap-2 mt-1")
            self._refresh_params()

            with ui.row().classes("items-center gap-2 mt-2"):
                ui.button(
                    "Run",
                    icon="play_arrow",
                    on_click=lambda _=None: on_run(),
                ).props("dense unelevated color=primary")
                ui.button(
                    "Apply",
                    icon="check",
                    on_click=lambda _=None: on_apply(),
                ).props("dense unelevated color=positive")

            self.meta = ui.label("").classes("text-xs opacity-50")
            self.plot_container = ui.column().classes("w-full")

    # ---- algorithm parameter switching ------------------------------------
    def _refresh_params(self) -> None:
        self.param_container.clear()
        algo = self.algo_select.value or "kmeans"
        with self.param_container:
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                if algo in ("kmeans", "minibatch_kmeans", "spectral", "ward",
                            "agglomerative", "birch", "gmm"):
                    self._n_clusters_input = (
                        ui.number(
                            value=3, min=2, max=1000, label="n_clusters"
                        )
                        .props("dense outlined")
                        .classes("w-28")
                    )
                if algo in ("kmeans", "minibatch_kmeans"):
                    self._init_select = (
                        ui.select(
                            {"k-means++": "k-means++", "random": "random"},
                            value="k-means++",
                            label="init",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                    self._n_init_input = (
                        ui.number(value=10, min=1, max=100, label="n_init")
                        .props("dense outlined")
                        .classes("w-24")
                    )
                if algo in ("kmeans", "minibatch_kmeans", "affinity_propagation",
                            "gmm"):
                    self._max_iter_input = (
                        ui.number(value=300, min=1, max=10000, label="max_iter")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                if algo in ("kmeans", "minibatch_kmeans", "gmm"):
                    self._random_state_input = (
                        ui.number(value=42, min=0, label="random_state")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                if algo == "minibatch_kmeans":
                    self._batch_size_input = (
                        ui.number(value=1024, min=1, label="batch_size")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                if algo == "affinity_propagation":
                    self._damping_input = (
                        ui.number(
                            value=0.5, min=0.5, max=1.0, step=0.05, label="damping"
                        )
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._preference_input = (
                        ui.input(value="", label="preference (blank=auto)")
                        .props("dense outlined")
                        .classes("w-40")
                    )
                if algo == "mean_shift":
                    self._bandwidth_input = (
                        ui.input(value="", label="bandwidth (blank=auto)")
                        .props("dense outlined")
                        .classes("w-40")
                    )
                    self._bin_seeding_switch = ui.switch(
                        "bin_seeding", value=False
                    ).props("dense")
                if algo == "spectral":
                    self._affinity_select = (
                        ui.select(
                            {"rbf": "rbf", "nearest_neighbors": "nearest_neighbors"},
                            value="rbf",
                            label="affinity",
                        )
                        .props("dense outlined")
                        .classes("w-44")
                    )
                    self._n_neighbors_input = (
                        ui.number(value=10, min=2, label="n_neighbors")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._assign_labels_select = (
                        ui.select(
                            {"kmeans": "kmeans", "discretize": "discretize"},
                            value="kmeans",
                            label="assign_labels",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                if algo == "agglomerative":
                    self._linkage_select = (
                        ui.select(
                            {
                                "ward": "ward",
                                "complete": "complete",
                                "average": "average",
                                "single": "single",
                            },
                            value="complete",
                            label="linkage",
                            on_change=lambda _e: self._refresh_agg_metric(),
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                    self._agg_metric_select = (
                        ui.select(
                            _METRIC_OPTIONS,
                            value="euclidean",
                            label="metric",
                            on_change=lambda _e: self._refresh_agg_metric(),
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                if algo in ("ward", "agglomerative"):
                    self._dist_threshold_input = (
                        ui.input(
                            value="", label="distance_threshold (blank=None)"
                        )
                        .props("dense outlined")
                        .classes("w-48")
                    )
                if algo in ("dbscan", "optics"):
                    self._eps_input = (
                        ui.number(value=0.5, min=0.01, step=0.1, label="eps")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                if algo in ("dbscan", "hdbscan", "optics"):
                    self._min_samples_input = (
                        ui.number(value=5, min=2, label="min_samples")
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._cluster_metric_select = (
                        ui.select(
                            _METRIC_OPTIONS,
                            value="euclidean",
                            label="metric",
                        )
                        .props("dense outlined")
                        .classes("w-32")
                    )
                if algo == "hdbscan":
                    self._min_cluster_size_input = (
                        ui.number(value=5, min=2, label="min_cluster_size")
                        .props("dense outlined")
                        .classes("w-32")
                    )
                    self._cluster_selection_select = (
                        ui.select(
                            {"eom": "eom", "leaf": "leaf"},
                            value="eom",
                            label="cluster_selection_method",
                        )
                        .props("dense outlined")
                        .classes("w-48")
                    )
                if algo == "optics":
                    self._max_eps_input = (
                        ui.input(value="", label="max_eps (blank=inf)")
                        .props("dense outlined")
                        .classes("w-36")
                    )
                    self._cluster_method_select = (
                        ui.select(
                            {"dbscan": "dbscan", "xi": "xi"},
                            value="dbscan",
                            label="cluster_method",
                        )
                        .props("dense outlined")
                        .classes("w-36")
                    )
                if algo == "birch":
                    self._threshold_input = (
                        ui.number(
                            value=0.5, min=0.01, step=0.1, label="threshold"
                        )
                        .props("dense outlined")
                        .classes("w-28")
                    )
                    self._branching_factor_input = (
                        ui.number(value=50, min=2, label="branching_factor")
                        .props("dense outlined")
                        .classes("w-32")
                    )
                if algo == "gmm":
                    self._covariance_select = (
                        ui.select(
                            {
                                "full": "full",
                                "tied": "tied",
                                "diag": "diag",
                                "spherical": "spherical",
                            },
                            value="full",
                            label="covariance_type",
                        )
                        .props("dense outlined")
                        .classes("w-36")
                    )

    def _refresh_agg_metric(self) -> None:
        if not hasattr(self, "_agg_metric_select"):
            return
        linkage = self._linkage_select.value
        if linkage == "ward":
            self._agg_metric_select.value = "euclidean"
            self._agg_metric_select.set_visibility(False)
        else:
            self._agg_metric_select.set_visibility(True)

    # ---- spec accessor ----------------------------------------------------
    @property
    def spec(self) -> ClusterSpec:
        algo = self.algo_select.value or "kmeans"
        columns = list(self.col_select.value or [])
        scale = bool(self.scale_switch.value)
        out_col = (self.out_col_input.value or "cluster").strip() or "cluster"

        def _num(attr: str, default):
            w = getattr(self, attr, None)
            if w is None:
                return default
            v = w.value
            return v if v is not None else default

        def _opt_float(attr: str) -> float | None:
            w = getattr(self, attr, None)
            if w is None:
                return None
            v = (w.value or "").strip() if isinstance(w.value, str) else w.value
            if v == "" or v is None:
                return None
            return float(v)

        return ClusterSpec(
            algorithm=algo,
            columns=columns,
            scale=scale,
            output_column=out_col,
            n_clusters=int(_num("_n_clusters_input", 3)),
            random_state=int(_num("_random_state_input", 42)),
            max_iter=int(_num("_max_iter_input", 300)),
            init=getattr(self, "_init_select", None) and self._init_select.value or "k-means++",
            n_init=int(_num("_n_init_input", 10)),
            batch_size=int(_num("_batch_size_input", 1024)),
            damping=float(_num("_damping_input", 0.5)),
            preference=_opt_float("_preference_input"),
            bandwidth=_opt_float("_bandwidth_input"),
            bin_seeding=bool(
                getattr(self, "_bin_seeding_switch", None)
                and self._bin_seeding_switch.value
            ),
            affinity=getattr(self, "_affinity_select", None) and self._affinity_select.value or "rbf",
            n_neighbors=int(_num("_n_neighbors_input", 10)),
            assign_labels=getattr(self, "_assign_labels_select", None) and self._assign_labels_select.value or "kmeans",
            linkage=getattr(self, "_linkage_select", None) and self._linkage_select.value or "ward",
            metric=getattr(self, "_cluster_metric_select", None) and self._cluster_metric_select.value
                or getattr(self, "_agg_metric_select", None) and self._agg_metric_select.value
                or "euclidean",
            distance_threshold=_opt_float("_dist_threshold_input"),
            eps=float(_num("_eps_input", 0.5)),
            min_samples=int(_num("_min_samples_input", 5)),
            min_cluster_size=int(_num("_min_cluster_size_input", 5)),
            cluster_selection_method=getattr(self, "_cluster_selection_select", None) and self._cluster_selection_select.value or "eom",
            max_eps=_opt_float("_max_eps_input"),
            cluster_method=getattr(self, "_cluster_method_select", None) and self._cluster_method_select.value or "dbscan",
            threshold=float(_num("_threshold_input", 0.5)),
            branching_factor=int(_num("_branching_factor_input", 50)),
            covariance_type=getattr(self, "_covariance_select", None) and self._covariance_select.value or "full",
        )

    # ---- output slots -----------------------------------------------------
    def set_meta(self, text: str) -> None:
        self.meta.set_text(text)

    def render_plot(self, fig: dict) -> None:
        self.plot_container.clear()
        with self.plot_container:
            ui.plotly(fig).classes("w-full h-96")

    def render_error(self, msg: str) -> None:
        self.plot_container.clear()
        with self.plot_container:
            ui.label(msg).classes("text-sm text-negative")
