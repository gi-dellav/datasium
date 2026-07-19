"""Tests for the pure transform helpers in ``datasium.transform``."""

from __future__ import annotations

import polars as pl
import pytest

from datasium.transform import (
    add_computed_column,
    group_by_agg,
    join_frames,
    one_hot_encode,
    pivot_frame,
    rename_column,
    sort_frame,
    unpivot_frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.scan_csv("sample.csv")


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.read_csv("sample.csv")


# ---------------------------------------------------------------------------
# sort_frame
# ---------------------------------------------------------------------------
def test_sort_single_asc(lf):
    out = sort_frame(lf, ["age"]).collect()
    assert out["age"].to_list() == sorted([30, 12, 40, 25, 55, 9, 33, 28])


def test_sort_single_desc(lf):
    out = sort_frame(lf, ["age"], [True]).collect()
    assert out["age"].to_list() == sorted([30, 12, 40, 25, 55, 9, 33, 28], reverse=True)


def test_sort_multi(lf):
    out = sort_frame(lf, ["city", "age"], [False, True]).collect()
    cities = out["city"].to_list()
    assert cities == sorted(cities)
    # within London, ages should be descending
    london = out.filter(pl.col("city") == "London")
    assert london["age"].to_list() == [40, 33, 30]


def test_sort_empty_raises(lf):
    with pytest.raises(ValueError, match="at least one column"):
        sort_frame(lf, [])


def test_sort_unknown_column(lf):
    with pytest.raises(ValueError, match="not found"):
        sort_frame(lf, ["bogus"])


def test_sort_desc_length_mismatch(lf):
    with pytest.raises(ValueError, match="must match"):
        sort_frame(lf, ["age", "name"], [True])


# ---------------------------------------------------------------------------
# rename_column
# ---------------------------------------------------------------------------
def test_rename(lf):
    out = rename_column(lf, "age", "years").collect()
    assert "years" in out.columns
    assert "age" not in out.columns
    assert out["years"].to_list() == [30, 12, 40, 25, 55, 9, 33, 28]


def test_rename_unknown(lf):
    with pytest.raises(ValueError, match="not found"):
        rename_column(lf, "bogus", "x")


def test_rename_empty_name(lf):
    with pytest.raises(ValueError, match="enter a new column name"):
        rename_column(lf, "age", "  ")


def test_rename_duplicate(lf):
    with pytest.raises(ValueError, match="already exists"):
        rename_column(lf, "age", "name")


def test_rename_same_name_is_noop(lf):
    out = rename_column(lf, "age", "age").collect()
    assert "age" in out.columns


# ---------------------------------------------------------------------------
# add_computed_column – arithmetic
# ---------------------------------------------------------------------------
def test_arith_add_two_cols(lf):
    out = add_computed_column(
        lf,
        "age_plus_score",
        "arithmetic",
        "add",
        col_a="age",
        col_b="score",
    ).collect()
    expected = [
        30 + 8.5,
        12 + 3.2,
        40 + 9.1,
        25 + 5.0,
        55 + 7.7,
        9 + 2.1,
        33 + 6.4,
        28 + 4.9,
    ]
    assert out["age_plus_score"].to_list() == pytest.approx(expected)


def test_arith_sub_scalar(lf):
    out = add_computed_column(
        lf,
        "age_minus_10",
        "arithmetic",
        "sub",
        col_a="age",
        scalar="10",
    ).collect()
    assert out["age_minus_10"].to_list() == [20, 2, 30, 15, 45, -1, 23, 18]


def test_arith_mul(lf):
    out = add_computed_column(
        lf,
        "double_age",
        "arithmetic",
        "mul",
        col_a="age",
        scalar="2",
    ).collect()
    assert out["double_age"].to_list() == [60, 24, 80, 50, 110, 18, 66, 56]


def test_arith_div(lf):
    out = add_computed_column(
        lf,
        "half_score",
        "arithmetic",
        "div",
        col_a="score",
        scalar="2",
    ).collect()
    assert out["half_score"].to_list() == pytest.approx(
        [4.25, 1.6, 4.55, 2.5, 3.85, 1.05, 3.2, 2.45]
    )


def test_arith_mod(lf):
    out = add_computed_column(
        lf,
        "age_mod_10",
        "arithmetic",
        "mod",
        col_a="age",
        scalar="10",
    ).collect()
    assert out["age_mod_10"].to_list() == [0, 2, 0, 5, 5, 9, 3, 8]


def test_arith_pow(lf):
    out = add_computed_column(
        lf,
        "age_sq",
        "arithmetic",
        "pow",
        col_a="age",
        scalar="2",
    ).collect()
    assert out["age_sq"].to_list() == pytest.approx(
        [900, 144, 1600, 625, 3025, 81, 1089, 784]
    )


def test_arith_floordiv(lf):
    out = add_computed_column(
        lf,
        "age_floordiv_10",
        "arithmetic",
        "floordiv",
        col_a="age",
        scalar="10",
    ).collect()
    assert out["age_floordiv_10"].to_list() == [3, 1, 4, 2, 5, 0, 3, 2]


def test_arith_no_col_a(lf):
    with pytest.raises(ValueError, match="select column A"):
        add_computed_column(lf, "x", "arithmetic", "add", col_b="age")


def test_arith_no_col_b_or_scalar(lf):
    with pytest.raises(ValueError, match="column B or supply a scalar"):
        add_computed_column(lf, "x", "arithmetic", "add", col_a="age")


def test_arith_non_numeric(lf):
    with pytest.raises(ValueError, match="not numeric"):
        add_computed_column(lf, "x", "arithmetic", "add", col_a="name", col_b="age")


# ---------------------------------------------------------------------------
# add_computed_column – aggregation broadcast
# ---------------------------------------------------------------------------
def test_agg_sum(lf):
    out = add_computed_column(
        lf,
        "total_age",
        "aggregation",
        "sum",
        col_a="age",
    ).collect()
    assert out["total_age"].to_list() == [232] * 8


def test_agg_mean(lf):
    out = add_computed_column(
        lf,
        "avg_age",
        "aggregation",
        "mean",
        col_a="age",
    ).collect()
    assert out["avg_age"].to_list() == pytest.approx([29.0] * 8)


def test_agg_min_max(lf):
    out = add_computed_column(
        lf,
        "min_age",
        "aggregation",
        "min",
        col_a="age",
    ).collect()
    assert out["min_age"].to_list() == [9] * 8
    out2 = add_computed_column(
        lf,
        "max_age",
        "aggregation",
        "max",
        col_a="age",
    ).collect()
    assert out2["max_age"].to_list() == [55] * 8


def test_agg_count(lf):
    out = add_computed_column(
        lf,
        "n",
        "aggregation",
        "count",
        col_a="age",
    ).collect()
    assert out["n"].to_list() == [8] * 8


def test_agg_no_col(lf):
    with pytest.raises(ValueError, match="select a source column"):
        add_computed_column(lf, "x", "aggregation", "sum")


# ---------------------------------------------------------------------------
# add_computed_column – cumulative
# ---------------------------------------------------------------------------
def test_cum_sum(lf):
    out = add_computed_column(
        lf,
        "running_age",
        "cumulative",
        "cum_sum",
        col_a="age",
    ).collect()
    ages = [30, 12, 40, 25, 55, 9, 33, 28]
    expected = []
    acc = 0
    for a in ages:
        acc += a
        expected.append(acc)
    assert out["running_age"].to_list() == expected


def test_cum_count(lf):
    out = add_computed_column(
        lf,
        "row_num",
        "cumulative",
        "cum_count",
        col_a="age",
    ).collect()
    assert out["row_num"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_cum_min_max(lf):
    out = add_computed_column(
        lf,
        "run_min",
        "cumulative",
        "cum_min",
        col_a="age",
    ).collect()
    assert out["run_min"].to_list() == [30, 12, 12, 12, 12, 9, 9, 9]


# ---------------------------------------------------------------------------
# add_computed_column – string
# ---------------------------------------------------------------------------
def test_str_upper(lf):
    out = add_computed_column(
        lf,
        "upper_name",
        "string",
        "upper",
        col_a="name",
    ).collect()
    assert out["upper_name"].to_list() == [
        "ADA",
        "BO",
        "CY",
        "DE",
        "ED",
        "FINN",
        "GIO",
        "HAL",
    ]


def test_str_lower(lf):
    out = add_computed_column(
        lf,
        "lower_city",
        "string",
        "lower",
        col_a="city",
    ).collect()
    assert out["lower_city"].to_list() == [
        "london",
        "paris",
        "london",
        "paris",
        "rome",
        "rome",
        "london",
        "paris",
    ]


def test_str_len(lf):
    out = add_computed_column(
        lf,
        "name_len",
        "string",
        "str_len",
        col_a="name",
    ).collect()
    assert out["name_len"].to_list() == [3, 2, 2, 2, 2, 4, 3, 3]


def test_str_reverse(lf):
    out = add_computed_column(
        lf,
        "rev_name",
        "string",
        "reverse",
        col_a="name",
    ).collect()
    assert out["rev_name"].to_list() == [
        "adA",
        "oB",
        "yC",
        "eD",
        "dE",
        "nniF",
        "oiG",
        "laH",
    ]


def test_str_title(lf):
    df2 = pl.DataFrame({"s": ["hello world", "FOO BAR"]}).lazy()
    out = add_computed_column(df2, "t", "string", "title", col_a="s").collect()
    assert out["t"].to_list() == ["Hello World", "Foo Bar"]


def test_str_strip():
    df2 = pl.DataFrame({"s": ["  hi  ", " ok "]}).lazy()
    out = add_computed_column(df2, "t", "string", "strip", col_a="s").collect()
    assert out["t"].to_list() == ["hi", "ok"]


# ---------------------------------------------------------------------------
# add_computed_column – rank / index
# ---------------------------------------------------------------------------
def test_rank(lf):
    out = add_computed_column(
        lf,
        "age_rank",
        "rank / index",
        "rank",
        col_a="age",
    ).collect()
    # ages: 30,12,40,25,55,9,33,28 → sorted: 9,12,25,28,30,33,40,55
    # dense ranks: 9→1, 12→2, 25→3, 28→4, 30→5, 33→6, 40→7, 55→8
    assert out["age_rank"].to_list() == [5, 2, 7, 3, 8, 1, 6, 4]


def test_row_index(lf):
    out = add_computed_column(
        lf,
        "idx",
        "rank / index",
        "row_index",
    ).collect()
    assert out["idx"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_rank_no_col(lf):
    with pytest.raises(ValueError, match="select a column to rank"):
        add_computed_column(lf, "x", "rank / index", "rank")


# ---------------------------------------------------------------------------
# add_computed_column – conditional
# ---------------------------------------------------------------------------
def test_cond_gt(lf):
    out = add_computed_column(
        lf,
        "label",
        "conditional",
        "cond_gt",
        col_a="age",
        scalar="30",
        then_value="old",
        else_value="young",
    ).collect()
    assert out["label"].to_list() == [
        "young",
        "young",
        "old",
        "young",
        "old",
        "young",
        "old",
        "young",
    ]


def test_cond_lt(lf):
    out = add_computed_column(
        lf,
        "label",
        "conditional",
        "cond_lt",
        col_a="age",
        scalar="20",
        then_value="kid",
        else_value="adult",
    ).collect()
    assert out["label"].to_list() == [
        "adult",
        "kid",
        "adult",
        "adult",
        "adult",
        "kid",
        "adult",
        "adult",
    ]


def test_cond_eq(lf):
    out = add_computed_column(
        lf,
        "is_london",
        "conditional",
        "cond_eq",
        col_a="city",
        scalar="London",
        then_value="yes",
        else_value="no",
    ).collect()
    assert out["is_london"].to_list() == [
        "yes",
        "no",
        "yes",
        "no",
        "no",
        "no",
        "yes",
        "no",
    ]


def test_cond_null():
    df2 = pl.DataFrame({"a": [1, None, 3, None]}).lazy()
    out = add_computed_column(
        df2,
        "filled",
        "conditional",
        "cond_null",
        col_a="a",
        then_value="missing",
        else_value="present",
    ).collect()
    assert out["filled"].to_list() == ["present", "missing", "present", "missing"]


# ---------------------------------------------------------------------------
# add_computed_column – validation
# ---------------------------------------------------------------------------
def test_computed_no_name(lf):
    with pytest.raises(ValueError, match="enter a name"):
        add_computed_column(lf, "  ", "arithmetic", "add", col_a="age", scalar="1")


def test_computed_duplicate_name(lf):
    with pytest.raises(ValueError, match="already exists"):
        add_computed_column(lf, "age", "arithmetic", "add", col_a="age", scalar="1")


def test_computed_unknown_category(lf):
    with pytest.raises(ValueError, match="unknown category"):
        add_computed_column(lf, "x", "bogus", "add", col_a="age")


def test_computed_unknown_op(lf):
    with pytest.raises(ValueError, match="unknown"):
        add_computed_column(lf, "x", "arithmetic", "bogus", col_a="age", scalar="1")


def test_computed_unknown_col(lf):
    with pytest.raises(ValueError, match="not found"):
        add_computed_column(lf, "x", "arithmetic", "add", col_a="bogus", scalar="1")


# ---------------------------------------------------------------------------
# group_by_agg
# ---------------------------------------------------------------------------
def test_group_by_mean(lf):
    out = group_by_agg(lf, ["city"], "score", "mean", "avg_score").collect()
    assert out.columns == ["city", "avg_score"]
    assert out.height == 3
    london = out.filter(pl.col("city") == "London")
    assert london["avg_score"].item() == pytest.approx((8.5 + 9.1 + 6.4) / 3)


def test_group_by_sum(lf):
    out = group_by_agg(lf, ["city"], "age", "sum", "total_age").collect()
    london = out.filter(pl.col("city") == "London")
    assert london["total_age"].item() == 30 + 40 + 33


def test_group_by_count(lf):
    out = group_by_agg(lf, ["city"], None, "count", "n").collect()
    london = out.filter(pl.col("city") == "London")
    assert london["n"].item() == 3


def test_group_by_multi_cols(lf):
    out = group_by_agg(lf, ["city"], "age", "max", "oldest").collect()
    assert out.height == 3
    rome = out.filter(pl.col("city") == "Rome")
    assert rome["oldest"].item() == 55


def test_group_by_n_unique(lf):
    out = group_by_agg(lf, ["city"], "name", "n_unique", "n_names").collect()
    london = out.filter(pl.col("city") == "London")
    assert london["n_names"].item() == 3


def test_group_by_no_cols(lf):
    with pytest.raises(ValueError, match="at least one group-by"):
        group_by_agg(lf, [], "age", "mean", "x")


def test_group_by_unknown_col(lf):
    with pytest.raises(ValueError, match="not found"):
        group_by_agg(lf, ["bogus"], "age", "mean", "x")


def test_group_by_no_output_name(lf):
    with pytest.raises(ValueError, match="enter a name"):
        group_by_agg(lf, ["city"], "age", "mean", "  ")


def test_group_by_count_no_agg_col_ok(lf):
    out = group_by_agg(lf, ["city"], None, "count", "n").collect()
    assert out.height == 3


def test_group_by_unknown_agg(lf):
    with pytest.raises(ValueError, match="unknown aggregation"):
        group_by_agg(lf, ["city"], "age", "bogus", "x")


def test_group_by_agg_col_not_found(lf):
    with pytest.raises(ValueError, match="not found"):
        group_by_agg(lf, ["city"], "bogus", "mean", "x")


# ---------------------------------------------------------------------------
# datetime extraction
# ---------------------------------------------------------------------------
@pytest.fixture
def dt_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {"ts": [datetime(2024, 3, 15, 10, 30), datetime(2025, 12, 25, 23, 59)]}
    ).lazy()


from datetime import datetime


def test_dt_year(dt_lf):
    out = add_computed_column(dt_lf, "yr", "datetime", "dt_year", col_a="ts").collect()
    assert out["yr"].to_list() == [2024, 2025]


def test_dt_month(dt_lf):
    out = add_computed_column(dt_lf, "mo", "datetime", "dt_month", col_a="ts").collect()
    assert out["mo"].to_list() == [3, 12]


def test_dt_hour(dt_lf):
    out = add_computed_column(dt_lf, "h", "datetime", "dt_hour", col_a="ts").collect()
    assert out["h"].to_list() == [10, 23]


def test_dt_quarter(dt_lf):
    out = add_computed_column(dt_lf, "q", "datetime", "dt_quarter", col_a="ts").collect()
    assert out["q"].to_list() == [1, 4]


def test_dt_rejects_non_temporal(lf):
    with pytest.raises(ValueError, match="not a date/time"):
        add_computed_column(lf, "x", "datetime", "dt_year", col_a="age")


# ---------------------------------------------------------------------------
# window functions
# ---------------------------------------------------------------------------
def test_lag(lf):
    out = add_computed_column(
        lf, "prev", "window", "lag", col_a="age", scalar="1"
    ).collect()
    assert out["prev"][0] is None
    assert out["prev"][1] == 30


def test_lead(lf):
    out = add_computed_column(
        lf, "next", "window", "lead", col_a="age", scalar="1"
    ).collect()
    assert out["next"][-1] is None
    assert out["next"][0] == 12


def test_diff(lf):
    out = add_computed_column(
        lf, "d", "window", "diff", col_a="age", scalar="1"
    ).collect()
    assert out["d"][0] is None
    assert out["d"][1] == 12 - 30


def test_rolling_mean(lf):
    out = add_computed_column(
        lf, "rm", "window", "rolling_mean", col_a="age", scalar="3"
    ).collect()
    # first 2 rows have incomplete windows → null
    assert out["rm"][0] is None
    assert out["rm"][1] is None
    assert out["rm"][2] == (30 + 12 + 40) / 3


def test_rolling_sum(lf):
    out = add_computed_column(
        lf, "rs", "window", "rolling_sum", col_a="age", scalar="2"
    ).collect()
    assert out["rs"][1] == 30 + 12


def test_window_rejects_non_numeric(lf):
    with pytest.raises(ValueError, match="not numeric"):
        add_computed_column(lf, "x", "window", "lag", col_a="name")


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
def test_bin_freq(lf):
    out = add_computed_column(
        lf, "bins", "binning", "bin_freq", col_a="age", scalar="3"
    ).collect()
    assert out["bins"].dtype == pl.Categorical
    assert out.height == 8


def test_bin_width(lf):
    out = add_computed_column(
        lf, "bins", "binning", "bin_width", col_a="age", scalar="3"
    ).collect()
    assert out["bins"].dtype == pl.Enum
    assert out.height == 8


def test_bin_rejects_few_bins(lf):
    with pytest.raises(ValueError, match="≥ 2"):
        add_computed_column(lf, "b", "binning", "bin_freq", col_a="age", scalar="1")


# ---------------------------------------------------------------------------
# one_hot_encode
# ---------------------------------------------------------------------------
def test_one_hot_basic(lf):
    out = one_hot_encode(lf, "city").collect()
    assert "city_London" in out.columns
    assert "city_Paris" in out.columns
    assert "city_Rome" in out.columns
    assert out["city_London"].dtype == pl.Boolean


def test_one_hot_unknown_column(lf):
    with pytest.raises(ValueError, match="not found"):
        one_hot_encode(lf, "bogus")


# ---------------------------------------------------------------------------
# pivot_frame
# ---------------------------------------------------------------------------
def test_pivot_basic():
    lf = pl.DataFrame(
        {"idx": ["a", "a", "b", "b"], "col": ["x", "y", "x", "y"], "val": [1, 2, 3, 4]}
    ).lazy()
    out = pivot_frame(lf, ["idx"], "col", "val", "first").collect()
    assert "x" in out.columns
    assert "y" in out.columns
    assert out.height == 2


def test_pivot_no_index():
    lf = pl.DataFrame({"a": [1], "b": [2]}).lazy()
    with pytest.raises(ValueError, match="at least one index"):
        pivot_frame(lf, [], "a", "b")


# ---------------------------------------------------------------------------
# unpivot_frame
# ---------------------------------------------------------------------------
def test_unpivot_basic():
    lf = pl.DataFrame({"id": [1, 2], "x": [10, 20], "y": [30, 40]}).lazy()
    out = unpivot_frame(lf, ["id"], ["x", "y"]).collect()
    assert out.height == 4
    assert "variable" in out.columns
    assert "value" in out.columns


def test_unpivot_custom_names():
    lf = pl.DataFrame({"id": [1], "a": [10]}).lazy()
    out = unpivot_frame(lf, ["id"], ["a"], "metric", "amount").collect()
    assert "metric" in out.columns
    assert "amount" in out.columns


# ---------------------------------------------------------------------------
# join_frames
# ---------------------------------------------------------------------------
def test_join_inner():
    left = pl.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]}).lazy()
    right = pl.DataFrame({"k": [2, 3, 4], "w": [10, 20, 30]}).lazy()
    out = join_frames(left, right, ["k"], ["k"], "inner").collect()
    assert out.height == 2
    assert "w" in out.columns


def test_join_left():
    left = pl.DataFrame({"k": [1, 2], "v": ["a", "b"]}).lazy()
    right = pl.DataFrame({"k": [2, 3], "w": [10, 20]}).lazy()
    out = join_frames(left, right, ["k"], ["k"], "left").collect()
    assert out.height == 2


def test_join_cross():
    left = pl.DataFrame({"a": [1, 2]}).lazy()
    right = pl.DataFrame({"b": [3, 4, 5]}).lazy()
    out = join_frames(left, right, [], [], "cross").collect()
    assert out.height == 6


def test_join_bad_how():
    left = pl.DataFrame({"k": [1]}).lazy()
    right = pl.DataFrame({"k": [1]}).lazy()
    with pytest.raises(ValueError, match="unknown join type"):
        join_frames(left, right, ["k"], ["k"], "banana")


def test_join_missing_left_col():
    left = pl.DataFrame({"k": [1]}).lazy()
    right = pl.DataFrame({"k": [1]}).lazy()
    with pytest.raises(ValueError, match="left column"):
        join_frames(left, right, ["missing"], ["k"])
