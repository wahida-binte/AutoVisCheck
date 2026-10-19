import numpy as np
import pandas as pd
import pytest

from autovischeck.cleaning import (
    OUTLIER_ACTIONS, OUTLIER_METHODS, apply_op, is_helper_col, make_log, outlier_bounds, outlier_table,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 100, np.nan],
        "b": ["x", "y", None, "x", "x", "y"],
        "c": [1, 1, 1, 1, 1, 1],
    })


def test_drop_duplicates():
    d = pd.DataFrame({"a": [1, 1, 2]})
    out, msg = apply_op(d, {"type": "drop_duplicates"})
    assert len(out) == 2 and "1" in msg


def test_drop_columns_ignores_missing(df):
    out, _ = apply_op(df, {"type": "drop_columns", "cols": ["c", "nope"]})
    assert "c" not in out.columns


def test_apply_op_does_not_mutate_input(df):
    before = df.copy()
    apply_op(df, {"type": "missing", "cols": ["a"], "method": "Fill with median"})
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("method", ["Fill with mean", "Fill with median", "Linear interpolation"])
def test_numeric_fill(df, method):
    out, _ = apply_op(df, {"type": "missing", "cols": ["a"], "method": method})
    assert out["a"].isna().sum() == 0


def test_numeric_method_skips_text(df):
    out, msg = apply_op(df, {"type": "missing", "cols": ["b"], "method": "Fill with mean"})
    assert out["b"].isna().sum() == 1 and "skipped" in msg


def test_mode_fill_and_indicator(df):
    out, _ = apply_op(df, {"type": "missing", "cols": ["b"], "method": "Fill with mode (most frequent)",
                           "indicator": True})
    assert out["b"].isna().sum() == 0
    assert out["b_was_missing"].sum() == 1
    assert is_helper_col("b_was_missing")


def test_drop_rows(df):
    out, _ = apply_op(df, {"type": "missing", "cols": ["a"], "method": "Drop rows with missing values"})
    assert len(out) == 5


@pytest.mark.parametrize("method", list(OUTLIER_METHODS))
def test_outlier_bounds_finite_or_inf(method):
    s = pd.Series(np.r_[np.random.default_rng(0).normal(0, 1, 200), 50])
    lo, hi = outlier_bounds(s, method, OUTLIER_METHODS[method][0])
    assert lo < hi


def test_iqr_detects_extreme(df):
    table, masks, _, any_n = outlier_table(df, ["a"], "IQR (Tukey fences)", 1.5)
    assert any_n == 1 and masks["a"].iloc[4]


@pytest.mark.parametrize("action", OUTLIER_ACTIONS)
def test_outlier_actions(df, action):
    out, _ = apply_op(df, {"type": "outliers", "cols": ["a"], "method": "IQR (Tukey fences)",
                           "param": 1.5, "action": action})
    if action == OUTLIER_ACTIONS[4]:
        assert len(out) == len(df) - 1
    elif action == OUTLIER_ACTIONS[0]:
        assert out["outlier_flag"].sum() == 1
    else:
        assert (out["a"] == 100).sum() == 0


def test_convert_numeric_counts_coercions():
    d = pd.DataFrame({"x": ["1", "2", "oops"]})
    out, msg = apply_op(d, {"type": "convert", "col": "x", "to": "Numeric"})
    assert out["x"].isna().sum() == 1 and "unparseable" in msg


def test_transform_log_skips_invalid():
    d = pd.DataFrame({"p": [0.0, 1.0, 9.0], "n": [-5.0, 0.0, 1.0]})
    out, msg = apply_op(d, {"type": "transform", "cols": ["p", "n"], "how": "Log (log1p)"})
    assert "p_log1p" in out and "n_log1p" not in out and "skipped" in msg


def test_unknown_op_raises(df):
    with pytest.raises(ValueError):
        apply_op(df, {"type": "bogus"})


def test_make_log(df):
    log = make_log(1, "msg", df)
    assert log["Rows"] == 6 and log["Missing cells"] == 2
