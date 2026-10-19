"""Data cleaning helpers for AutoVisCheck (pure pandas/numpy, no Streamlit).

Every cleaning step is described by a small dict ("op"). `apply_op` takes a
DataFrame and an op and returns (new_df, message). Because ops are plain dicts,
the app can undo a step by replaying the remaining ops on the original data.
"""
import numpy as np
import pandas as pd

MISSING_METHODS = [
    "Drop rows with missing values",
    "Fill with mean",
    "Fill with median",
    "Fill with mode (most frequent)",
    "Fill with constant",
    "Forward fill",
    "Backward fill",
    "Linear interpolation",
]

OUTLIER_METHODS = {
    # name: (default, min, max, step, label)
    "IQR (Tukey fences)": (1.5, 0.5, 5.0, 0.25, "IQR multiplier"),
    "Z-score": (3.0, 1.0, 6.0, 0.25, "Std-dev threshold"),
    "Modified Z-score (MAD)": (3.5, 2.0, 8.0, 0.25, "MAD threshold"),
    "Percentile": (1.0, 0.1, 10.0, 0.1, "% cut from each tail"),
}

OUTLIER_ACTIONS = [
    "Flag only (add outlier_flag column)",
    "Cap / Winsorize to bounds",
    "Replace with NaN (then impute)",
    "Replace with median",
    "Remove rows",
]

CONVERT_TYPES = ["Numeric", "Category", "Text", "Datetime"]
TRANSFORMS = ["Log (log1p)", "Standardize (z-score)", "Min-max scale (0-1)"]


def is_helper_col(name):
    """Columns created by the cleaning steps (never used as target/feature)."""
    return str(name).endswith("_was_missing") or name == "outlier_flag"


# ------------------------------------------------------------------
# Outlier detection
# ------------------------------------------------------------------
def outlier_bounds(series, method, param):
    """Return (lower, upper) fences; values outside are outliers."""
    s = series.dropna()
    if s.empty:
        return (np.nan, np.nan)
    if method == "IQR (Tukey fences)":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (q1 - param * iqr, q3 + param * iqr)
    if method == "Z-score":
        m, sd = s.mean(), s.std()
        if not sd or np.isnan(sd):
            return (-np.inf, np.inf)
        return (m - param * sd, m + param * sd)
    if method == "Modified Z-score (MAD)":
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return (-np.inf, np.inf)
        return (med - param * mad / 0.6745, med + param * mad / 0.6745)
    if method == "Percentile":
        return (s.quantile(param / 100), s.quantile(1 - param / 100))
    raise ValueError(f"Unknown outlier method: {method}")


def outlier_table(df, cols, method, param):
    """Return (report_df, masks, bounds, rows_with_any_outlier)."""
    rows, masks, bounds = [], {}, {}
    for c in cols:
        lo, hi = outlier_bounds(df[c], method, param)
        mask = (df[c] < lo) | (df[c] > hi)  # NaN compares False
        masks[c], bounds[c] = mask, (lo, hi)
        n_valid = int(df[c].notna().sum())
        n_out = int(mask.sum())
        rows.append({
            "Column": c, "Lower bound": lo, "Upper bound": hi,
            "Outliers": n_out,
            "Outlier %": round(100 * n_out / n_valid, 2) if n_valid else 0.0,
        })
    any_n = int(pd.concat(masks, axis=1).any(axis=1).sum()) if masks else 0
    return pd.DataFrame(rows), masks, bounds, any_n


# ------------------------------------------------------------------
# Applying one cleaning step
# ------------------------------------------------------------------
def _float(df, c):
    """Integer columns can't hold NaN/fractions -> promote to float first."""
    if pd.api.types.is_integer_dtype(df[c]):
        df[c] = df[c].astype(float)


def apply_op(df, op):
    """Apply one cleaning step. Returns (new_df, human-readable message)."""
    t = op["type"]
    df = df.copy()

    if t == "drop_duplicates":
        n0 = len(df)
        df = df.drop_duplicates()
        return df, f"Removed {n0 - len(df):,} duplicate rows"

    if t == "drop_columns":
        cols = [c for c in op["cols"] if c in df.columns]
        df = df.drop(columns=cols)
        shown = ", ".join(cols[:6]) + ("…" if len(cols) > 6 else "")
        return df, f"Dropped {len(cols)} column(s): {shown}"

    if t == "missing":
        cols = [c for c in op["cols"] if c in df.columns]
        m = op["method"]
        before = int(df[cols].isna().sum().sum())
        if op.get("indicator"):
            for c in cols:
                if df[c].isna().any():
                    df[f"{c}_was_missing"] = df[c].isna()
        skipped, extra = [], ""
        if m == "Drop rows with missing values":
            n0 = len(df)
            df = df.dropna(subset=cols)
            extra = f", removed {n0 - len(df):,} rows"
        else:
            for c in cols:
                s = df[c]
                numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                if m in ("Fill with mean", "Fill with median", "Linear interpolation") and not numeric:
                    skipped.append(c)
                    continue
                if m == "Fill with mean":
                    _float(df, c)
                    df[c] = df[c].fillna(s.mean())
                elif m == "Fill with median":
                    _float(df, c)
                    df[c] = df[c].fillna(s.median())
                elif m == "Fill with mode (most frequent)":
                    mode = s.mode()
                    if len(mode):
                        df[c] = s.fillna(mode.iloc[0])
                elif m == "Fill with constant":
                    val = op.get("value", "")
                    if numeric:
                        try:
                            val = float(val)
                        except ValueError:
                            skipped.append(c)
                            continue
                        _float(df, c)
                    df[c] = df[c].fillna(val)
                elif m == "Forward fill":
                    df[c] = s.ffill()
                elif m == "Backward fill":
                    df[c] = s.bfill()
                elif m == "Linear interpolation":
                    _float(df, c)
                    df[c] = df[c].interpolate(limit_direction="both")
        after = int(df[cols].isna().sum().sum())
        msg = f"Missing values: '{m}' on {len(cols)} column(s) ({before:,} → {after:,} missing cells{extra})"
        if skipped:
            msg += f"; skipped {len(skipped)} unsuitable column(s)"
        return df, msg

    if t == "outliers":
        cols = [c for c in op["cols"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
        method, param, action = op["method"], op["param"], op["action"]
        _, masks, bounds, any_n = outlier_table(df, cols, method, param)
        n_cells = int(sum(m.sum() for m in masks.values()))
        any_mask = pd.concat(masks, axis=1).any(axis=1) if masks else pd.Series(False, index=df.index)
        if action == OUTLIER_ACTIONS[0]:
            df["outlier_flag"] = any_mask
            what = f"flagged {any_n:,} rows"
        elif action == OUTLIER_ACTIONS[1]:
            for c in cols:
                lo, hi = bounds[c]
                _float(df, c)
                df[c] = df[c].clip(lower=lo, upper=hi)
            what = f"capped {n_cells:,} values"
        elif action == OUTLIER_ACTIONS[2]:
            for c in cols:
                if masks[c].any():
                    _float(df, c)
                    df.loc[masks[c], c] = np.nan
            what = f"set {n_cells:,} values to NaN"
        elif action == OUTLIER_ACTIONS[3]:
            for c in cols:
                if masks[c].any():
                    med = df.loc[~masks[c], c].median()
                    _float(df, c)
                    df.loc[masks[c], c] = med
            what = f"replaced {n_cells:,} values with the column median"
        elif action == OUTLIER_ACTIONS[4]:
            n0 = len(df)
            df = df[~any_mask]
            what = f"removed {n0 - len(df):,} rows"
        else:
            raise ValueError(f"Unknown outlier action: {action}")
        return df, f"Outliers: {method} ({param}) on {len(cols)} column(s) – {what}"

    if t == "convert":
        c, to = op["col"], op["to"]
        if c not in df.columns:
            return df, f"Column '{c}' not found"
        before_na = int(df[c].isna().sum())
        if to == "Numeric":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif to == "Category":
            df[c] = df[c].astype("category")
        elif to == "Text":
            df[c] = df[c].astype("object")
        elif to == "Datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        new_na = int(df[c].isna().sum()) - before_na
        return df, f"Converted '{c}' to {to}" + (f" ({new_na:,} unparseable values became NaN)" if new_na > 0 else "")

    if t == "transform":
        how, made, skipped = op["how"], [], []
        for c in op["cols"]:
            if c not in df.columns:
                continue
            s = df[c].astype(float)
            if how == TRANSFORMS[0]:
                if s.min() <= -1:
                    skipped.append(c)
                    continue
                df[f"{c}_log1p"] = np.log1p(s)
                made.append(f"{c}_log1p")
            elif how == TRANSFORMS[1]:
                sd = s.std()
                df[f"{c}_z"] = (s - s.mean()) / sd if sd else 0.0
                made.append(f"{c}_z")
            elif how == TRANSFORMS[2]:
                rng = s.max() - s.min()
                df[f"{c}_minmax"] = (s - s.min()) / rng if rng else 0.0
                made.append(f"{c}_minmax")
        msg = f"Transform '{how}': created {len(made)} new column(s)"
        if skipped:
            msg += f" (skipped {len(skipped)} with values ≤ -1)"
        return df, msg

    raise ValueError(f"Unknown op type: {t}")


def make_log(step_no, msg, df):
    return {
        "Step": step_no,
        "Action": msg,
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing cells": int(df.isna().sum().sum()),
    }
