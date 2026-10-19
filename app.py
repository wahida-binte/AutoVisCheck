import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from autovischeck.cleaning import (
    CONVERT_TYPES, MISSING_METHODS, OUTLIER_ACTIONS, OUTLIER_METHODS, TRANSFORMS,
    apply_op, is_helper_col, make_log, outlier_table,
)

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="AutoVisCheck", layout="wide")
st.title("📊 AutoVisCheck: Intelligent Visualization System")
st.info(
    "Follow the EDA workflow: **1. Clean & Prepare Data → 2. Analyze Data → "
    "3. Interpret Results → 4. Communicate Findings**."
)
st.markdown("---")

ss = st.session_state


_pc_counter = 0  # reset on every script rerun (module code re-executes top to bottom)


def pc(fig):
    """Render a Plotly chart with a unique, run-stable key (avoids DuplicateElementId)."""
    global _pc_counter
    _pc_counter += 1
    st.plotly_chart(fig, width="stretch", key=f"pc_{_pc_counter}")


def show_df(d, **kw):
    st.dataframe(d, width="stretch", **kw)


# =============================
# DATA LOADING (once per file; cleaned copy lives in session state)
# =============================
def make_default():
    rng = np.random.default_rng(42)
    n = 150
    group = rng.choice(["A", "B", "C"], n)
    shift = pd.Series(group).map({"A": 0, "B": 15, "C": 30}).values
    d = pd.DataFrame({
        "Category": group,
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Value": rng.normal(100, 15, n) + shift,
        "Score": rng.normal(50, 10, n) + shift * 0.5,
        "Age": rng.integers(18, 65, n).astype(float),
    })
    d.loc[rng.choice(n, 12, replace=False), "Score"] = np.nan
    d.loc[rng.choice(n, 6, replace=False), "Age"] = np.nan
    d.loc[rng.choice(n, 4, replace=False), "Region"] = None
    d.loc[rng.choice(n, 4, replace=False), "Value"] *= 3  # outliers
    return d.sort_index()


uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
src_id = f"{uploaded_file.name}-{uploaded_file.size}" if uploaded_file is not None else "default"

if ss.get("src_id") != src_id:
    raw = pd.read_csv(uploaded_file) if uploaded_file is not None else make_default()
    ss.update(src_id=src_id, raw_df=raw, clean_df=raw, ops=[], log=[], version=0)

if uploaded_file is None:
    st.warning("Using default dataset (upload a CSV to use your own)")

df = ss.clean_df
ver = f"{ss.src_id}:{ss.version}"  # changes after every cleaning step / new file -> invalidates caches & widget keys


def rebuild():
    """Replay all ops on the raw data (used by undo / reset)."""
    d, log = ss.raw_df, []
    for i, op in enumerate(ss.ops, 1):
        d, msg = apply_op(d, op)
        log.append(make_log(i, msg, d))
    ss.clean_df, ss.log = d, log
    ss.version += 1


def run_op(op):
    d, msg = apply_op(ss.clean_df, op)
    ss.clean_df = d
    ss.ops.append(op)
    ss.log.append(make_log(len(ss.ops), msg, d))
    ss.version += 1
    ss.flash = msg
    st.rerun()


def history_controls(prefix):
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("↩️ Undo last step", disabled=not ss.ops, key=f"{prefix}_undo"):
        ss.ops.pop()
        rebuild()
        ss.flash = "Last step undone"
        st.rerun()
    if c2.button("🔄 Reset all", disabled=not ss.ops, key=f"{prefix}_reset"):
        ss.ops = []
        rebuild()
        ss.flash = "All cleaning steps reset"
        st.rerun()


if df.empty or df.shape[1] == 0:
    st.error("The dataset is empty after cleaning. Undo the last step or reset.")
    history_controls("empty")
    st.stop()


# =============================
# CACHED HELPERS (keyed on `ver`; `_df` is excluded from hashing)
# =============================
@st.cache_data(show_spinner=False)
def c_missing(_df, v):
    return _df.isna().sum()


@st.cache_data(show_spinner=False)
def c_nunique(_df, v):
    return _df.nunique()


@st.cache_data(show_spinner=False)
def c_dups(_df, v):
    return int(_df.duplicated().sum())


@st.cache_data(show_spinner=False)
def c_summary(_df, v):
    return _df.describe(include="all").T


@st.cache_data(show_spinner=False)
def c_corr(_df, cols, method, v):
    return _df[list(cols)].corr(method=method)


@st.cache_data(show_spinner=False)
def c_skew(_df, cols, v):
    return _df[list(cols)].skew()


@st.cache_data(show_spinner=False)
def c_outliers(_df, cols, method, param, v):
    table, _, _, any_n = outlier_table(_df, list(cols), method, param)
    return table, any_n


miss = c_missing(df, ver)
nun = c_nunique(df, ver)
n_rows = len(df)

# =============================
# PLOT SAMPLING
# =============================
max_rows = st.sidebar.slider("Max rows used for plots", 1000, 100000, 20000, step=1000)
if n_rows > max_rows:
    plot_df = df.sample(max_rows, random_state=0)
    st.sidebar.caption(f"Plots use a random sample of {max_rows:,} of {n_rows:,} rows. "
                       "Statistics and tables use all rows.")
else:
    plot_df = df

# =============================
# DATA TYPES
# =============================
num_cols = [c for c in df.select_dtypes(include="number").columns if miss[c] < n_rows]
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
cat_small = [c for c in cat_cols if nun[c] <= 30]  # sensible for grouping/colouring


def is_discrete(col, max_unique=10):
    return nun[col] <= max_unique


# =============================
# TARGET / FEATURE / CHART RECOMMENDATION
# =============================
def detect_target(d):
    for col in d.columns:
        if col.lower() in ["target", "label", "class", "output", "y"]:
            return col
    for col in cat_cols:
        if not is_helper_col(col) and nun[col] <= 10:
            return col
    for col in num_cols:
        if not is_helper_col(col) and nun[col] <= 10:
            return col
    if num_cols:
        return d[num_cols].var().idxmax()
    return d.columns[-1]


def select_feature(d, target):
    cols = [c for c in d.columns if c != target and not is_helper_col(c)]
    if not cols:
        return target
    for col in cat_cols:
        if col != target and col in cols:
            return col
    return cols[0]


target_col = detect_target(df)
feature_col = select_feature(df, target_col)

CHART_OPTIONS = ["Bar", "Line", "Pie", "Scatter", "Box", "Violin", "Histogram",
                 "Strip", "ECDF", "Density Heatmap"]


def recommend_chart(target, feature):
    t_num, f_num = target in num_cols, feature in num_cols
    if t_num and f_num and not is_discrete(target):
        return "Scatter"
    if t_num or f_num:
        return "Box"
    return "Bar"


recommended_chart = recommend_chart(target_col, feature_col)


@st.cache_data(show_spinner=False)
def c_importance(_df, target, cols, v):
    cols = list(cols)
    if target in cols:
        others = [c for c in cols if c != target]
        if not others:
            return pd.Series(dtype=float)
        return _df[cols].corr()[target].drop(target).abs().sort_values(ascending=False).head(3)
    imp = {}
    for col in cols:
        try:
            g = _df.groupby(target, observed=True)[col].mean()
            imp[col] = g.max() - g.min()
        except Exception:
            continue
    return pd.Series(imp, dtype=float).sort_values(ascending=False).head(3)


# =============================
# SIDEBAR CONTROLS (chart builder)
# =============================
st.sidebar.header("⚙️ Chart Controls")
chart_type = st.sidebar.selectbox("Chart Type (Auto-selected)", CHART_OPTIONS,
                                  index=CHART_OPTIONS.index(recommended_chart))
x_col = st.sidebar.selectbox("X-axis", df.columns, index=list(df.columns).index(feature_col))
y_col = st.sidebar.selectbox("Y-axis", df.columns, index=list(df.columns).index(target_col))
color_col = st.sidebar.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
color_col = None if color_col == "None" else color_col
show_labels = st.sidebar.checkbox("Show Labels", True)
y_min_pct = st.sidebar.slider("Y-axis start (% of max, bar charts)", 0, 100, 0)
nbins = st.sidebar.slider("Histogram bins", 5, 100, 30)
points_mode = st.sidebar.selectbox("Box/Violin points", ["outliers", "all", False])

st.sidebar.markdown("---")
st.sidebar.caption(f"🧹 Cleaning steps applied: **{len(ss.ops)}**")


# =============================
# FIGURE BUILDER
# =============================
def build_figure(chart, data, x, y, color):
    if chart == "Bar":
        if y in num_cols:
            fig = px.bar(data, x=x, y=y, color=color, text_auto=show_labels)
            if y_min_pct > 0:
                fig.update_yaxes(range=[data[y].max() * y_min_pct / 100, data[y].max()])
        else:
            fig = px.histogram(data, x=x, color=y if color is None else color,
                               barmode="group", text_auto=show_labels)
        return fig, None
    if chart == "Line":
        return px.line(data, x=x, y=y, color=color, markers=True), None
    if chart == "Pie":
        if y not in num_cols:
            return None, "Pie chart needs a numeric Y-axis (values)."
        return px.pie(data, names=x, values=y), None
    if chart == "Scatter":
        return px.scatter(data, x=x, y=y, color=color), None
    if chart in ("Box", "Violin", "Strip"):
        if y in num_cols:
            num, grp, horizontal = y, x, False
        elif x in num_cols:
            num, grp, horizontal = x, y, True
        else:
            return None, f"{chart} plot needs at least one numeric axis."
        if grp in num_cols and nun[grp] > 20:
            grp = None
        kw = dict(data_frame=data, color=color)
        kw.update(dict(x=num, y=grp, orientation="h") if horizontal else dict(x=grp, y=num))
        if chart == "Box":
            return px.box(points=points_mode, **kw), None
        if chart == "Violin":
            return px.violin(box=True, points=points_mode, **kw), None
        return px.strip(**kw), None
    if chart == "Histogram":
        col = x if x in num_cols else y
        if col not in num_cols:
            return px.histogram(data, x=x, color=color, text_auto=show_labels), None
        return px.histogram(data, x=col, color=color, nbins=nbins, marginal="box",
                            text_auto=show_labels), None
    if chart == "ECDF":
        col = x if x in num_cols else y
        if col not in num_cols:
            return None, "ECDF needs a numeric column."
        return px.ecdf(data, x=col, color=color), None
    if chart == "Density Heatmap":
        if x not in num_cols or y not in num_cols:
            return None, "Density heatmap needs numeric X and Y axes."
        return px.density_heatmap(data, x=x, y=y, marginal_x="histogram",
                                  marginal_y="histogram", color_continuous_scale="Viridis"), None
    return None, "Unknown chart type."


# =============================
# CHART GUIDANCE (which chart suits which pair of column types)
# =============================
def col_kind(c):
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Datetime"
    if c in num_cols:
        return "Numeric (discrete)" if is_discrete(c) else "Numeric (continuous)"
    return "Categorical"


def _cls(kind):
    return "T" if kind == "Datetime" else "N" if kind.startswith("Numeric") else "C"


# key = sorted pair of classes (N = numeric, C = categorical, T = datetime)
# value = (best, ok, avoid, tip); best/ok map chart -> short note
CHART_GUIDE = {
    ("N", "N"): (
        {"Scatter": "shows the relationship / correlation between two numbers",
         "Density Heatmap": "best when there are thousands of points (avoids overplotting)"},
        {"Line": "only if X is ordered (e.g. a sequence or index)",
         "Histogram": "shows one column only (X)",
         "ECDF": "shows one column only (X)"},
        ["Pie", "Box", "Violin", "Strip"],
        "Put the explanatory variable on X and the outcome on Y. Use 'Color by' with a category "
        "to compare groups.",
    ),
    ("C", "N"): (
        {"Box": "compares median, spread and outliers per category",
         "Violin": "compares the full distribution shape per category",
         "Strip": "shows every individual point per category"},
        {"Bar": "adds up the values per category (use for totals)",
         "Pie": "share of total; only for a few categories (≤ 8)",
         "Histogram": "distribution of the number, split by 'Color by'",
         "ECDF": "cumulative distribution, split by 'Color by'"},
        ["Scatter", "Line", "Density Heatmap"],
        "Put the category on X and the numeric column on Y (Bar, Pie and Line need a numeric Y).",
    ),
    ("C", "C"): (
        {"Bar": "counts rows for each combination of the two categories (grouped bars)"},
        {"Histogram": "grouped counts of X, split by Y"},
        ["Scatter", "Line", "Pie", "Box", "Violin", "Strip", "ECDF", "Density Heatmap"],
        "The chart counts rows. With many categories (> 15) the chart gets unreadable, "
        "so pick columns with fewer levels. A contingency heatmap is in Analyze → Bivariate.",
    ),
    ("N", "T"): (
        {"Line": "shows the trend of a number over time"},
        {"Scatter": "shows individual observations over time",
         "Bar": "only if there are few time periods"},
        ["Pie", "Box", "Violin", "Strip", "Density Heatmap", "ECDF"],
        "Put the date/time on X and the number on Y. If many rows share the same date, "
        "aggregate first or split with 'Color by'.",
    ),
    ("C", "T"): (
        {"Bar": "counts events per category over time",
         "Histogram": "counts rows per time bin, split by category"},
        {"Scatter": "shows when each category occurred"},
        ["Pie", "Box", "Violin", "Strip", "ECDF", "Density Heatmap"],
        "Put the date/time on X. To see a trend over time, add a numeric column on Y.",
    ),
    ("T", "T"): (
        {},
        {"Scatter": "compares two timestamps (rarely useful)"},
        ["Pie", "Box", "Violin", "Strip", "Density Heatmap", "ECDF", "Bar", "Line"],
        "Two date columns rarely make a useful chart. Pick another column, or create a numeric "
        "'duration' column first.",
    ),
}

SAME_COLUMN_GUIDE = {
    "N": ({"Histogram": "distribution of one numeric column", "ECDF": "cumulative distribution",
           "Box": "median, spread and outliers"}, {}, ["Scatter", "Line", "Pie"],
          "Same column on X and Y. Single-column charts (Histogram / ECDF) are the right choice."),
    "C": ({"Bar": "counts of each category"}, {"Histogram": "same counts as a histogram"},
          ["Scatter", "Line", "Pie", "Box", "Violin", "Strip"],
          "Same column on X and Y. For one categorical column, a count bar chart is the right choice."),
    "T": ({"Histogram": "how many rows fall in each time period"}, {},
          ["Scatter", "Pie", "Box", "Violin", "Strip"],
          "Same column on X and Y. A histogram shows how the data is spread over time."),
}


def chart_guidance(x, y):
    """Return (x_kind, y_kind, best, ok, avoid, tip, extra_note) for two columns."""
    kx, ky = col_kind(x), col_kind(y)
    cx, cy = _cls(kx), _cls(ky)
    if x == y:
        best, ok, avoid, tip = SAME_COLUMN_GUIDE[cx]
        return kx, ky, dict(best), dict(ok), list(avoid), tip, ""
    best, ok, avoid, tip = CHART_GUIDE[tuple(sorted((cx, cy)))]
    best, ok, avoid = dict(best), dict(ok), list(avoid)
    extra = ""
    # numeric columns with very few distinct values behave like categories
    disc = [c for c, k in ((x, kx), (y, ky)) if k == "Numeric (discrete)"]
    if disc:
        extra = (f"`{', '.join(disc)}` has ≤ 10 distinct values, so it behaves like a category. "
                 "Box / Violin / Bar charts also work well with it.")
        if (cx, cy) == ("N", "N"):
            for ch in ("Box", "Violin", "Strip", "Bar"):
                if ch in avoid:
                    avoid.remove(ch)
                ok.setdefault(ch, "treats the few-valued column as groups")
    return kx, ky, best, ok, avoid, tip, extra


CHEAT_SHEET = pd.DataFrame([
    ["Numeric  ×  Numeric", "Scatter, Density Heatmap", "Line (ordered X), Histogram / ECDF (one column)",
     "Pie, Box, Violin"],
    ["Categorical  ×  Numeric", "Box, Violin, Strip", "Bar (totals), Pie (≤ 8 categories)",
     "Scatter, Line"],
    ["Categorical  ×  Categorical", "Bar (grouped counts)", "Histogram", "Scatter, Line, Pie, Box"],
    ["Datetime  ×  Numeric", "Line", "Scatter, Bar (few periods)", "Pie, Box, Violin"],
    ["Datetime  ×  Categorical", "Bar, Histogram", "Scatter", "Pie, Box, Violin"],
    ["One numeric column", "Histogram, ECDF, Box, Violin", "", "Pie"],
    ["One categorical column", "Bar (counts), Pie (few categories)", "", "Scatter, Line"],
], columns=["Column types", "Best charts", "Also OK", "Avoid"])


# =============================
# KEY FINDINGS (auto-generated, used in Interpret + Communicate)
# =============================
def build_findings():
    out = []

    def add(icon, topic, text):
        out.append({"icon": icon, "topic": topic, "text": text})

    total_missing = int(miss.sum())
    pct = 100 * total_missing / (n_rows * df.shape[1])
    if total_missing == 0:
        add("✅", "Missing values", "No missing values remain.")
    else:
        worst = (miss / n_rows * 100).sort_values(ascending=False)
        worst = worst[worst > 0].head(5)
        listing = ", ".join(f"{c} ({v:.1f}%)" for c, v in worst.items())
        add("⚠️" if pct > 5 else "ℹ️", "Missing values",
            f"{total_missing:,} cells ({pct:.1f}%) are still missing. Most affected: {listing}.")

    dups = c_dups(df, ver)
    add("⚠️" if dups else "✅", "Duplicates", f"{dups:,} duplicate rows." if dups else "No duplicate rows.")

    const = [c for c in df.columns if nun[c] <= 1 and miss[c] < n_rows]
    if const:
        add("ℹ️", "Constant columns", f"No variation in: {', '.join(const[:8])}. They carry no information.")

    if num_cols:
        skew = c_skew(df, tuple(num_cols), ver)
        sk = skew[(skew.abs() > 1) & (pd.Series({c: nun[c] for c in skew.index}) > 10)]
        sk = sk.reindex(sk.abs().sort_values(ascending=False).index).head(5)
        if len(sk):
            add("ℹ️", "Skewed distributions",
                "Strongly skewed: " + ", ".join(f"{c} ({v:+.1f})" for c, v in sk.items())
                + ". Consider a log transform (Clean step 4) or use medians instead of means.")

        cont = [c for c in num_cols if nun[c] > 10][:40]
        if len(cont) >= 2:
            cm = c_corr(df, tuple(cont), "pearson", ver)
            pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
                     .rename("r").reset_index())
            pairs = pairs[pairs["r"].abs() >= 0.7]
            pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(5)
            if len(pairs):
                txt = "; ".join(f"{a} ↔ {b} (r={r:+.2f})" for a, b, r in pairs.itertuples(index=False))
                add("ℹ️", "Strong correlations",
                    txt + ". Highly correlated features are redundant; correlation is not causation.")
            else:
                add("✅", "Correlations", "No strong linear correlations (|r| ≥ 0.7) between numeric features.")

        cands = tuple(c for c in num_cols if nun[c] > 10)
        if cands:
            tbl, _ = c_outliers(df, cands, "IQR (Tukey fences)", 1.5, ver)
            heavy = tbl[tbl["Outlier %"] > 5].sort_values("Outlier %", ascending=False).head(5)
            if len(heavy):
                listing = ", ".join(f"{row['Column']} ({row['Outlier %']:.1f}%)" for _, row in heavy.iterrows())
                add("⚠️", "Outliers", "More than 5% of values fall outside the IQR fences in: " + listing
                    + ". Check whether they are errors or genuine extremes.")
            else:
                add("✅", "Outliers", "No numeric column has more than 5% IQR outliers.")

    if target_col in cat_cols or (target_col in num_cols and is_discrete(target_col)):
        dist = df[target_col].value_counts(normalize=True, dropna=True)
        if len(dist):
            top = dist.iloc[0]
            if top > 0.8:
                add("⚠️", "Class imbalance",
                    f"Target '{target_col}' is imbalanced: '{dist.index[0]}' makes up {top:.0%} of rows. "
                    "Accuracy alone will be misleading.")
            else:
                add("✅", "Target balance", f"Target '{target_col}' is reasonably balanced "
                                            f"(largest class {top:.0%}).")

    imp = c_importance(df, target_col, tuple(num_cols), ver)
    if len(imp):
        add("ℹ️", "Top features", f"Features most related to '{target_col}': " + ", ".join(imp.index) + ".")
    return out


# =============================
# TABS
# =============================
current_fig = None
tab_clean, tab_analyze, tab_interp, tab_comm = st.tabs(
    ["1️⃣ Clean & Prepare Data", "2️⃣ Analyze Data", "3️⃣ Interpret Results", "4️⃣ Communicate Findings"]
)

# =====================================================================
# TAB 1: CLEAN & PREPARE
# =====================================================================
with tab_clean:
    if ss.get("flash"):
        st.success(ss.pop("flash"))

    raw_df = ss.raw_df
    raw_missing = int(c_missing(raw_df, f"raw-{ss.src_id}").sum())
    dups = c_dups(df, ver)

    st.markdown("### Data quality snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{n_rows:,}", delta=n_rows - len(raw_df) or None)
    m2.metric("Columns", df.shape[1], delta=df.shape[1] - raw_df.shape[1] or None)
    m3.metric("Missing cells", f"{int(miss.sum()):,}", delta=int(miss.sum()) - raw_missing or None,
              delta_color="inverse")
    m4.metric("Duplicate rows", f"{dups:,}")

    st.markdown("#### Cleaning history")
    if ss.log:
        show_df(pd.DataFrame(ss.log), hide_index=True)
    else:
        st.caption("No cleaning steps applied yet. Every step can be undone.")
    history_controls("main")

    # ---- automatic suggestions
    empty_cols = miss[miss == n_rows].index.tolist()
    const_cols = [c for c in df.columns if nun[c] <= 1 and c not in empty_cols]
    index_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    sugg = []
    if empty_cols:
        sugg.append(f"**Completely empty columns:** {', '.join(empty_cols)}")
    if index_cols:
        sugg.append(f"**Looks like a saved row index:** {', '.join(index_cols)}")
    if const_cols:
        sugg.append(f"**Constant columns:** {', '.join(const_cols)}")
    if dups:
        sugg.append(f"**{dups:,} duplicate rows**")
    if sugg:
        st.info("💡 **Suggestions**\n\n" + "\n\n".join(f"- {s}" for s in sugg))

    st.markdown("---")

    # ---- Step 1: duplicates & columns
    with st.expander("1️⃣ Duplicates & unwanted columns", expanded=True):
        if st.button(f"Remove {dups:,} duplicate rows", disabled=dups == 0, key=f"dup_{ver}"):
            run_op({"type": "drop_duplicates"})
        default_drop = [c for c in dict.fromkeys(empty_cols + index_cols + const_cols)]
        drop_sel = st.multiselect("Columns to drop", df.columns.tolist(), default=default_drop,
                                  key=f"dropcols_{ver}")
        if st.button(f"Drop {len(drop_sel)} column(s)", disabled=not drop_sel, key=f"dropbtn_{ver}"):
            run_op({"type": "drop_columns", "cols": drop_sel})

    # ---- Step 2: missing values
    with st.expander("2️⃣ Missing values", expanded=False):
        mdf = pd.DataFrame({"Column": miss.index, "Missing": miss.values,
                            "Missing %": (miss.values / n_rows * 100).round(2),
                            "Type": [str(df[c].dtype) for c in miss.index]})
        mdf = mdf[mdf["Missing"] > 0].sort_values("Missing %", ascending=False)
        if mdf.empty:
            st.success("No missing values 🎉")
        else:
            a, b = st.columns([1, 1])
            with a:
                show_df(mdf, hide_index=True, height=300)
            with b:
                pc(px.bar(mdf.head(20), x="Missing %", y="Column", orientation="h",
                          title="Top 20 columns by % missing").update_yaxes(autorange="reversed"))

            st.markdown("**Option A – drop sparse columns**")
            thr = st.slider("Drop columns with more than this % missing", 10, 100, 60, key=f"thr_{ver}")
            sparse = mdf[mdf["Missing %"] > thr]["Column"].tolist()
            st.caption(f"{len(sparse)} column(s) above {thr}%: {', '.join(sparse[:10])}"
                       f"{'…' if len(sparse) > 10 else ''}")
            if st.button(f"Drop {len(sparse)} sparse column(s)", disabled=not sparse, key=f"sparse_{ver}"):
                run_op({"type": "drop_columns", "cols": sparse})

            st.markdown("**Option B – handle remaining missing values**")
            m_cols = st.multiselect("Columns to treat", mdf["Column"].tolist(),
                                    default=mdf["Column"].tolist(), key=f"mcols_{ver}")
            m_method = st.selectbox("Method", MISSING_METHODS, index=2, key=f"mmethod_{ver}")
            const_val = ""
            if m_method == "Fill with constant":
                const_val = st.text_input("Constant value", "0", key=f"mconst_{ver}")
            m_ind = st.checkbox("Add '<column>_was_missing' indicator columns (keeps the information "
                                "that a value was missing)", key=f"mind_{ver}")
            tips = {
                "Drop rows with missing values": "Simple, but can discard a lot of data (or bias it) if many rows are affected.",
                "Fill with mean": "Fine for symmetric data; sensitive to outliers; shrinks variance. Numeric only.",
                "Fill with median": "Robust to outliers; a good default for numeric data. Numeric only.",
                "Fill with mode (most frequent)": "Best for categorical columns.",
                "Fill with constant": "Use for a meaningful default (e.g. 0 or 'Unknown').",
                "Forward fill": "Carries the last value forward. Only sensible for ordered data such as time series.",
                "Backward fill": "Carries the next value backward. Only sensible for ordered data.",
                "Linear interpolation": "Estimates gaps from neighbours. Only sensible for ordered numeric data.",
            }
            st.caption(tips[m_method])
            if st.button("Apply missing-value method", disabled=not m_cols, key=f"mapply_{ver}"):
                run_op({"type": "missing", "cols": m_cols, "method": m_method,
                        "value": const_val, "indicator": m_ind})

    # ---- Step 3: outliers
    with st.expander("3️⃣ Outliers", expanded=False):
        cand = [c for c in num_cols if nun[c] > 10] or num_cols
        if not cand:
            st.info("No numeric columns available.")
        else:
            o_cols = st.multiselect("Numeric columns to check", num_cols, default=cand, key=f"ocols_{ver}")
            o_method = st.selectbox("Detection method", list(OUTLIER_METHODS), key=f"omethod_{ver}")
            d0, lo_, hi_, step_, lbl = OUTLIER_METHODS[o_method]
            o_param = st.slider(lbl, lo_, hi_, d0, step_, key=f"oparam_{o_method}_{ver}")
            if o_cols:
                table, any_n = c_outliers(df, tuple(o_cols), o_method, o_param, ver)
                show_df(table.round(3), hide_index=True)
                st.write(f"**{any_n:,} rows** ({100 * any_n / n_rows:.2f}%) contain at least one outlier.")

                prev = st.selectbox("Preview a column", o_cols, key=f"oprev_{ver}")
                row = table[table["Column"] == prev].iloc[0]
                fig = px.box(plot_df, y=prev, points="outliers", title=f"{prev} (red lines = bounds)")
                for b_ in (row["Lower bound"], row["Upper bound"]):
                    if np.isfinite(b_):
                        fig.add_hline(y=b_, line_dash="dash", line_color="red")
                pc(fig)

                o_action = st.selectbox("What to do with outliers", OUTLIER_ACTIONS, key=f"oact_{ver}")
                notes = {
                    OUTLIER_ACTIONS[0]: "Keeps the data unchanged and adds a boolean 'outlier_flag' column.",
                    OUTLIER_ACTIONS[1]: "Clips extreme values to the bounds. Keeps all rows.",
                    OUTLIER_ACTIONS[2]: "Treats outliers as missing so you can impute them in step 2.",
                    OUTLIER_ACTIONS[3]: "Replaces outliers with the column median.",
                    OUTLIER_ACTIONS[4]: "Deletes every row with an outlier in a selected column.",
                }
                st.caption(notes[o_action])
                if o_action == OUTLIER_ACTIONS[4] and any_n / n_rows > 0.05:
                    st.warning(f"This removes {100 * any_n / n_rows:.1f}% of your rows. "
                               "Consider capping instead. Outliers may be genuine.")
                if st.button("Apply outlier handling", key=f"oapply_{ver}"):
                    run_op({"type": "outliers", "cols": o_cols, "method": o_method,
                            "param": o_param, "action": o_action})

    # ---- Step 4: types & transforms
    with st.expander("4️⃣ Fix data types & transform", expanded=False):
        st.markdown("**Convert a column's type**")
        c1, c2, c3 = st.columns([2, 2, 1])
        conv_col = c1.selectbox("Column", df.columns.tolist(), key=f"cvcol_{ver}")
        c1.caption(f"Current type: {df[conv_col].dtype}")
        conv_to = c2.selectbox("Convert to", CONVERT_TYPES, key=f"cvto_{ver}")
        c3.write("")
        c3.write("")
        if c3.button("Convert", key=f"cvbtn_{ver}"):
            run_op({"type": "convert", "col": conv_col, "to": conv_to})

        st.markdown("**Transform numeric columns** (creates new columns, originals are kept)")
        t_cols = st.multiselect("Columns", num_cols, key=f"tcols_{ver}")
        t_how = st.selectbox("Transformation", TRANSFORMS, key=f"thow_{ver}")
        st.caption("Log reduces right-skew; standardizing/min-max puts columns on a comparable scale.")
        if st.button("Apply transformation", disabled=not t_cols, key=f"tbtn_{ver}"):
            run_op({"type": "transform", "cols": t_cols, "how": t_how})

# =====================================================================
# TAB 2: ANALYZE
# =====================================================================
with tab_analyze:
    st.caption("Explore the **cleaned** dataset. Charts use a sample for speed; tables use all rows.")
    (a_chart, a_over, a_uni, a_bi, a_out, a_corr, a_pair) = st.tabs(
        ["📊 Chart Builder", "Overview", "Univariate", "Bivariate", "Outliers", "Correlation", "Pair Plot"]
    )

    # ---- chart builder (controls are in the sidebar)
    with a_chart:
        st.subheader(f"{chart_type}: {x_col} vs {y_col}")

        # ---- column-type guidance for the selected X / Y
        kx, ky, g_best, g_ok, g_avoid, g_tip, g_extra = chart_guidance(x_col, y_col)
        st.markdown(f"**Column types:** X `{x_col}` → *{kx}*  ·  Y `{y_col}` → *{ky}*")
        best_txt = ", ".join(f"**{c}**" for c in g_best) or "none"
        if chart_type in g_best:
            st.success(f"✅ {chart_type} is a great choice for {kx} × {ky}.")
        elif chart_type in g_ok:
            st.info(f"ℹ️ {chart_type} works here ({g_ok[chart_type]}). "
                    f"Best choices for these columns: {best_txt}.")
        elif chart_type in g_avoid:
            st.warning(f"⚠️ {chart_type} is not suitable for {kx} × {ky}. "
                       f"Change **Chart Type** in the sidebar to: {best_txt}.")
        else:
            st.info(f"{chart_type} can be used, but better choices are: {best_txt}.")

        with st.expander("📘 Which chart should I choose for these two columns?", expanded=False):
            if g_best:
                st.markdown("**Best:**")
                for c, why in g_best.items():
                    st.markdown(f"- ✅ **{c}**: {why}")
            if g_ok:
                st.markdown("**Also fine:**")
                for c, why in g_ok.items():
                    st.markdown(f"- ➕ **{c}**: {why}")
            if g_avoid:
                st.markdown("**Avoid:** " + ", ".join(g_avoid))
            st.markdown(f"💡 {g_tip}")
            if g_extra:
                st.markdown(f"💡 {g_extra}")
            st.markdown("**Cheat sheet: chart by column types**")
            show_df(CHEAT_SHEET, hide_index=True)

        try:
            current_fig, err = build_figure(chart_type, plot_df, x_col, y_col, color_col)
            if err:
                st.warning(err)
            elif current_fig is not None:
                pc(current_fig)
        except Exception as e:
            st.error(f"Error: {e}")
        st.caption("Change chart type, axes and colour in the sidebar.")

    # ---- overview
    with a_over:
        st.markdown("#### Summary statistics")
        show_df(c_summary(df, ver))
        st.markdown("#### Data types")
        show_df(pd.DataFrame({"dtype": df.dtypes.astype(str), "unique": nun, "missing": miss}))

    # ---- univariate
    with a_uni:
        col = st.selectbox("Select a column", df.columns, key=f"uni_{ver}")
        if col in num_cols:
            s = df[col].dropna()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mean", f"{s.mean():.2f}")
            k2.metric("Median", f"{s.median():.2f}")
            k3.metric("Std", f"{s.std():.2f}")
            k4.metric("Skewness", f"{s.skew():.2f}")
            a, b = st.columns(2)
            with a:
                pc(px.histogram(plot_df, x=col, nbins=nbins, marginal="box", title=f"Histogram + Box: {col}"))
            with b:
                pc(px.violin(plot_df, y=col, box=True, points="outliers", title=f"Violin: {col}"))
            pc(px.ecdf(plot_df, x=col, title=f"ECDF: {col}"))
        else:
            counts = df[col].value_counts(dropna=False).head(30).reset_index()
            counts.columns = [col, "Count"]
            a, b = st.columns(2)
            with a:
                pc(px.bar(counts, x=col, y="Count", text_auto=True, title=f"Count plot: {col} (top 30)"))
            with b:
                if len(counts) <= 8:
                    pc(px.pie(counts, names=col, values="Count", hole=0.4, title=f"Share: {col}"))
                else:
                    st.info("Donut hidden: too many categories (> 8).")
        if num_cols:
            st.markdown("#### Skewness of all numeric columns")
            sk = c_skew(df, tuple(num_cols), ver).sort_values().reset_index()
            sk.columns = ["Column", "Skewness"]
            pc(px.bar(sk, x="Skewness", y="Column", orientation="h", text_auto=".2f"))

    # ---- bivariate
    with a_bi:
        if num_cols and cat_small:
            st.markdown("#### Numeric by category")
            n_sel = st.selectbox("Numeric column", num_cols, key=f"bi_n_{ver}")
            c_sel = st.selectbox("Category column", cat_small, key=f"bi_c_{ver}")
            a, b = st.columns(2)
            with a:
                pc(px.box(plot_df, x=c_sel, y=n_sel, color=c_sel, points="outliers", title="Box plot"))
            with b:
                pc(px.violin(plot_df, x=c_sel, y=n_sel, color=c_sel, box=True, points="outliers",
                             title="Violin plot"))
            pc(px.histogram(plot_df, x=n_sel, color=c_sel, barmode="overlay", opacity=0.6, nbins=nbins,
                            marginal="rug", title="Overlaid distributions"))
        else:
            st.info("Need a numeric column and a categorical column (≤ 30 categories) for group comparisons.")
        if len(num_cols) >= 2:
            st.markdown("#### Numeric vs numeric")
            xs = st.selectbox("X", num_cols, key=f"bi_x_{ver}")
            ys = st.selectbox("Y", num_cols, index=1, key=f"bi_y_{ver}")
            cs = st.selectbox("Color", ["None"] + cat_small, key=f"bi_col_{ver}")
            pc(px.scatter(plot_df, x=xs, y=ys, color=None if cs == "None" else cs,
                          marginal_x="box", marginal_y="violin"))
            pc(px.density_heatmap(plot_df, x=xs, y=ys, nbinsx=30, nbinsy=30,
                                  color_continuous_scale="Viridis", title="2D density"))
        if len(cat_small) >= 2:
            st.markdown("#### Category vs category")
            c1_ = st.selectbox("First category", cat_small, key=f"cc1_{ver}")
            c2_ = st.selectbox("Second category", cat_small, index=1, key=f"cc2_{ver}")
            ct = pd.crosstab(df[c1_], df[c2_])
            pc(px.imshow(ct, text_auto=True, color_continuous_scale="Blues", title="Contingency heatmap"))
            pc(px.histogram(plot_df, x=c1_, color=c2_, barmode="group", text_auto=True,
                            title="Grouped counts"))

    # ---- outliers
    with a_out:
        if num_cols:
            st.caption("Detection only. Use **1️⃣ Clean & Prepare → Outliers** to handle them.")
            cand = tuple(c for c in num_cols if nun[c] > 10) or tuple(num_cols)
            tbl, _ = c_outliers(df, cand, "IQR (Tukey fences)", 1.5, ver)
            show_df(tbl.round(3), hide_index=True)
            st.markdown("#### Box plots (standardized so scales are comparable)")
            small = df[list(cand)].sample(min(n_rows, 3000), random_state=0)
            std = (small - df[list(cand)].mean()) / df[list(cand)].std(ddof=0)
            long = std.melt(var_name="Feature", value_name="Z-score").dropna()
            pc(px.box(long, x="Feature", y="Z-score", points="outliers"))
        else:
            st.info("No numeric columns found.")

    # ---- correlation
    with a_corr:
        if len(num_cols) > 1:
            try:
                import scipy  # noqa: F401
                methods = ["pearson", "spearman", "kendall"]
            except ImportError:
                methods = ["pearson", "spearman"]
            method = st.radio("Correlation method", methods, horizontal=True, key=f"cm_{ver}")
            sel = st.multiselect("Columns (heatmap gets crowded beyond ~20)", num_cols,
                                 default=num_cols[:20], key=f"cs_{ver}")
            if len(sel) >= 2:
                corr = (plot_df[sel].corr(method=method) if method == "kendall"
                        else c_corr(df, tuple(sel), method, ver))
                pc(px.imshow(corr, text_auto=".2f" if len(sel) <= 15 else False, aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1))
            else:
                st.info("Select at least two columns.")
        else:
            st.info("Not enough numeric features for a heatmap.")

    # ---- pair plot
    with a_pair:
        if len(num_cols) >= 2:
            dims = st.multiselect("Columns (max 6 recommended)", num_cols, default=num_cols[:4],
                                  key=f"pd_{ver}")
            hue_opts = ["None"] + [c for c in df.columns if nun[c] <= 10 and not is_helper_col(c)]
            hue = st.selectbox("Color by", hue_opts,
                               index=hue_opts.index(target_col) if target_col in hue_opts else 0,
                               key=f"ph_{ver}")
            if len(dims) >= 2:
                smp = df.sample(min(n_rows, 2000), random_state=0)
                if n_rows > 2000:
                    st.caption("Showing a random sample of 2,000 rows for speed.")
                fig = px.scatter_matrix(smp, dimensions=dims, color=None if hue == "None" else hue)
                fig.update_traces(diagonal_visible=False, marker=dict(size=4))
                fig.update_layout(height=700)
                pc(fig)
            else:
                st.info("Pick at least two columns.")
        else:
            st.info("Need at least two numeric columns for a pair plot.")

# =====================================================================
# TAB 3: INTERPRET
# =====================================================================
findings = build_findings()

with tab_interp:
    i_find, i_feat, i_chart = st.tabs(["🧾 Key Findings", "🧠 Feature Importance", "🔍 Chart Check"])

    with i_find:
        st.caption("Automatically generated from the cleaned data. Always sanity-check against domain knowledge.")
        for f in findings:
            st.markdown(f"{f['icon']} **{f['topic']}:** {f['text']}")

        st.markdown("#### Impact of cleaning")
        r = ss.raw_df
        cmp = pd.DataFrame({
            "": ["Rows", "Columns", "Missing cells"],
            "Original": [len(r), r.shape[1], int(c_missing(r, f'raw-{ss.src_id}').sum())],
            "Cleaned": [n_rows, df.shape[1], int(miss.sum())],
        })
        show_df(cmp, hide_index=True)

    with i_feat:
        st.write(f"Target Variable: **{target_col}**  |  Selected Feature: **{feature_col}**")
        imp = c_importance(df, target_col, tuple(num_cols), ver)
        if len(imp):
            for feat, val in imp.items():
                if target_col in num_cols:
                    st.write(f"• **{feat}** is important because it has a strong correlation with "
                             f"{target_col} (|r| = {val:.2f})")
                else:
                    st.write(f"• **{feat}** is important because its mean varies across categories "
                             f"of {target_col} (range: {val:.2f})")
        else:
            st.write("No strong features detected.")

    with i_chart:
        st.markdown(f"**Recommended chart:** {recommended_chart}  |  **Currently selected:** {chart_type}")
        expl = {
            "Bar": "Bar charts compare categories using numeric values or counts.",
            "Scatter": "Scatter plots show relationships and clusters between two numeric variables.",
            "Box": "Box plots compare median, spread and outliers of a numeric variable across groups.",
            "Violin": "Violin plots show the full distribution shape per group.",
            "Histogram": "Histograms show the distribution of one numeric variable.",
            "Pie": "Pie charts show composition but should be used carefully.",
        }
        st.write(expl.get(recommended_chart, ""))

        score, issues = 0, []
        if chart_type == "Bar" and y_min_pct > 0:
            score += 30
            issues.append(("Y-axis does not start from zero", "A truncated Y-axis exaggerates differences between bars."))
        if chart_type == "Pie":
            score += 20
            issues.append(("Pie chart hides small differences", "Angles and areas are hard to compare; a bar chart is clearer."))
            if y_col in num_cols and (df[y_col] < 0).any():
                score += 20
                issues.append(("Pie chart with negative values", "Negative values cannot be shown as slices of a whole."))
        if chart_type == "Line" and x_col in cat_cols:
            score += 10
            issues.append(("Line chart used for categorical data", "Lines imply an order or trend that may not exist."))
        if chart_type in ("Box", "Violin") and n_rows < 30:
            score += 15
            issues.append((f"{chart_type} plot on a small sample (n={n_rows})", "Shapes are unreliable below ~30 points."))
        if chart_type == "Violin" and n_rows < 100:
            score += 5
            issues.append(("Violin smoothing may invent structure", "Kernel smoothing can show bumps not in the data."))
        if chart_type == "Histogram" and (nbins < 8 or nbins > 60):
            score += 10
            issues.append((f"Histogram bin count ({nbins}) may distort the shape", "Too few bins hide structure; too many show noise."))
        if chart_type == "Scatter" and (x_col not in num_cols or y_col not in num_cols):
            score += 15
            issues.append(("Scatter plot with a non-numeric axis", "Scatter plots are for two numeric variables."))
        if chart_type == "Scatter" and n_rows > 5000:
            score += 10
            issues.append(("Scatter plot with many points (overplotting)", "Overlapping points hide density; try a density heatmap."))
        if not show_labels:
            score += 15
            issues.append(("Missing labels", "Missing labels reduce clarity."))
        if n_rows > max_rows:
            issues.append((f"Chart is based on a {max_rows:,}-row sample", "Random samples can miss rare values; check against full-data statistics."))

        score = min(score, 100)
        st.write(f"Misleading-chart score: **{score}/100**")
        (st.success if score <= 20 else st.warning if score <= 50 else st.error)(
            "Low risk" if score <= 20 else "Moderate risk" if score <= 50 else "High risk")
        for msg, why in issues:
            st.write(f"• **{msg}** – {why}")
        if not issues:
            st.write("No issues detected for the current chart settings.")

# =====================================================================
# TAB 4: COMMUNICATE
# =====================================================================
with tab_comm:
    st.markdown("### Build your report")
    cc1, cc2 = st.columns(2)
    title = cc1.text_input("Report title", f"EDA Report – {'uploaded dataset' if uploaded_file else 'sample dataset'}")
    author = cc2.text_input("Author", "")
    audience = st.selectbox("Audience", ["Technical (analysts / data scientists)", "Business stakeholders",
                                         "General audience"])
    notes = st.text_area(
        "Your interpretation & recommendations",
        placeholder="What do the findings mean? What decisions or next steps do they support? "
                    "What are the limitations (sample size, missing data, bias)?",
        height=150,
    )

    fig_choices = ["Correlation heatmap", "Missing values by column", "Distribution of a column"]
    if current_fig is not None:
        fig_choices.insert(0, "Current Chart Builder chart")
    picked = st.multiselect("Charts to include", fig_choices,
                            default=[c for c in fig_choices if c.startswith("Current")] or fig_choices[:1])
    dist_col = None
    if "Distribution of a column" in picked:
        dist_col = st.selectbox("Column for distribution chart", df.columns.tolist(), key=f"rep_col_{ver}")

    figs = []
    for name in picked:
        if name == "Current Chart Builder chart":
            figs.append((f"{chart_type}: {x_col} vs {y_col}", current_fig))
        elif name == "Correlation heatmap" and len(num_cols) > 1:
            cols_ = tuple(num_cols[:20])
            figs.append(("Correlation heatmap",
                         px.imshow(c_corr(df, cols_, "pearson", ver), text_auto=".2f" if len(cols_) <= 15 else False,
                                   color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")))
        elif name == "Missing values by column":
            mm = (miss[miss > 0] / n_rows * 100).sort_values(ascending=False).head(20).reset_index()
            mm.columns = ["Column", "Missing %"]
            if len(mm):
                figs.append(("Missing values (% per column)", px.bar(mm, x="Missing %", y="Column", orientation="h")
                             .update_yaxes(autorange="reversed")))
        elif name == "Distribution of a column" and dist_col:
            figs.append((f"Distribution of {dist_col}",
                         px.histogram(plot_df, x=dist_col, marginal="box") if dist_col in num_cols
                         else px.histogram(plot_df, x=dist_col)))

    st.markdown("---")
    st.markdown(f"## {title}")
    st.caption(f"{author + ' · ' if author else ''}{datetime.now():%d %b %Y} · Audience: {audience}")
    st.markdown("#### Key findings")
    for f in findings:
        st.markdown(f"{f['icon']} **{f['topic']}:** {f['text']}")
    if notes.strip():
        st.markdown("#### Interpretation & recommendations")
        st.write(notes)
    for name, fig in figs:
        st.markdown(f"#### {name}")
        pc(fig)
    st.markdown("#### Data preparation log")
    if ss.log:
        show_df(pd.DataFrame(ss.log), hide_index=True)
    else:
        st.caption("No cleaning steps were applied.")

    # ---- exports
    st.markdown("---")
    st.markdown("### 📥 Export")
    log_df = pd.DataFrame(ss.log)

    def make_markdown():
        lines = [f"# {title}", f"*{author + ' · ' if author else ''}{datetime.now():%d %b %Y}*", "",
                 f"Dataset: {n_rows:,} rows × {df.shape[1]} columns", "", "## Key findings"]
        lines += [f"- {f['icon']} **{f['topic']}:** {f['text']}" for f in findings]
        if notes.strip():
            lines += ["", "## Interpretation & recommendations", notes]
        lines += ["", "## Data preparation log"]
        lines += ([f"{r['Step']}. {r['Action']}  (rows: {r['Rows']:,}, missing: {r['Missing cells']:,})"
                   for r in ss.log] or ["No cleaning steps were applied."])
        return "\n".join(lines)

    def make_html():
        e = html.escape
        parts = [f"<html><head><meta charset='utf-8'><title>{e(title)}</title><style>"
                 "body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}"
                 "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
                 "h1{margin-bottom:0}.sub{color:#777}</style></head><body>",
                 f"<h1>{e(title)}</h1><p class='sub'>{e(author)} {datetime.now():%d %b %Y} · {e(audience)}</p>",
                 f"<p>Dataset: <b>{n_rows:,}</b> rows × <b>{df.shape[1]}</b> columns</p>",
                 "<h2>Key findings</h2><ul>"]
        parts += [f"<li>{f['icon']} <b>{e(f['topic'])}:</b> {e(f['text'])}</li>" for f in findings]
        parts.append("</ul>")
        if notes.strip():
            parts.append(f"<h2>Interpretation &amp; recommendations</h2><p>{e(notes).replace(chr(10), '<br>')}</p>")
        for i, (name, fig) in enumerate(figs):
            parts.append(f"<h2>{e(name)}</h2>" + fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
        parts.append("<h2>Data preparation log</h2>")
        parts.append(log_df.to_html(index=False) if len(log_df) else "<p>No cleaning steps were applied.</p>")
        parts.append("</body></html>")
        return "\n".join(parts)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Report (HTML, with charts)", make_html(), file_name="eda_report.html", mime="text/html")
    d2.download_button("⬇️ Report (Markdown)", make_markdown(), file_name="eda_report.md", mime="text/markdown")
    d3.download_button("⬇️ Cleaning log (CSV)", log_df.to_csv(index=False) if len(log_df) else "",
                       file_name="cleaning_log.csv", mime="text/csv", disabled=log_df.empty)
    if st.checkbox("Prepare cleaned dataset for download (can take a few seconds for large files)",
                   key=f"prep_csv_{ver}"):
        st.download_button("⬇️ Cleaned dataset (CSV)", df.to_csv(index=False), file_name="cleaned_data.csv",
                           mime="text/csv")
    st.caption("The HTML report loads Plotly from the internet; open it while online to see the charts.")