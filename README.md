# AutoVisCheck

**An intelligent EDA and visualization assistant that recommends charts and flags misleading ones.**

AutoVisCheck is a Streamlit app that walks you through a four-step exploratory data analysis workflow:

1. **Clean & Prepare**: remove duplicates, drop sparse/constant columns, impute missing values, detect and treat outliers (IQR, Z-score, MAD, percentile), fix types and transform columns. Every step is logged and can be undone.
2. **Analyze**: chart builder with column-type-aware guidance, plus univariate, bivariate, outlier, correlation and pair-plot views.
3. **Interpret**: auto-generated key findings, simple feature-relevance ranking, and a *misleading-chart score* for the current chart settings.
4. **Communicate**: build a report and export it as HTML (with interactive charts), Markdown, a cleaning log CSV, or the cleaned dataset.

<!-- Add a screenshot: docs/screenshot.png -->

## Quick start

```bash
git clone https://github.com/<your-username>/autovischeck.git
cd autovischeck
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Without an upload, the app uses a built-in synthetic dataset (150 rows, with injected missing values and outliers) so every feature can be tried immediately.

## Project structure

```
autovischeck/
├── app.py                  # Streamlit UI (entry point)
├── autovischeck/
│   ├── __init__.py
│   └── cleaning.py         # Pure pandas/numpy cleaning logic (no Streamlit)
├── tests/
│   └── test_cleaning.py    # Unit tests for the cleaning ops
├── docs/                   # Screenshots, figures
├── .streamlit/config.toml  # Streamlit settings
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

### Design note: replayable cleaning

Each cleaning step is a plain dict (an "op"), e.g. `{"type": "missing", "cols": ["Age"], "method": "Fill with median"}`.
`apply_op(df, op)` returns a new DataFrame and a log message. The app stores the raw data plus the op list, so **undo** is implemented by replaying all remaining ops on the raw data. This keeps the cleaning logic testable in isolation from the UI.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs the tests on Python 3.10 and 3.12 for every push and pull request.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, choose **New app**, select the repo, branch `main`, and main file `app.py`.
3. Add the live URL to the top of this README.

## Limitations

- Feature relevance is a heuristic (absolute Pearson r, or the range of group means), not a model-based importance measure, and the group-mean range is not scale-normalized.
- Plots use a random sample above the configurable row limit; statistics use all rows.
- The exported HTML report loads Plotly from a CDN, so it needs an internet connection to display charts.

## License

MIT, see [LICENSE](LICENSE).

## Author

Faiza Binte Zaman
