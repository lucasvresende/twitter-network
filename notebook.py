import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    import polars as pl

    from pathlib import Path

    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(Path, pl):
    ROOT_PATH = Path(__file__).resolve().parent
    DATA_PATH = ROOT_PATH / "data"

    df = pl.read_csv(DATA_PATH / "", separator=" ")
    return


if __name__ == "__main__":
    app.run()
