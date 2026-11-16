import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    from pathlib import Path
    import pandas as pd
    import tarfile
    import urllib.request
    import matplotlib.pyplot as plt

    return Path, np, pd, plt, tarfile, urllib


@app.cell
def _():
    SEED = 0
    return (SEED,)


@app.cell
def _(Path, pd, tarfile, urllib):
    def load_housing_data():
        tarfile_path = Path("datasets/housing.tgz")
        if not tarfile_path.is_file():
            print("Downloading .tgz file")
            Path("datasets").mkdir(exist_ok=True, parents=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarfile_path)
        csv_path = Path("datasets/housing/housing.csv")
        if not csv_path.is_file():
            print("Downloading .csv file")
            with tarfile.open(tarfile_path) as housing_tarfile:
                housing_tarfile.extractall(path="datasets")
        return pd.read_csv(csv_path)
    housing_full = load_housing_data()
    return (housing_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    # EDA
    """)
    return


@app.cell
def _(housing_full):
    housing_full.head()
    return


@app.cell
def _(housing_full):
    housing_full.info()
    return


@app.cell
def _(housing_full):
    housing_full["ocean_proximity"].value_counts()
    return


@app.cell
def _(housing_full):
    housing_full.describe()
    return


@app.cell
def _(housing_full, plt):
    housing_full.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Split
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split

    return (train_test_split,)


@app.cell
def _(housing_full, plt):
    plt.hist(housing_full["median_income"])
    plt.show()
    return


@app.cell
def _(housing_full, np, pd, plt):
    housing_full["income_cat"] = pd.cut(housing_full["median_income"], bins=[0, 2.2, 3, 3.75, 4.8, 6.5, np.inf], labels=[1, 2, 3, 4, 5, 6])

    plt.hist(housing_full["income_cat"])
    plt.show()
    return


@app.cell
def _(SEED, housing_full, train_test_split):
    strat_train_set, strat_test_set = train_test_split(housing_full, test_size=0.2, random_state=SEED, stratify=housing_full["income_cat"])
    return strat_test_set, strat_train_set


@app.cell
def _(housing_full, plt, strat_test_set, strat_train_set):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].hist(housing_full["median_income"])
    ax[1].hist(strat_train_set["median_income"])
    ax[2].hist(strat_test_set["median_income"])

    plt.show()
    return


@app.cell
def _(strat_test_set, strat_train_set):
    try:
        for set_ in (strat_train_set, strat_test_set):
            set_ = set_.drop("income_cat", axis=1)
    except KeyError:
        print("Already dropped")
    except Exception as e:
        raise e
    return


if __name__ == "__main__":
    app.run()
