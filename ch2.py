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
    # Quick Look
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
    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3))
    _ax[0].hist(housing_full["median_income"])
    _ax[1].hist(strat_train_set["median_income"])
    _ax[2].hist(strat_test_set["median_income"])

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell
def _(strat_train_set):
    df = strat_train_set.copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Geographical Data
    """)
    return


@app.cell
def _(df, plt):
    _fig, _ax = plt.subplots(figsize=(10,6))
    sc = _ax.scatter(
        x=df['longitude'], y=df['latitude'],
        s=df['population']/100,
        c=df['median_house_value'],
        cmap='jet'
    )
    _ax.set_xlabel('longitude')
    _ax.set_ylabel('latitude')
    _ax.grid(True)
    _fig.colorbar(sc)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations
    """)
    return


@app.cell
def _(df):
    df.corr(numeric_only=True)['median_income'].sort_values()
    return


@app.cell
def _(df, plt):
    most_relevant = ['housing_median_age', 'median_house_value', 'total_rooms']

    _fig, _ax = plt.subplots(1, 3, figsize=(12, 3))
    for i in range(3):
        _ax[i].scatter(x=df['median_income'], y=df[most_relevant[i]], s=2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attribute Combination
    """)
    return


@app.cell
def _(df):
    (df['total_rooms']/df['households']).describe()
    return


@app.cell
def _(df):
    (df['total_rooms']/df['households']).hist(bins=range(15))
    return


@app.cell
def _(df):
    df['rooms_per_house'] = df['total_rooms']/df['households']
    df['bedrooms_per_room'] = df['total_bedrooms']/df['total_rooms']
    df['people_per_house'] = df['population']/df['households']
    return


@app.cell
def _(df):
    df.corr(numeric_only=True)['median_income'].sort_values()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare for ML model
    """)
    return


@app.cell
def _(strat_train_set):
    X = strat_train_set.drop('median_income', axis=1)
    y = strat_train_set['median_income'].copy()
    return (X,)


@app.cell
def _(X, np, pd):
    from sklearn.impute import SimpleImputer

    mean_imputer = SimpleImputer(strategy="mean")
    X_num = X.select_dtypes(include=np.number)
    mean_imputer.fit(X_num)
    X_imputed = mean_imputer.transform(X_num)
    X_imputed = pd.DataFrame(X_imputed, index=X_num.index, columns=X_num.columns)
    X_imputed.info()
    return (mean_imputer,)


@app.cell
def _(mean_imputer):
    print(mean_imputer.feature_names_in_)
    print(mean_imputer.get_feature_names_out())
    return


@app.cell
def _(X):
    from sklearn.preprocessing import OrdinalEncoder

    X_cat = X.select_dtypes(include=str)
    ordinal_encoder = OrdinalEncoder()
    X_ord_encoded = ordinal_encoder.fit_transform(X_cat)
    X_ord_encoded
    return X_cat, ordinal_encoder


@app.cell
def _(ordinal_encoder):
    ordinal_encoder.get_feature_names_out()
    return


@app.cell
def _(X_cat):
    from sklearn.preprocessing import OneHotEncoder

    onehot_encoder = OneHotEncoder()
    X_onh_encoded = onehot_encoder.fit_transform(X_cat)
    X_onh_encoded.toarray()
    return


@app.cell
def _(X, np, plt):
    _fig, _ax = plt.subplots(2,2)

    _ax[0][0].hist(X['population'], bins=100)
    _ax[0][0].grid(True)
    _ax[0][0].set_xlabel("original")

    _ax[0][1].hist(np.sqrt(X['population']), bins=100)
    _ax[0][1].grid(True)
    _ax[0][1].set_xlabel("sqrt")

    _ax[1][0].hist(np.log(X['population']), bins=100)
    _ax[1][0].grid(True)
    _ax[1][0].set_xlabel("log")

    _ax[1][1].hist(np.pow(X['population'], 0.1), bins=100)
    _ax[1][1].grid(True)
    _ax[1][1].set_xlabel("x^0.1")

    _fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
