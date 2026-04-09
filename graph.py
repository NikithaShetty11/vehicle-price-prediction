import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_year_price_graph(df, year_col, price_col, title):
    data = df.copy()
    data.columns = data.columns.str.strip()

    if year_col not in data.columns or price_col not in data.columns:
        raise ValueError(f"Columns '{year_col}' and/or '{price_col}' not found in dataset.")

    data = data[[year_col, price_col]].dropna()

    X = data[[year_col]]
    y = data[price_col]

    lr = LinearRegression()
    lr.fit(X, y)

    x_min = X[year_col].min()
    x_max = X[year_col].max()

    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = lr.predict(x_line)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(data[year_col], data[price_col], label="Actual Data")
    ax.plot(x_line, y_line, color="red", label="Linear Regression Line")

    ax.set_xlabel(year_col)
    ax.set_ylabel(price_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig