from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------- paths ----------
FIG_DIR = Path("figures")
TRAIN_FIG = FIG_DIR / "train_actual_vs_pred.png"
TEST_FIG = FIG_DIR / "test_actual_vs_pred.png"


# ---------- helper functions ----------
def make_fig_dir():
    FIG_DIR.mkdir(exist_ok=True)


def plot_actual_vs_pred(y_true, y_pred, title, save_path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("R2  :", r2_score(y_true, y_pred))


# ---------- main ----------
def main():
    make_fig_dir()

    # 1. Load dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Model with early stopping
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.001,
            alpha=0.0001,
            early_stopping=True,
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # 4. Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 5. Metrics
    print_metrics("TRAIN SET", y_train, y_train_pred)
    print_metrics("TEST SET", y_test, y_test_pred)

    # 6. Plots
    plot_actual_vs_pred(
        y_train, y_train_pred,
        "Train: Actual vs Predicted",
        TRAIN_FIG
    )

    plot_actual_vs_pred(
        y_test, y_test_pred,
        "Test: Actual vs Predicted",
        TEST_FIG
    )

    print("\nPlots saved in figures/ folder")


if __name__ == "__main__":
    main()
