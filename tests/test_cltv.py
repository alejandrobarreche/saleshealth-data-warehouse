"""Unit tests for the analytics layer that do not need a database."""
import numpy as np
import pandas as pd

from src.analytics import compute_cltv, run_pca_kmeans


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "net_revenue": [1000.0, 250.0, 40.0],
            "margin_rate": [0.4, 0.5, 0.1],
            "frequency": [10, 2, 1],
            "retention_rate": [0.8, 0.25, 0.05],
        }
    )


def test_cltv_formula_consistency():
    df = compute_cltv(_customers())
    expected = df["net_revenue"] * df["margin_rate"] * df["frequency"] * df["retention_rate"]
    assert np.allclose(df["cltv"], expected)


def test_cltv_sorted_descending():
    df = compute_cltv(_customers())
    assert df["cltv"].is_monotonic_decreasing
    assert df.loc[0, "customer_id"] == 1


def test_pca_kmeans_shapes_and_labels():
    rng = np.random.default_rng(42)
    features = [f"f{i}" for i in range(8)]
    df = pd.DataFrame(rng.normal(size=(60, 8)), columns=features)
    df["customer_id"] = range(60)

    result = run_pca_kmeans(df, features, n_clusters=3, n_components=2)

    assert {"pc1", "pc2", "cluster"}.issubset(result.df.columns)
    assert len(result.df) == 60
    assert set(result.df["cluster"].unique()) == {0, 1, 2}
    assert result.explained_variance.shape == (2,)
    assert 0 < result.explained_variance.sum() <= 1.0
