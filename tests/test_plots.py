import matplotlib
import numpy as np
import pandas as pd
import pytest

from sbench import plots

matplotlib.use("Agg")   # headless


def test_charts_render_and_save(tmp_path, gbm):
    plots.use_style()
    plots.series_panels([gbm], shade={"gbm": np.arange(len(gbm)) % 50 < 10}, path=tmp_path / "a.png")
    df = pd.DataFrame({"model": ["a", "a", "b"], "fmt": ["x", "y", "x"], "v": [0.1, -0.2, 0.3]})
    plots.dot_plot(df.assign(lo=df["v"] - 0.1, hi=df["v"] + 0.1), "model", "v", hue="fmt", lo="lo", hi="hi",
                   path=tmp_path / "b.png")
    plots.heatmap(df.pivot(index="model", columns="fmt", values="v"), path=tmp_path / "c.png")
    rel = pd.DataFrame({"predictor": "a", "forecast": [0.2, 0.8], "observed": [0.3, 0.7], "n": [5, 9]})
    plots.reliability_panels(rel, path=tmp_path / "d.png")
    plots.scatter_panels(pd.DataFrame({"p": "a", "x": [0.1, 0.5, 0.9], "y": [0.2, 0.5, 0.8]}), "p", "x", "y",
                         path=tmp_path / "e.png")
    assert len(list(tmp_path.glob("*.png"))) == 5


def test_dot_plot_refuses_more_than_three_hues():
    df = pd.DataFrame({"m": "a", "h": list("wxyz"), "v": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="at most 3"):
        plots.dot_plot(df, "m", "v", hue="h")
