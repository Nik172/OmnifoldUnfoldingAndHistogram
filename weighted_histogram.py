import numpy as np
import matplotlib.pyplot as plt
import warnings
import pytest


def weighted_histogram(
    values,
    weights,
    bins=50,
    hist_range=None,
    observable_name="",
    uncertainty=None,
    plot=True,
    ax=None,
):
    """
    Parameters->
    values: array-like - Observable values, one per event (e.g. df['pT_ll']).
    weights: array-like - Per-event weights, one per event (e.g. df['weights_nominal']). Must be the same length as values.
    bins: int or array-like, optional - Number of bins or explicit bin edges. Default set to 50.
    hist_range: tuple (float, float), optional - (min, max) range for the histogram. If None, uses (min(values), max(values)) after NaN removal.
    observable_name: str, optional - Label for the x-axis and plot title.
    uncertainty: array-like of shape (n_events, n_ensemble), optional - Ensemble weight columns (e.g. weights_ensemble_0..99).
        If provided, computes per-bin std across ensemble members and plots uncertainty bands. Must have the same number of rows as values.
    plot: bool, optional - Whether to produce a matplotlib plot. Default is True.
    ax: matplotlib.axes.Axes, optional - Existing axis to plot onto. If None and plot=True, a new figure is created.

    Returns->

    dict with keys:
        bin_centers   : ndarray, shape (n_bins,)
        bin_edges     : ndarray, shape (n_bins + 1,)
        counts        : ndarray, shape (n_bins,)  - summed weights per bin
        uncertainty   : ndarray or None           - per-bin std from ensemble
        fig           : matplotlib Figure or None
        ax            : matplotlib Axes or None

    Raises->
    ValueError
        If values and weights have different lengths, are empty, or if
        uncertainty has an incompatible shape.
    """

    # 1. Input validation and conversion
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            f"values must be 1-dimensional, got shape {values.shape}."
        )
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be 1-dimensional, got shape {weights.shape}."
        )
    if len(values) == 0 or len(weights) == 0:
        raise ValueError("values and weights must not be empty.")
    if len(values) != len(weights):
        raise ValueError(
            f"values and weights must have the same length. "
            f"Got {len(values)} and {len(weights)}."
        )

    # 2. Handle NaN values - drop events where either value or weight is NaN
    # Identify events where either the value or weight is missing
    missing_data = np.isnan(values) | np.isnan(weights)
    missing_count = int(missing_data.sum())

    if missing_count > 0:
        warnings.warn(
            f"{missing_count} event(s) were dropped due to missing values or weights.",
            UserWarning,
            stacklevel=2,
        )
        # Keep only the events with complete data
        valid = ~missing_data
        values = values[valid]
        weights = weights[valid]

    # 3. Handle negative weights - allowed (valid in theory reweightings) but warn the user.
    negative_wt_count = int((weights < 0).sum())
    if negative_wt_count > 0:
        warnings.warn(
            f"{negative_wt_count} events have negative weights. "
            "This is valid for some theory reweightings but may indicate "
            "an issue with the input.",
            UserWarning,
            stacklevel=2,
        )

    # 4. Handle all-zero weights
    if np.all(weights == 0):
        warnings.warn(
            "All weights are zero. Returning a zero histogram.",
            UserWarning,
            stacklevel=2,
        )

    # 5. Determine histogram range and check for events within range
    if hist_range is None:
        hist_range = (float(values.min()), float(values.max()))

    # Check which events fall within the specified range
    within_range = (values >= hist_range[0]) & (values <= hist_range[1])
    any_in_range = within_range.any()
    
    if not any_in_range:
        warnings.warn(
            f"No events fall within the specified range {hist_range}. "
            "Returning a zero histogram.",
            UserWarning,
            stacklevel=2,
        )

    # 6. Compute the nominal weighted histogram
    # Binning the values into a weighted histogram
    counts, bin_edges = np.histogram(
        values, bins=bins, range=hist_range, weights=weights
    )
    
    # Find the center of each bin by averaging its left and right edges
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_centers = 0.5 * (left_edges + right_edges)

    # 7. Compute uncertainty from ensemble weights (if provided)
    uncertainty_per_bin = None
    ensemble_size = 0

    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)

        if uncertainty.ndim != 2:
            raise ValueError(
                f"uncertainty must be 2-dimensional (n_events, n_ensemble), "
                f"got shape {uncertainty.shape}."
            )
        
        total_events = len(missing_data)
        if uncertainty.shape[0] != total_events:
            raise ValueError(
                f"uncertainty must have the same number of rows as values. "
                f"Got {uncertainty.shape[0]} rows and {total_events} events."
            )
        
        # Drop rows corresponding to missing events
        valid = np.logical_not(missing_data)
        uncertainty = uncertainty[valid]

        ensemble_size = uncertainty.shape[1]
        ensemble_histograms = np.zeros((ensemble_size, len(bin_centers)))

        # Histogram the values once per ensemble member using its weights
        for i in range(ensemble_size):
            weights_for_this_member = uncertainty[:, i]
            ensemble_histograms[i], _ = np.histogram(
                values, bins=bin_edges, weights=weights_for_this_member
            )

        # The uncertainty on each bin is how much it varies across ensemble members
        uncertainty_per_bin = ensemble_histograms.std(axis=0)

    # 8. Plotting
    figure = None
    axes = None

    if plot:
        if ax is None:
            figure, axes = plt.subplots(figsize=(8, 5))
        else:
            figure = ax.get_figure()
            axes = ax

        axes.stairs(
            counts,
            bin_edges,
            color="#1F4E79",
            linewidth=1.8,
            label="Unfolded (nominal)",
        )

        if uncertainty_per_bin is not None:
            upper_edge = counts + uncertainty_per_bin
            lower_edge = np.maximum(counts - uncertainty_per_bin, 0)

            axes.stairs(
                upper_edge,
                bin_edges,
                baseline=lower_edge,
                fill=True,
                color="#1F4E79",
                alpha=0.25,
                label=f"Stat. uncertainty ({ensemble_size} ensemble members)",
            )

        x_label = observable_name if observable_name else "Observable"
        axes.set_xlabel(x_label, fontsize=12)
        axes.set_ylabel("Weighted event count", fontsize=12)

        title = f"Unfolded distribution: {observable_name}" if observable_name else "Unfolded distribution"
        axes.set_title(title, fontsize=13)

        axes.legend(fontsize=10)
        axes.set_xlim(hist_range)

        # Don't let the y-axis dip below zero if all weights are positive
        y_bottom = axes.get_ylim()[0]
        all_weights_positive = np.all(weights >= 0)
        if y_bottom < 0 and all_weights_positive:
            axes.set_ylim(bottom=0)

        plt.tight_layout()

    return {
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "counts": counts,
        "uncertainty": uncertainty_per_bin,
        "fig": figure,
        "ax": axes,
    }


# TESTS
# Edge case reasoning:
# 1. basic correctness - the most fundamental check: a weighted histogram must sum weights per bin, not count events.
# 2. mismatched lengths - must raise a clear error rather than silently producing a wrong result.
# 3. empty arrays - a degenerate input that must not crash numpy internals.
# 4. NaN handling - NaNs propagate silently through numpy histograms,
#    producing wrong results without any error. Must be caught explicitly.
# 5. all-zero weights - produces a valid but empty histogram; should warn
#    rather than silently mislead the user into thinking there is no data.
# 6. negative weights - Must be allowed but warned about.
# 7. uncertainty shape mismatch - ensemble weights are a 2D array and a wrong shape must raise a clear error.
# 8. single event - degenerate but valid input and must not crash.
# 9. range with no data - could be a mistake when copy-pasting ranges from another analysis; should warn and return zeros, not crash.
# 10. plot=False - must return None for fig and ax without crashing.
# 11. return value completeness - all keys must always be present.
# 12. uncertainty correctness - identical ensemble members must give zero uncertainty, verifying the std computation is correct.
# 13. bin centers - must be midpoints of bin edges, not edges themselves.


def test_basic_correctness():
    """Weights are summed per bin, not counted."""
    values = np.array([1.0, 1.5, 3.0])
    weights = np.array([2.0, 3.0, 4.0])
    result = weighted_histogram(
        values, weights, bins=2, hist_range=(0, 4), plot=False
    )
    # bin 1: [0,2) -> events at 1.0 (w=2) and 1.5 (w=3) -> sum=5
    # bin 2: [2,4] -> event at 3.0 (w=4) -> sum=4
    assert result["counts"][0] == pytest.approx(5.0), "Bin 1 weight sum incorrect"
    assert result["counts"][1] == pytest.approx(4.0), "Bin 2 weight sum incorrect"


def test_mismatched_lengths():
    """Different length values and weights must raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        weighted_histogram(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            plot=False,
        )


def test_empty_arrays():
    """Empty input arrays must raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        weighted_histogram(np.array([]), np.array([]), plot=False)


def test_nan_handling(recwarn):
    """NaN events are dropped with a warning; result is still valid."""
    values = np.array([1.0, np.nan, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    result = weighted_histogram(
        values, weights, bins=5, hist_range=(0, 5), plot=False
    )
    assert result["counts"].sum() == pytest.approx(2.0)
    assert any("missing" in str(w.message) for w in recwarn.list)


def test_all_zero_weights():
    """All-zero weights produce a zero histogram and emit a warning."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.zeros(3)
    with pytest.warns(UserWarning, match="All weights are zero"):
        result = weighted_histogram(values, weights, bins=5, plot=False)
    assert np.all(result["counts"] == 0)


def test_negative_weights_allowed():
    """Negative weights are allowed (valid for NLO theory) but warned about."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, -0.5, 1.0])
    with pytest.warns(UserWarning, match="negative weights"):
        result = weighted_histogram(
            values, weights, bins=3, hist_range=(0, 4), plot=False
        )
    # total = 1.0 + (-0.5) + 1.0 = 1.5
    assert result["counts"].sum() == pytest.approx(1.5)


def test_uncertainty_shape_mismatch():
    """Uncertainty array with wrong number of rows must raise ValueError."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    bad_uncertainty = np.ones((5, 10))  # 5 rows but 3 events
    with pytest.raises(ValueError, match="same number of rows"):
        weighted_histogram(
            values, weights, uncertainty=bad_uncertainty, plot=False
        )


def test_uncertainty_wrong_dimensions():
    """1D uncertainty array must raise ValueError (must be 2D)."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="2-dimensional"):
        weighted_histogram(
            values, weights, uncertainty=np.ones(3), plot=False
        )


def test_single_event():
    """A single event must not crash."""
    result = weighted_histogram(
        np.array([5.0]), np.array([1.5]),
        bins=10, hist_range=(0, 10), plot=False,
    )
    assert result["counts"].sum() == pytest.approx(1.5)


def test_range_no_data():
    """Range that excludes all data produces zero histogram with a warning."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.warns(UserWarning, match="No events fall within"):
        result = weighted_histogram(
            values, weights, bins=5, hist_range=(100, 200), plot=False
        )
    assert np.all(result["counts"] == 0)


def test_plot_false_returns_none():
    """plot=False must return None for fig and ax."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    result = weighted_histogram(values, weights, plot=False)
    assert result["fig"] is None
    assert result["ax"] is None


def test_return_value_completeness():
    """Return dict must always contain all expected keys."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    result = weighted_histogram(values, weights, plot=False)
    expected_keys = {"bin_centers", "bin_edges", "counts", "uncertainty", "fig", "ax"}
    assert expected_keys == set(result.keys())


def test_uncertainty_computed_correctly():
    """Ensemble std is computed per bin across ensemble members."""
    rng = np.random.default_rng(42)
    n_events = 1000
    values = rng.uniform(0, 10, n_events)
    weights = np.ones(n_events)
    # Identical ensemble members -> std should be zero everywhere
    ensemble = np.tile(weights, (10, 1)).T  # shape (1000, 10)
    result = weighted_histogram(
        values, weights,
        bins=10, hist_range=(0, 10),
        uncertainty=ensemble, plot=False,
    )
    assert np.allclose(result["uncertainty"], 0.0), \
        "Identical ensemble members should give zero uncertainty"


def test_bin_centers_correct():
    """Bin centers must be midpoints of bin edges."""
    values = np.linspace(0, 10, 100)
    weights = np.ones(100)
    result = weighted_histogram(
        values, weights, bins=5, hist_range=(0, 10), plot=False
    )
    expected_centers = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    assert np.allclose(result["bin_centers"], expected_centers)


if __name__ == "__main__":
    try:
        import pandas as pd
        df = pd.read_hdf("data/multifold.h5")

        ensemble_cols = [f"weights_ensemble_{i}" for i in range(100)]
        ensemble_weights = df[ensemble_cols].values

        print("Running on real OmniFold data...")
        result = weighted_histogram(
            values=df["pT_ll"].values,
            weights=df["weights_nominal"].values,
            bins=50,
            hist_range=(0, 800),
            observable_name="pT_ll (GeV)",
            uncertainty=ensemble_weights,
            plot=True,
        )
        plt.savefig("pT_ll_unfolded.png", dpi=150, bbox_inches="tight")
        print("Plot saved to pT_ll_unfolded.png")
        print(f"Total weighted events: {result['counts'].sum():.4f}")

    except FileNotFoundError:
        print("Data file not found - skipping demo. Run pytest for tests.")

    pytest.main([__file__, "-v"])