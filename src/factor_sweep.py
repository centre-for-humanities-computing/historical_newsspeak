"""
Systematic sweep over factor count (2 through max_factors), checking at
each step whether the solution is statistically PROPER (no Heywood
cases -- no negative uniquenesses) rather than just eyeballing loadings
one factor-count at a time.

This turns "we tried 3, 5, and 6 individually and picked 3" into a
reportable, systematic result: either some k>3 also gives a proper,
interpretable solution (in which case that's worth adopting instead),
or NOTHING beyond 3 is statistically valid -- which is itself a strong,
direct rebuttal to "a more generalized metric could be derived via
dimensionality reduction": the data structurally does not support more
dimensions than that, evidenced systematically rather than assumed.

Usage:
    from factor_sweep import run_factor_sweep
    sweep_df = run_factor_sweep(X_final, fa_features_final, factor_range=range(2, 9))
    print(sweep_df.to_string(index=False))
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer


def run_factor_sweep(X, feature_names, factor_range=range(2, 9), rotation="promax"):
    """
    X: the same dropna()'d feature matrix used for the main factor analysis.
    feature_names: column names, in the same order as X's columns.
    factor_range: which factor counts to test.

    Returns a DataFrame with one row per factor count, including whether
    the solution is PROPER (no negative/near-zero-forced uniquenesses)
    and how much cumulative variance it explains.
    """
    results = []
    all_loadings = {}
    all_uniquenesses = {}

    for k in factor_range:
        fa = FactorAnalyzer(n_factors=k, rotation=rotation)
        try:
            fa.fit(X)
        except Exception as e:
            results.append({
                "n_factors": k,
                "cumulative_var": np.nan,
                "n_heywood_cases": np.nan,
                "min_uniqueness": np.nan,
                "proper_solution": False,
                "fit_error": str(e),
            })
            continue

        uniq = fa.get_uniquenesses()
        n_heywood = int((uniq < 0).sum())
        # A uniqueness that's exactly 0 (not just negative) is also a
        # degenerate/improper case worth flagging, not just strictly negative.
        n_degenerate = int((uniq <= 1e-6).sum())

        _, _, cum_var = fa.get_factor_variance()

        results.append({
            "n_factors": k,
            "cumulative_var": cum_var[-1],
            "n_heywood_cases": n_heywood,
            "n_degenerate_cases": n_degenerate,
            "min_uniqueness": uniq.min(),
            "proper_solution": (n_heywood == 0),
            "fit_error": None,
        })

        all_loadings[k] = pd.DataFrame(fa.loadings_, index=feature_names,
                                        columns=[f"F{i+1}" for i in range(k)])
        all_uniquenesses[k] = pd.Series(uniq, index=feature_names)

    sweep_df = pd.DataFrame(results)

    proper = sweep_df[sweep_df["proper_solution"]]
    if len(proper) > 0:
        best_k = int(proper["n_factors"].max())
        print(f"\nLargest PROPER (no Heywood case) solution: {best_k} factors")
    else:
        best_k = None
        print("\nNo tested factor count produced a fully proper solution.")

    return sweep_df, all_loadings, all_uniquenesses, best_k


if __name__ == "__main__":
    print("Import this module and call run_factor_sweep(X_final, fa_features_final)")
    print("against your actual dropna()'d feature matrix -- see docstring above.")