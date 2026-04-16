"""
Tests for iterative parameter reconciliation with correct prior accounting.

Key invariants:
  - n_iter=1: identical results with and without cumulative_log_corrections
  - log-linear model + n_iter=2 WITH fix: stable (no change vs iter 1)
  - nonlinear model + n_iter=2: WITH fix differs from WITHOUT fix
"""
import numpy as np
import pytest
import flodym as fd

from remind_mfa.common.common_parameter_reconciliation import CommonParameterReconciliation


def _make_dims():
    return fd.DimensionSet(dim_list=[fd.Dimension(name="Region", letter="r", items=["A"])])


class _BaseToyReconciliation(CommonParameterReconciliation):
    """Toy reconciliation with 1D scalar parameters, no real MFA system needed."""

    def __init__(self, p1: float, p2: float, **kwargs):
        self._p1_init = p1
        self._p2_init = p2
        super().__init__(ref_mfa=None, uncoupled=True, **kwargs)

    def prepare_dims(self):
        self.input_dims = None  # no reduced stock-type expansion needed

    def prepare_prms(self):
        dims = _make_dims()
        p1 = fd.FlodymArray(dims=dims, values=np.array([self._p1_init]))
        p2 = fd.FlodymArray(dims=dims, values=np.array([self._p2_init]))
        self.prms = {"p1": p1.copy(), "p2": p2.copy()}
        self.input_prms = {
            "p1": fd.Parameter(name="p1", dims=dims, values=p1.values.copy()),
            "p2": fd.Parameter(name="p2", dims=dims, values=p2.values.copy()),
        }
        self.prms_adj_dims = {"p1": dims, "p2": dims}

    def prepare_flws(self):
        self.flws = {}

    def prepare_stks(self):
        self.stks = {}

    def prepare_trds(self):
        self.trds = None

    def rel_std(self, prm_name: str) -> fd.FlodymArray:
        return fd.FlodymArray(dims=_make_dims(), values=np.array([0.5]))

    def cast_correction_to_original_prm_dim(self, correction_factor):
        return correction_factor  # no stock-type dimension expansion needed


class LinearToyReconciliation(_BaseToyReconciliation):
    """td = p1, bu = p2  (log-linear: exact solution in 1 iteration)."""

    def calc_top_down_stock(self, prm):
        return prm["p1"].copy()

    def calc_bottom_up_stock(self, prm):
        return prm["p2"].copy()


class NonlinearToyReconciliation(_BaseToyReconciliation):
    """td = p1 + 1, bu = p2  (additive offset breaks log-linearity)."""

    def calc_top_down_stock(self, prm):
        dims = _make_dims()
        return fd.FlodymArray(dims=dims, values=prm["p1"].values + 1.0)

    def calc_bottom_up_stock(self, prm):
        return prm["p2"].copy()


def _run_n_iter(cls, p1_init: float, p2_init: float, n_iter: int, use_fix: bool):
    """Simulate the reconcile_parameters loop for n_iter iterations."""
    cumulative_log_corrections = {}
    p1, p2 = p1_init, p2_init
    output_prms = None
    for _ in range(n_iter):
        kwargs = {"cumulative_log_corrections": cumulative_log_corrections} if use_fix else {}
        toy = cls(p1=p1, p2=p2, **kwargs)
        output_prms = toy.correct_parameters()
        if use_fix:
            cumulative_log_corrections = toy.cumulative_log_corrections
        # feed corrected params into next iteration (mirrors common_model.py)
        p1 = output_prms["p1"].values.item()
        p2 = output_prms["p2"].values.item()
    return output_prms


# ── tests ──────────────────────────────────────────────────────────────────────

def test_single_iteration_unchanged_linear():
    """n=1: fix must not change the result (linear model)."""
    res_fix = _run_n_iter(LinearToyReconciliation, 4.0, 2.0, n_iter=1, use_fix=True)
    res_no = _run_n_iter(LinearToyReconciliation, 4.0, 2.0, n_iter=1, use_fix=False)
    np.testing.assert_allclose(res_fix["p1"].values, res_no["p1"].values, rtol=1e-10)
    np.testing.assert_allclose(res_fix["p2"].values, res_no["p2"].values, rtol=1e-10)


def test_single_iteration_unchanged_nonlinear():
    """n=1: fix must not change the result (nonlinear model)."""
    res_fix = _run_n_iter(NonlinearToyReconciliation, 1.0, 10.0, n_iter=1, use_fix=True)
    res_no = _run_n_iter(NonlinearToyReconciliation, 1.0, 10.0, n_iter=1, use_fix=False)
    np.testing.assert_allclose(res_fix["p1"].values, res_no["p1"].values, rtol=1e-10)
    np.testing.assert_allclose(res_fix["p2"].values, res_no["p2"].values, rtol=1e-10)


def test_linear_model_stable_at_n2():
    """For a log-linear model the fix must produce zero incremental change in iter 2."""
    res1 = _run_n_iter(LinearToyReconciliation, 4.0, 2.0, n_iter=1, use_fix=True)
    res2 = _run_n_iter(LinearToyReconciliation, 4.0, 2.0, n_iter=2, use_fix=True)
    np.testing.assert_allclose(res1["p1"].values, res2["p1"].values, rtol=1e-6)
    np.testing.assert_allclose(res1["p2"].values, res2["p2"].values, rtol=1e-6)


def test_nonlinear_fix_changes_n2_result():
    """For the nonlinear model, fix and no-fix must diverge at n=2."""
    res_fix = _run_n_iter(NonlinearToyReconciliation, 1.0, 10.0, n_iter=2, use_fix=True)
    res_no = _run_n_iter(NonlinearToyReconciliation, 1.0, 10.0, n_iter=2, use_fix=False)
    assert not np.allclose(res_fix["p1"].values, res_no["p1"].values, rtol=1e-6), (
        "Fix should produce different p1 for n_iter=2 on a nonlinear model"
    )
