"""
Offline guards for eval_pareto.py's post-generation code paths.

These exist because a KeyError in the Pareto-printing block destroyed 2.2 h of
completed generation: the crash happened *before* the json was written, so the
run produced nothing. Anything that touches `results` after generation must be
exercised here, on synthetic data, in milliseconds.
"""
import json
import os
import sys

try:
    import pytest
except ImportError:  # standalone mode, see __main__
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _results_like_a_real_run():
    """Mirror the shape eval_pareto builds, diagnostics included."""
    return {
        1: {"k": 1, "method": "greedy (T=0, 1 sample)", "accuracy": 0.52,
            "avg_k": 1.0, "avg_tokens": 210.0, "correct": 208, "total": 400},
        8: {"k": 8, "method": "fixed-8", "accuracy": 0.66, "avg_k": 8.0,
            "avg_tokens": 1680.0, "correct": 264, "total": 400},
        "ave@k": {"k": 32, "method": "Ave@32 (mean single-rollout, T=0.7)",
                  "accuracy": 0.48, "avg_k": 1.0, "correct": 6144, "total": 12800},
        "conf": {"k": "dynamic", "method": "conformal", "accuracy": 0.68,
                 "avg_k": 9.175, "avg_tokens": 1930.0, "correct": 272,
                 "total": 400, "qhats": {2: 1.0, 32: 0.93}},
        # diagnostic entry: NOT a budget point, has none of avg_k/accuracy/method
        "theorem2_coverage_per_k": {
            "2": {"k": 2, "qhat": 1.0, "empirical_coverage": 0.99, "n": 400},
            "32": {"k": 32, "qhat": 0.93, "empirical_coverage": 0.88, "n": 400},
        },
    }


def _pareto_points(results):
    """The selection eval_pareto.py uses; must ignore diagnostic entries."""
    return [(r["avg_k"], r["accuracy"], r["method"])
            for r in results.values()
            if isinstance(r, dict) and {"avg_k", "accuracy", "method"} <= r.keys()]


def test_diagnostic_entries_do_not_break_pareto_points():
    """The exact regression: theorem2_coverage_per_k has no 'avg_k'."""
    pts = _pareto_points(_results_like_a_real_run())
    assert len(pts) == 4
    assert all(isinstance(c, (int, float)) for c, _, _ in pts)


def test_naive_selection_would_still_crash():
    """Pin why the guard is needed, so nobody 'simplifies' it away."""
    with pytest.raises(KeyError):
        [(r["avg_k"], r["accuracy"], r["method"])
         for r in _results_like_a_real_run().values()]


def test_points_are_sortable_by_cost():
    pts = sorted(_pareto_points(_results_like_a_real_run()), key=lambda x: x[0])
    assert [p[0] for p in pts] == sorted(p[0] for p in pts)


def test_results_payload_is_json_serialisable():
    """The save step strips nested dicts; make sure what remains round-trips."""
    results = _results_like_a_real_run()
    payload = {"results": {str(k): {kk: vv for kk, vv in v.items()
                                    if not isinstance(vv, dict)}
                           for k, v in results.items()},
               "theorem2_coverage_per_k": results["theorem2_coverage_per_k"]}
    json.loads(json.dumps(payload))


def test_coverage_block_survives_the_strip():
    """theorem2_coverage_per_k must reach the saved json intact.

    The "results" flattening keeps only scalars, so the coverage dict is
    reduced to {} there. It has to be carried at top level instead. An earlier
    version of this test asserted the {} outcome and passed while every run's
    coverage data was being discarded.
    """
    results = _results_like_a_real_run()
    stripped = {str(k): {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                for k, v in results.items()}
    saved = {"results": stripped,
             "theorem2_coverage_per_k": results.get("theorem2_coverage_per_k", {})}
    assert stripped["theorem2_coverage_per_k"] == {}                   # why top level
    assert saved["theorem2_coverage_per_k"] == results["theorem2_coverage_per_k"]
    assert saved["theorem2_coverage_per_k"]["32"]["empirical_coverage"] == 0.88


def test_eval_pareto_saves_coverage_at_top_level():
    """Guard the real file, not a re-implementation of it."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "eval_pareto.py")).read()
    block = src[src.index("save_data = {"):src.index("save_data = {") + 1500]
    assert '"theorem2_coverage_per_k": results.get("theorem2_coverage_per_k"' in block


def _eval_pareto_src():
    return open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "eval_pareto.py")).read()


def test_fixed_k_rows_never_exceed_generated_samples():
    """A k=64 row built from 32 samples is a mislabeled duplicate of k=32."""
    src = _eval_pareto_src()
    loop = src[src.index("    for k in FIXED_K_VALUES:"):]
    loop = loop[:loop.index("results[k] = {")]
    assert "k > K_MAX" in loop


def test_conformal_grid_limited_to_generated_budgets():
    """--k-max below the training grid must not index qhats that were never fit."""
    src = _eval_pareto_src()
    assert "conf_k_values = tuple(sorted(k for k in cfg.k_values if k <= K_MAX))" in src
    assert "generated_k_values = [k for k in FIXED_K_VALUES if k <= K_MAX]" in src
    assert "choose_k_softmax(answers, FIXED_K_VALUES" not in src
    assert "choose_k_entropy(answers, FIXED_K_VALUES" not in src


def test_no_hardcoded_k64_lookup():
    assert "results.get(64" not in _eval_pareto_src()


def test_small_kmax_stopping_rule_runs():
    """Behavioural check of the filtered grid on synthetic answers (k_max=8)."""
    K_MAX, k_values = 8, (2, 4, 8, 16, 32)
    qhats = {k: 0.9 for k in (2, 4, 8)}                 # only generated budgets are fit
    conf_k_values = tuple(sorted(k for k in k_values if k <= K_MAX))
    answers = ["7"] * K_MAX
    for k in conf_k_values:                              # would KeyError on 16 unfiltered
        assert k in qhats
    assert max(conf_k_values) == K_MAX and answers[:max(conf_k_values)] == answers


def test_eval_pareto_module_parses():
    import ast
    ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "eval_pareto.py")).read())


def _main():
    """Run every check without pytest, so preflight can use it in any env."""
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"  {len(fns)-failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # minimal pytest.raises shim so the file runs with no test framework
    if "pytest" not in sys.modules:
        import contextlib, types

        @contextlib.contextmanager
        def _raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__}")
        pytest = types.SimpleNamespace(raises=_raises)  # noqa: F811
        globals()["pytest"] = pytest
    sys.exit(_main())
