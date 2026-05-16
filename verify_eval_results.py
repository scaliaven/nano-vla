"""Verify and summarize the JSON output of eval_libero.py.

Checks each file in eval_results/:
  - sum(per_task.success) / sum(per_task.trials) matches the reported
    overall_success_rate (catches stale or hand-edited values)
  - every per-task entry has 0 <= success <= trials and trials > 0
  - trials are uniform within a suite (sanity check on --num-trials)

Then prints three tables: main per-suite, the per-step sweep, and the
chunk-length diagnostic. No heavy deps and no model loading.
"""
import argparse
import json
import math
from pathlib import Path


def _check(path: Path) -> tuple[float, int, int, int, list[str]]:
    """Return (recomputed_rate, total_success, total_trials, n_tasks, errors)."""
    d = json.loads(path.read_text())
    per_task = d.get("per_task", {})
    errors: list[str] = []

    total_s = total_t = 0
    trials_seen: set[int] = set()
    for name, row in per_task.items():
        s, t = row.get("success"), row.get("trials")
        if not isinstance(s, int) or not isinstance(t, int) or t <= 0:
            errors.append(f"{name}: bad success/trials ({s}/{t})")
            continue
        if s < 0 or s > t:
            errors.append(f"{name}: success {s} out of range [0, {t}]")
        rate_row = row.get("rate")
        if rate_row is not None and not math.isclose(rate_row, s / t, abs_tol=1e-9):
            errors.append(f"{name}: per-task rate {rate_row} != {s}/{t}")
        total_s += s
        total_t += t
        trials_seen.add(t)

    recomputed = total_s / total_t if total_t else 0.0
    reported = d.get("overall_success_rate")
    if reported is None or not math.isclose(reported, recomputed, abs_tol=1e-9):
        errors.append(f"overall_success_rate {reported} != recomputed {recomputed}")
    if len(trials_seen) > 1:
        errors.append(f"non-uniform trials per task: {sorted(trials_seen)}")

    return recomputed, total_s, total_t, len(per_task), errors


def _fmt_rate(r: float) -> str:
    return f"{r * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("eval_results"))
    args = ap.parse_args()

    root: Path = args.dir
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    main_files = sorted(p for p in root.glob("*.json"))
    sweep_files = sorted(root.glob("sweep/*/step_*.json"))
    diag_files = sorted(root.glob("diag/*.json"))

    any_err = False

    print(f"== main per-suite ({root}/*.json) ==")
    print(f"{'suite':<18} {'tasks':>5} {'trials':>7} {'rate':>7}  errors")
    for p in main_files:
        rate, s, t, n, errs = _check(p)
        any_err |= bool(errs)
        marker = "" if not errs else "  " + "; ".join(errs)
        print(f"{p.stem:<18} {n:>5} {t:>7} {_fmt_rate(rate):>7}{marker}")

    if sweep_files:
        print(f"\n== per-step sweep ({root}/sweep/<suite>/step_*.json) ==")
        by_suite: dict[str, list[tuple[int, float, int]]] = {}
        for p in sweep_files:
            suite = p.parent.name
            step = int(p.stem.split("_")[-1])
            rate, _, t, _, errs = _check(p)
            any_err |= bool(errs)
            by_suite.setdefault(suite, []).append((step, rate, t))
            if errs:
                print(f"  ! {suite}/{p.name}: {'; '.join(errs)}")
        for suite, rows in by_suite.items():
            rows.sort()
            best_step, best_rate, _ = max(rows, key=lambda r: r[1])
            trials = rows[0][2]
            print(f"\n  {suite}  (trials/run = {trials})")
            for step, rate, _ in rows:
                mark = "  <- best" if step == best_step else ""
                print(f"    step {step:>6}  {_fmt_rate(rate)}{mark}")
            print(f"    best step (this sweep): {best_step}")

        best_path = root / "sweep" / "_best_steps.json"
        if best_path.is_file():
            recorded = json.loads(best_path.read_text())
            mismatches = []
            for suite, rows in by_suite.items():
                best_step = max(rows, key=lambda r: r[1])[0]
                if recorded.get(suite) != best_step:
                    mismatches.append(f"{suite}: recorded {recorded.get(suite)} vs recomputed {best_step}")
            if mismatches:
                any_err = True
                print("\n  ! _best_steps.json disagrees with sweep:")
                for m in mismatches:
                    print(f"    {m}")
            else:
                print("\n  _best_steps.json matches recomputed best steps.")

    if diag_files:
        print(f"\n== chunk-length diagnostic ({root}/diag/*.json) ==")
        # Filenames look like libero_goal_step030000_cl8.json — split off "_cl<int>".
        groups: dict[tuple[str, int], list[tuple[int, float, int]]] = {}
        for p in diag_files:
            stem = p.stem
            assert "_cl" in stem, f"unexpected diag filename: {p.name}"
            head, cl = stem.rsplit("_cl", 1)
            suite, step_tok = head.rsplit("_step", 1)
            rate, _, t, _, errs = _check(p)
            any_err |= bool(errs)
            if errs:
                print(f"  ! {p.name}: {'; '.join(errs)}")
            groups.setdefault((suite, int(step_tok)), []).append((int(cl), rate, t))
        for (suite, step), rows in groups.items():
            rows.sort()
            trials = rows[0][2]
            print(f"\n  {suite} @ step {step}  (trials/run = {trials})")
            for cl, rate, _ in rows:
                print(f"    chunk_size={cl}  {_fmt_rate(rate)}")

    print("\n" + ("OK — no inconsistencies." if not any_err else "FAILED — see errors above."))
    raise SystemExit(1 if any_err else 0)


if __name__ == "__main__":
    main()
