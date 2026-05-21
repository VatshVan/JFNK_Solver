"""Build a multi-panel convergence dashboard from JSON-Lines solver telemetry."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("logs") / ".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _show_unavailable(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")


def _maybe_parse_history(value) -> list[float] | None:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    return None


def plot_dashboard(metrics_path: Path, output_path: Path) -> Path:
    plt.style.use("seaborn-v0_8-darkgrid")
    frame = pd.read_json(metrics_path, lines=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=200)
    axes = axes.ravel()

    if "timestep" not in frame:
        raise ValueError("metrics.jsonl must contain a 'timestep' column.")

    timestep = frame["timestep"]

    ax = axes[0]
    if 'fgmres_baseline' in frame.columns and 'fgmres_drrn' in frame.columns:
        cont = frame.dropna(subset=['fgmres_baseline', 'fgmres_drrn']).copy()

        # Keep only plain DRRN initial-guess records, not two-phase solve records.
        # Two-phase records have acceleration_factor < 1.0 (they are slower).
        # Plain continuation records always have acceleration_factor >= 1.0.
        if 'acceleration_factor' in cont.columns:
            cont = cont[cont['acceleration_factor'].fillna(0) >= 1.0]

        # Deduplicate: for each Re, keep the row with the highest fgmres_baseline
        # (the first/cleanest continuation evaluation, before any two-phase contamination).
        cont = (cont.sort_values('fgmres_baseline', ascending=False)
                    .drop_duplicates(subset=['Re'])
                    .sort_values('Re'))

        if len(cont) > 0:
            re_labels = [f"Re={int(r)}" for r in cont['Re']]
            x = np.arange(len(re_labels))
            width = 0.35

            ax.bar(x - width/2, cont['fgmres_baseline'], width,
                   label='Baseline FGMRES', color='steelblue', alpha=0.85)
            ax.bar(x + width/2, cont['fgmres_drrn'], width,
                   label='DRRN initial guess', color='darkorange', alpha=0.85)

            for i, (b, d) in enumerate(zip(cont['fgmres_baseline'], cont['fgmres_drrn'])):
                speedup = b / max(d, 1)
                ax.text(x[i], max(b, d) + 0.5, f'{speedup:.2f}×',
                        ha='center', va='bottom', fontsize=8, color='black',
                        fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(re_labels, rotation=30, ha='right')
            ax.set_ylabel('FGMRES Iterations')
            ax.set_title('FGMRES Iterations: Baseline vs DRRN Initial Guess')
            ax.legend()
        else:
            _show_unavailable(ax, "No plain continuation records found")
    else:
        _show_unavailable(ax, "No baseline/DRRN iteration data")

    ax = axes[1]
    histories = []
    if "fgmres_residual_history" in frame:
        histories = [_maybe_parse_history(value) for value in frame["fgmres_residual_history"].tail(5)]
        histories = [history for history in histories if history]
    if histories:
        for index, history in enumerate(histories, start=1):
            ax.semilogy(range(1, len(history) + 1), history, linewidth=1.8, label=f"Step {-len(histories) + index - 1:+d}")
        ax.set_xlabel("Inner Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_title("FGMRES Residual Histories")
        ax.legend()
    else:
        _show_unavailable(ax, "No residual histories available")

    ax = axes[2]
    if 'cond_estimate' in frame.columns:
        cond_data = frame[
            frame['cond_estimate'].notna() &
            (frame['cond_estimate'] > 1e-10) &
            (frame['cond_estimate'] < 1.0)    # reciprocal values are always < 1
        ].copy()
        if len(cond_data) > 0:
            # cond_estimate stores sigma_min/sigma_max = 1/kappa
            # Invert to get the true condition number kappa
            kappa = 1.0 / cond_data['cond_estimate']

            # Label x-axis by Re value if available, otherwise use timestep
            if 'Re' in cond_data.columns:
                x_vals = cond_data['Re']
                xlabel = 'Reynolds Number'
            else:
                x_vals = cond_data['timestep']
                xlabel = 'Continuation Step'

            ax.scatter(x_vals, kappa, color='darkorange', s=60, zorder=5,
                       label='κ = 1/cond_estimate')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Condition Number κ (log scale)')
            ax.set_title('Preconditioned System Conditioning')
            ax.legend(fontsize=8)
        else:
            _show_unavailable(ax,
                "KSPComputeExtremeSingularValues not called\n"
                "per timestep. Only evaluated at\n"
                "continuation boundaries (7 points).")
    else:
        _show_unavailable(ax, "No condition estimate data")

    ax = axes[3]
    if 'div_l2' in frame.columns:
        # Projection step records have ppe_iters logged; predictor records do not.
        # We want ONLY the projection step divergence (the true mass conservation metric).
        if 'ppe_iters' in frame.columns:
            proj = frame[frame['ppe_iters'].notna() & frame['div_l2'].notna()].copy()
        else:
            # Fallback: use all records but filter to those with div_l2 < 1e-3
            # (projection-step values are always < 1e-6; predictor values are 20-200)
            proj = frame[frame['div_l2'].notna() & (frame['div_l2'] < 1.0)].copy()

        if len(proj) > 0 and proj['div_l2'].max() > 0:
            ax.semilogy(proj['timestep'], proj['div_l2'],
                        label='Divergence L2', color='steelblue',
                        linewidth=0.8, alpha=0.7)
            if 'div_linf' in proj.columns and proj['div_linf'].notna().any():
                ax.semilogy(proj['timestep'], proj['div_linf'],
                            label='Divergence Linf', color='darkorange',
                            linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Divergence Norm (log scale)')
            ax.set_title('Mass Conservation (post-projection ∇·u)')
            ax.legend()
            print(f"Panel 4: {len(proj)} projection records, "
                  f"div_l2 range [{proj['div_l2'].min():.2e}, {proj['div_l2'].max():.2e}]")
        else:
            # No projection records distinguishable — show all on log scale with note
            valid = frame[frame['div_l2'].notna() & (frame['div_l2'] > 0)].copy()
            if len(valid) > 0:
                ax.semilogy(valid['timestep'], valid['div_l2'],
                            color='steelblue', linewidth=0.6, alpha=0.6,
                            label='div_l2 (predictor + projection mixed)')
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Divergence Norm (log scale)')
                ax.set_title('Mass Conservation (mixed: see note)')
                ax.legend(fontsize=7)
            else:
                _show_unavailable(ax, "No div_l2 data")
    else:
        _show_unavailable(ax, "No divergence data")

    ax = axes[4]
    if 'cfl_max' in frame.columns:
        cfl_data = frame[
            frame['cfl_max'].notna() & (frame['cfl_max'] > 0)
        ].copy()

        if len(cfl_data) > 0:
            if len(cfl_data) <= 20:
                # Few points: bar chart per Re
                if 'Re' in cfl_data.columns:
                    cfl_data = cfl_data.sort_values('Re')
                    re_labels = [f"Re={int(r)}" for r in cfl_data['Re']]
                    x = np.arange(len(re_labels))
                    ax.bar(x, cfl_data['cfl_max'], color='steelblue', alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(re_labels, rotation=30, ha='right')
                else:
                    ax.bar(range(len(cfl_data)), cfl_data['cfl_max'],
                           color='steelblue', alpha=0.8)
            else:
                # Many points: line plot per Re segment to avoid spaghetti joins.
                # Strategy: assign a monotonically increasing global index,
                # then insert NaN rows between Re segments so matplotlib
                # draws a break rather than a connecting diagonal line.
                if 'Re' in cfl_data.columns:
                    cfl_data = cfl_data.sort_values(['Re', 'timestep'])
                    re_groups = cfl_data['Re'].unique()
                    plot_idx = []
                    plot_cfl = []
                    cursor = 0
                    for re_val in sorted(re_groups):
                        seg = cfl_data[cfl_data['Re'] == re_val].sort_values('timestep')
                        n = len(seg)
                        plot_idx.extend(range(cursor, cursor + n))
                        plot_cfl.extend(seg['cfl_max'].tolist())
                        # Insert a NaN break so matplotlib does not join segments
                        plot_idx.append(cursor + n)
                        plot_cfl.append(float('nan'))
                        cursor += n + 1
                else:
                    # No Re column: use a sequential global index (no breaks needed)
                    cfl_data = cfl_data.sort_values('timestep')
                    plot_idx = list(range(len(cfl_data)))
                    plot_cfl = cfl_data['cfl_max'].tolist()

                ax.plot(plot_idx, plot_cfl, color='steelblue', linewidth=0.7,
                        alpha=0.85)
                ax.axhline(y=1.0, color='black', linestyle='--',
                           linewidth=0.9, label='CFL = 1')
                ax.set_xlabel('Global Solver Step Index')
                ax.legend()

            ax.set_ylabel('CFL Number')
            ax.set_title('CFL Monitor')
        else:
            _show_unavailable(ax,
                "CFL not logged per timestep.\n"
                "Instrument predictor_step() to log CFL.")
    else:
        _show_unavailable(ax, "No CFL data")

    ax = axes[5]
    if "t_petsc_solve_ms" in frame or "t_drrn_inference_ms" in frame:
        petsc_ms = frame["t_petsc_solve_ms"] if "t_petsc_solve_ms" in frame else 0.0
        drrn_ms = frame["t_drrn_inference_ms"] if "t_drrn_inference_ms" in frame else 0.0
        ax.bar(timestep, petsc_ms, label="PETSc solve", color="tab:blue")
        ax.bar(timestep, drrn_ms, bottom=petsc_ms, label="DRRN inference", color="tab:purple")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Time [ms]")
        ax.set_title("Runtime + VRAM")
        ax.legend(loc="upper left")
        if "vram_peak_mb" in frame:
            ax2 = ax.twinx()
            ax2.plot(timestep, frame["vram_peak_mb"], color="tab:green", linewidth=2.0, marker="o")
            ax2.set_ylabel("VRAM Peak [MB]")
    else:
        _show_unavailable(ax, "No runtime metrics available")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the solver convergence dashboard.")
    parser.add_argument("--metrics", type=Path, default=Path("logs/metrics.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/convergence_dashboard.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_dashboard(metrics_path=args.metrics, output_path=args.output)
    print(output_path)


if __name__ == "__main__":
    main()
