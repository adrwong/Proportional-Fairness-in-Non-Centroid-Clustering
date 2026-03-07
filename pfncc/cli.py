from __future__ import annotations

import argparse

from .experiment import run_dataset_experiment
from .plotting import plot_all_metrics


def _k_values(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError("k-end must be >= k-start")
    return list(range(start, end + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PFNC clustering experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run experiments and save CSV files")
    run_p.add_argument("--dataset", choices=["iris", "diabetes", "adult"], required=True)
    run_p.add_argument("--k-start", type=int, required=True)
    run_p.add_argument("--k-end", type=int, required=True)
    run_p.add_argument("--out-dir", default="results")
    run_p.add_argument("--dist", default="euclidean")
    run_p.add_argument("--seed", type=int, default=0)

    plot_p = sub.add_parser("plot", help="Generate figures from CSV files")
    plot_p.add_argument("--dataset", choices=["iris", "diabetes", "adult"], required=True)
    plot_p.add_argument("--k-start", type=int, required=True)
    plot_p.add_argument("--k-end", type=int, required=True)
    plot_p.add_argument("--result-dir", default="results")
    plot_p.add_argument("--out-dir", default="figures")
    plot_p.add_argument("--repeats", type=int, default=40)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ks = _k_values(args.k_start, args.k_end)

    if args.command == "run":
        run_dataset_experiment(
            dataset=args.dataset,
            k_values=ks,
            out_dir=args.out_dir,
            dist_function=args.dist,
            seed=args.seed,
        )
        return

    if args.command == "plot":
        plot_all_metrics(
            dataset=args.dataset,
            k_values=ks,
            result_dir=args.result_dir,
            out_dir=args.out_dir,
            repeats=args.repeats,
        )
        return


if __name__ == "__main__":
    main()
