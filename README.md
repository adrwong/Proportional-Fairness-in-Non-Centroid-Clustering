PFNC Clustering Experiments

What is implemented

- Greedy Capture (`FGC`) baseline
- `k-means++` baseline
- `k-medoids` baseline (PAM-style implementation)
- Approximate FJR, exact FJR violation, and Core violation
- Cost metrics used in the reference scripts
- CSV export compatible with the original format (`6 x 7` matrix per `k`)
- Plot generation for all reported metrics

Run setup

From this folder:

- `uv run pfncc --help`

Run experiments

- Iris (single run for Greedy Capture, repeated baselines):
	- `uv run pfncc run --dataset iris --k-start 2 --k-end 10 --out-dir results/iris`
- Diabetes (40 outer repeats with sample size 100):
	- `uv run pfncc run --dataset diabetes --k-start 5 --k-end 15 --out-dir results/diabetes`
- Adult (40 outer repeats with sample size 100):
	- `uv run pfncc run --dataset adult --k-start 5 --k-end 25 --out-dir results/adult`

Generate figures

- `uv run pfncc plot --dataset adult --k-start 5 --k-end 25 --result-dir results/adult --out-dir figures/adult --repeats 40`

Run all paper pipelines automatically

- Full run (all datasets + plots, saved under `artifacts/`):
	- `./scripts/run_all_pipelines.sh`
- Single dataset examples:
	- `./scripts/run_all_pipelines.sh --only iris`
	- `./scripts/run_all_pipelines.sh --only diabetes`
	- `./scripts/run_all_pipelines.sh --only adult`
- Custom output root:
	- `./scripts/run_all_pipelines.sh --out-root /path/to/output`

Notes

- Dataset loading is portable:
	- `iris` from scikit-learn
	- `diabetes` from scikit-learn numeric diabetes dataset
	- `adult` from OpenML (`adult`, v2)
- The MILP checks for FJR/Core use SciPy HiGHS (`scipy.optimize.milp`) to avoid requiring Gurobi.
