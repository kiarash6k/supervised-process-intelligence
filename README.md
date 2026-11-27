# supervised-process-intelligence
## Requirements

- Python 3.12
- Install the dependencies with `pip install -r requirements.txt`



## Data

The raw logs are stored in `data/`. The Helpdesk log is included in the repository, so
it needs no separate download.

**Helpdesk** (`data/help-desk-log.csv`): "Dataset belonging to the help desk log of an Italian Company",
M. Polato, 4TU.ResearchData, 2017, DOI 10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb.
Original file name `finale.csv`. MD5: `a91d74377c55be5cbdfda05dae69db30`.
4,580 cases, 21,348 events, 14 activities. The code never modifies the raw file.


## Usage

Run all commands from the repository root with the dependencies installed.

Run the tests (some of them read `data/help-desk-log.csv`):

    python -m pytest

Compute the descriptive statistics of a log (case, event, activity and variant counts
and the case duration). The result is printed and saved to `artifacts/statistics_<name>.json`:

    python -m src.explore config/helpdesk.yaml

### Run the pipeline

From the raw log to the errors of the mean baseline and the Ridge model:

    python -m src.pipeline config/helpdesk.yaml

It builds the prefixes with their remaining time in hours, splits the cases by start time,
fits both models on the training cases and prints MAE and RMSE for the validation and test
sets. It writes these files to `artifacts/` (change the folder with `--output`):

- `results_<name>.csv`: the errors per model and set
- `predictions_<name>.csv`: the true and predicted remaining time of every prefix
- `split_<name>.json`: which case is in which set

The split shares, the Ridge `alpha` and the seed are set in `config/pipeline.yaml`.
Plots are not generated yet.

Known limitation: training and validation cases that are still running when the next set
starts are not removed yet (REQ-F15), so the errors are preliminary.

### Random seed

All randomness uses the seed in `config/pipeline.yaml` (`seed: 42`). Read it with
`src.seed.load_seed()` and apply it with `src.seed.set_seed(seed)` at the start of a run.


## Docker

Build the image:

    docker build -t spi-pipeline .

The raw logs are not part of the image, so mount `data/` into the container. Run the
test suite inside it:

    docker run --rm -v "$(pwd)/data:/app/data" spi-pipeline

Run the pipeline. The second mount keeps the output files on your computer:

    docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/artifacts:/app/artifacts" \
        spi-pipeline python -m src.pipeline config/helpdesk.yaml