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

The pipeline command (raw log to results table and plots) is added once the
pipeline entry point exists.

### Random seed

All randomness uses the seed in `config/pipeline.yaml` (`seed: 42`). Read it with
`src.seed.load_seed()` and apply it with `src.seed.set_seed(seed)` at the start of a run.


## Docker

Build the image:

    docker build -t spi-pipeline .

The raw logs are not part of the image, so mount `data/` into the container. Run the
test suite inside it:

    docker run --rm -v "$(pwd)/data:/app/data" spi-pipeline

The pipeline will run the same way, with its command appended:

    docker run --rm -v "$(pwd)/data:/app/data" spi-pipeline <pipeline command>

The pipeline command is added once the pipeline entry point exists.