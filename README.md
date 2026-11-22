# supervised-process-intelligence
## Requirements

- Python 3.12
- Install the dependencies with `pip install -r requirements.txt`



## Data

The raw logs are not committed (`data/` is git-ignored). Place them in `data/`.

**Helpdesk** (`data/help-desk-log.csv`): "Dataset belonging to the help desk log of an Italian Company",
M. Polato, 4TU.ResearchData, 2017, DOI 10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb.
Original file name `finale.csv`. MD5: `a91d74377c55be5cbdfda05dae69db30`.
4,580 cases, 21,348 events, 14 activities. The code never modifies the raw file.


## Docker

Build the image:

    docker build -t spi-pipeline .

Run the test suite inside the container:

    docker run --rm spi-pipeline

The raw logs are not part of the image. Mount them when you run the pipeline:

    docker run --rm -v "$(pwd)/data:/app/data" spi-pipeline <pipeline command>

The pipeline command is added once the pipeline entry point exists.