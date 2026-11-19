# supervised-process-intelligence
## Requirements

- Python 3.12
- Install the dependencies with `pip install -r requirements.txt`

##Data

## Docker

Build the image:

    docker build -t spi-pipeline .

Run the test suite inside the container:

    docker run --rm spi-pipeline

The raw logs are not part of the image. Mount them when you run the pipeline:

    docker run --rm -v "$(pwd)/data:/app/data" spi-pipeline <pipeline command>

The pipeline command is added once the pipeline entry point exists.
