# STI Comparison

This project provides a small command line tool to compare two STI (System Technical Interface) matrices defined in a YAML configuration file. It outputs the differences for a set of fields and optionally verifies that referenced documents exist in a PPD spreadsheet.

## Usage

```bash
python compare_sti.py MATRIX1 MATRIX2 [-c path/to/config] [-o differences.xlsx] [--ppd path/to/ppd.xlsx]
```

- `MATRIX1` and `MATRIX2` are the names of the matrices as defined in the YAML configuration.
- `-c/--config` points to the configuration file (defaults to `sti_config.yaml`).
- `-o/--output` writes the differences to an Excel file.
- `--ppd` enables checking that all referenced documents are listed in the given PPD file.

## Development

Install the development dependencies and run the tests with coverage:

```bash
pip install -r requirements.txt  # optional: pandas, pytest, pytest-cov
pytest --cov=compare_sti.py --cov-report=term-missing
```

The test suite aims for 100% code coverage.
