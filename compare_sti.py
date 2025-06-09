"""Compare STI matrices defined by ``sti_config.yaml``.

This module provides a small command line interface that loads two matrices as
described in the YAML configuration and reports the differences for a set of
predefined fields. The code is organised with a few helper classes so the
script remains easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Set

import re

import argparse
import pandas as pd
import yaml

# Regex pattern for document identifiers within STI columns
ID_PATTERN = re.compile(r"(?:DID[0-9]{10}|CMD[0-9]{6,}|PM[0-9]{6,}|SETC[0-9]{6,})")


class STIConfig:
    """Load and query the ``sti_config.yaml`` configuration."""

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

    @property
    def fields_to_compare(self) -> list[str]:
        """Return the list of fields to compare between matrices."""

        return list(self._data.get("fields_to_compare", []))

    def matrix_definition(self, name: str) -> dict[str, Any]:
        """Return the raw configuration dictionary for a matrix."""

        for matrix in self._data.get("matrices", []):
            if matrix.get("name") == name:
                return matrix
        raise KeyError(f"Matrix '{name}' not found in {self.path}")


@dataclass
class STIMatrixDefinition:
    """Definition of a single STI matrix extracted from the YAML config."""

    name: str
    file: str
    sti_sheet: str
    column_mapping: dict[str, str]
    header_row: int | None

    @classmethod
    def from_config(cls, config: STIConfig, name: str) -> "STIMatrixDefinition":
        """Create the definition for ``name`` using ``config``."""

        raw = config.matrix_definition(name)
        sheet_name = raw["sti_sheet"]
        header_info = raw.get("sheets", {}).get(sheet_name, {})
        return cls(
            name=name,
            file=raw["file"],
            sti_sheet=sheet_name,
            column_mapping=raw.get("column_mapping", {}),
            header_row=header_info.get("header_row"),
        )


class STIMatrix:
    """Represent a STI matrix loaded from an Excel file."""

    def __init__(self, definition: STIMatrixDefinition, fields: Iterable[str]) -> None:
        self.definition = definition
        self.fields = list(fields)

    def load(self) -> pd.DataFrame:
        """Load the Excel file and normalise its columns."""

        file_path = Path(self.definition.file)
        header = self.definition.header_row if self.definition.header_row is not None else 0

        df = pd.read_excel(file_path, sheet_name=self.definition.sti_sheet, header=header)
        df.rename(columns=self.definition.column_mapping, inplace=True)

        required = ["Reference"] + self.fields
        for col in required:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[required]

        # Extract document identifiers from specific columns
        id_columns = {"MOP_design", "MOP_test", "CAF_Comments"}
        for col in id_columns.intersection(df.columns):
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .apply(lambda text: set(ID_PATTERN.findall(text)))
            )

        return df


class PPDChecker:
    """Load a PPD Excel file and query its document identifiers."""

    def __init__(self, path: str, column: str = "Référence ALSTOM") -> None:
        self.path = Path(path)
        self.column = column

    def load_ids(self) -> Set[str]:
        """Return the set of IDs present in the PPD file."""

        df = pd.read_excel(self.path)
        if self.column not in df.columns:
            raise KeyError(f"Column '{self.column}' not found in {self.path}")
        return set(df[self.column].dropna().astype(str))


class STIMatrixComparator:
    """Compare two :class:`STIMatrix` objects."""

    def __init__(self, fields: Iterable[str]) -> None:
        self.fields = list(fields)

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Return True if ``a`` and ``b`` should be considered equal."""

        if isinstance(a, set) or isinstance(b, set):
            a_set = a if isinstance(a, set) else set()
            b_set = b if isinstance(b, set) else set()
            return a_set == b_set
        if pd.isna(a) and pd.isna(b):
            return True
        return str(a) == str(b)

    def compare(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame describing the differences between ``df1`` and ``df2``."""

        merged = pd.merge(df1, df2, on="Reference", how="outer", suffixes=("_1", "_2"))
        diffs: list[pd.DataFrame] = []
        for field in self.fields:
            col1 = f"{field}_1"
            col2 = f"{field}_2"
            mism = merged[~merged.apply(lambda row: self._values_equal(row[col1], row[col2]), axis=1)]
            if not mism.empty:
                subset = mism[["Reference", col1, col2]].copy()
                subset["field"] = field
                subset.rename(columns={col1: "value_1", col2: "value_2"}, inplace=True)
                diffs.append(subset[["Reference", "field", "value_1", "value_2"]])
        if diffs:
            return pd.concat(diffs, ignore_index=True)
        return pd.DataFrame(columns=["Reference", "field", "value_1", "value_2"])


def collect_document_ids(df: pd.DataFrame, columns: Iterable[str]) -> Set[str]:
    """Return a set of IDs gathered from ``df`` across ``columns``."""

    ids: Set[str] = set()
    for col in columns:
        if col in df.columns:
            ids.update(
                {
                    id_str
                    for item in df[col]
                    if isinstance(item, set)
                    for id_str in item
                }
            )
    return ids


def main() -> None:
    """Entry point for the command line interface."""

    parser = argparse.ArgumentParser(
        description="Compare two STI matrices defined in sti_config.yaml"
    )
    parser.add_argument("matrix1", help="Name of the first matrix in the config")
    parser.add_argument("matrix2", help="Name of the second matrix in the config")
    parser.add_argument(
        "-c",
        "--config",
        default="sti_config.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument("-o", "--output", help="Optional Excel file to store the differences")
    parser.add_argument("--ppd", help="Optional PPD Excel file to verify documents")

    args = parser.parse_args()

    config = STIConfig(args.config)
    fields = config.fields_to_compare

    matrix1_def = STIMatrixDefinition.from_config(config, args.matrix1)
    matrix2_def = STIMatrixDefinition.from_config(config, args.matrix2)

    df1 = STIMatrix(matrix1_def, fields).load()
    df2 = STIMatrix(matrix2_def, fields).load()

    comparator = STIMatrixComparator(fields)
    diffs = comparator.compare(df1, df2)

    if diffs.empty:
        print("No differences found")
    else:
        print(diffs.to_string(index=False))
        if args.output:
            diffs.to_excel(args.output, index=False)
            print(f"Differences written to {args.output}")

    if args.ppd:
        checker = PPDChecker(args.ppd)
        ppd_ids = checker.load_ids()
        all_ids = collect_document_ids(
            pd.concat([df1, df2], ignore_index=True),
            ["MOP_design", "MOP_test", "CAF_Comments"],
        )
        missing = sorted(all_ids - ppd_ids)
        if missing:
            print("Missing in PPD:", ", ".join(missing))
        else:
            print("All referenced MOP documents are present in the PPD file")


if __name__ == "__main__":
    main()
