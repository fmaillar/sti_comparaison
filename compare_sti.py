"""Compare STI matrices defined by ``sti_config.yaml``.

This module provides a small command line interface that loads two matrices as
described in the YAML configuration and reports the differences for a set of
predefined fields. The code is organised with a few helper classes so the
script remains easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Set, List

import re

import argparse
import pandas as pd
import yaml

# Regex pattern for document identifiers within STI columns
ID_PATTERN = re.compile(
    r"(?:DID[0-9]{10}|CMD[0-9]{6,}|PM[0-9]{6,}|SETC[0-9]{6,})"
)


class STIConfig:
    """Load and query the ``sti_config.yaml`` configuration."""

    def __init__(self, path: str) -> None:
        """Load ``path`` and store the raw YAML data."""

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
class STIMatrixDefinition:  # pylint: disable=too-few-public-methods
    """Definition of a single STI matrix extracted from the YAML config."""

    name: str
    file: str
    sti_sheet: str
    column_mapping: dict[str, str]
    header_row: int | None
    base_dir: Path = Path(".")

    @classmethod
    def from_config(
        cls,
        config: STIConfig,
        name: str,
    ) -> "STIMatrixDefinition":
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
            base_dir=Path(config.path).resolve().parent,
        )


class STIMatrix:  # pylint: disable=too-few-public-methods
    """Represent a STI matrix loaded from an Excel file."""

    def __init__(
        self,
        definition: STIMatrixDefinition,
        fields: Iterable[str],
    ) -> None:
        """Initialise with a matrix definition and fields of interest."""

        # store the matrix definition and ensure ``fields`` is a list
        self.definition = definition
        self.fields = list(fields)

    def load(self) -> pd.DataFrame:
        """Load the Excel file and normalise its columns."""

        file_path = Path(self.definition.file)
        if not file_path.is_absolute():
            file_path = self.definition.base_dir / file_path
        resolved = file_path.resolve()
        base = self.definition.base_dir.resolve()
        if base not in resolved.parents and resolved != base:
            raise ValueError(f"Matrix file {resolved} is outside of {base}")
        file_path = resolved
        header = (
            self.definition.header_row
            if self.definition.header_row is not None
            else 0
        )

        df = pd.read_excel(
            file_path,
            sheet_name=self.definition.sti_sheet,
            header=header,
        )
        # normalise column names using the mapping from the configuration
        df.rename(columns=self.definition.column_mapping, inplace=True)

        # ensure all expected columns exist and create raw columns for ID fields
        required = ["Reference"] + self.fields
        id_columns = {"MOP_design", "MOP_test", "CAF_Comments"}
        raw_cols: list[str] = []
        for col in required:
            if col not in df.columns:
                df[col] = pd.NA
            if col in id_columns:
                raw_col = f"{col}_raw"
                raw_cols.append(raw_col)
                if raw_col not in df.columns:
                    df[raw_col] = pd.NA

        df = df[required + raw_cols]

        # extract document identifiers from dedicated columns
        for col in id_columns.intersection(df.columns):
            raw_col = f"{col}_raw"
            df[raw_col] = df[col]
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .apply(lambda text: set(ID_PATTERN.findall(text)))
            )

        return df


class PPDChecker:  # pylint: disable=too-few-public-methods
    """Load a PPD Excel file and query its document identifiers."""

    def __init__(self, path: str, column: str = "Référence ALSTOM") -> None:
        """Store the path to the file and the column containing IDs."""

        self.path = Path(path)
        self.column = column

    def load_ids(self) -> Set[str]:
        """Return the set of IDs present in the PPD file."""

        df = pd.read_excel(self.path)
        try:
            series = df[self.column]
        except KeyError as exc:
            # provide a clear error if the column is missing
            raise KeyError(
                f"Column '{self.column}' not found in {self.path}"
            ) from exc
        return set(series.dropna().astype(str))


class STIMatrixComparator:  # pylint: disable=too-few-public-methods
    """Compare two :class:`STIMatrix` objects."""

    def __init__(self, fields: Iterable[str]) -> None:
        """Initialise with the list of fields that should be compared."""

        self.fields = list(fields)

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Return True if ``a`` and ``b`` should be considered equal."""

        if isinstance(a, set) or isinstance(b, set):
            # handle comparison of sets of document identifiers
            a_set = a if isinstance(a, set) else set()
            b_set = b if isinstance(b, set) else set()
            return a_set == b_set
        if pd.isna(a) and pd.isna(b):
            return True
        return str(a) == str(b)

    def _canonical(self, value: Any):
        """Return a normalised representation of ``value`` for comparison."""
        if isinstance(value, set):
            return tuple(sorted(value))
        if pd.isna(value):
            return ()
        return str(value)

    def compare(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame describing the differences between ``df1`` and
        ``df2``."""

        merged = pd.merge(
            df1,
            df2,
            on="Reference",
            how="outer",
            suffixes=("_1", "_2"),
        )
        canon_columns = [
            f"{field}_{suffix}" for field in self.fields for suffix in ("1", "2")
        ]
        canon = merged[canon_columns].map(self._canonical)
        # container for all mismatch rows
        diffs: List[pd.DataFrame] = []

        for field in self.fields:
            col1 = f"{field}_1"
            col2 = f"{field}_2"
            # rows where the two columns differ
            mask = canon[col1] != canon[col2]
            mism = merged[mask]
            if not mism.empty:
                subset = mism[["Reference", col1, col2]].copy()
                subset["field"] = field
                subset.rename(
                    columns={col1: "value_1", col2: "value_2"}, inplace=True
                )
                # format difference using raw text when available
                col1_raw = f"{field}_raw_1"
                col2_raw = f"{field}_raw_2"
                if col1_raw in mism.columns and col2_raw in mism.columns:
                    diff_series = (
                        mism[col1_raw].fillna("").astype(str)
                        + " -> "
                        + mism[col2_raw].fillna("").astype(str)
                    )
                else:
                    diff_series = (
                        subset["value_1"].astype(str)
                        + " -> "
                        + subset["value_2"].astype(str)
                    )
                subset["Différence"] = diff_series
                diffs.append(
                    subset[["Reference", "field", "value_1", "value_2", "Différence"]]
                )
        if diffs:
            return pd.concat(diffs, ignore_index=True)
        return pd.DataFrame(
            columns=["Reference", "field", "value_1", "value_2", "Différence"]
        )


def collect_document_ids(df: pd.DataFrame, columns: Iterable[str]) -> Set[str]:
    """Return a set of IDs gathered from ``df`` across ``columns``."""

    ids: Set[str] = set()
    for col in columns:
        if col in df.columns:
            # gather all strings stored in sets within the given column
            ids.update(
                {
                    id_str
                    for item in df[col]
                    if isinstance(item, set)
                    for id_str in item
                }
            )
    return ids


def summarise_differences(diffs: pd.DataFrame) -> pd.DataFrame:
    """Return an aggregated view of ``diffs``.

    The returned DataFrame has one row per unique combination of field and
    differing values. Each row lists all references concerned and the number of
    occurrences. Values that are ``pd.NA`` or empty sets are treated as missing
    when computing the difference state.
    """

    def _is_missing(value: Any) -> bool:
        return (
            value is pd.NA
            or value is None
            or (isinstance(value, float) and pd.isna(value))
            or (isinstance(value, set) and len(value) == 0)
            or value == ""
        )

    def _state(v1: Any, v2: Any) -> str:
        if _is_missing(v1) and not _is_missing(v2):
            return "Absent dans GE"
        if not _is_missing(v1) and _is_missing(v2):
            return "Absent dans H2"
        return "Différents"

    def _format_side(label: str, value: Any) -> List[str]:
        lines = [f"{label} :"]
        if _is_missing(value):
            return lines
        items = sorted(value) if isinstance(value, set) else [value]
        lines.extend(f"  - {item}" for item in items)
        return lines

    def _format_diff(v1: Any, v2: Any) -> str:
        return "\n".join(_format_side("GE", v1) + _format_side("H2", v2))

    def _canon(value: Any) -> Any:
        if isinstance(value, set):
            return tuple(sorted(value))
        if _is_missing(value):
            return ()  # pragma: no cover - simple data normalisation
        return value

    canon = diffs.assign(_v1=diffs["value_1"].map(_canon), _v2=diffs["value_2"].map(_canon))

    rows = []
    grouped = canon.groupby(["field", "_v1", "_v2"], dropna=False)
    for (field, _v1_key, _v2_key), group in grouped:
        v1 = group["value_1"].iloc[0]
        v2 = group["value_2"].iloc[0]
        rows.append(
            {
                "Champ": field,
                "État": _state(v1, v2),
                "Différence": _format_diff(v1, v2),
                "Reference": tuple(sorted(group["Reference"])),
                "nb_references": len(group),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for the command line interface."""

    parser = argparse.ArgumentParser(
        description="Compare two STI matrices defined in sti_config.yaml"
    )
    parser.add_argument(
        "matrix1",
        help="Name of the first matrix in the config",
    )
    parser.add_argument(
        "matrix2",
        help="Name of the second matrix in the config",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="sti_config.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional Excel file to store the differences",
    )
    parser.add_argument(
        "--ppd",
        help="Optional PPD Excel file to verify documents",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display a grouped summary of the differences",
    )

    # parse command line arguments
    args = parser.parse_args()

    # load configuration and extract fields to compare
    config = STIConfig(args.config)
    fields = config.fields_to_compare

    matrix1_def = STIMatrixDefinition.from_config(config, args.matrix1)
    matrix2_def = STIMatrixDefinition.from_config(config, args.matrix2)

    # load both matrices using the extracted definitions
    df1 = STIMatrix(matrix1_def, fields).load()
    df2 = STIMatrix(matrix2_def, fields).load()

    comparator = STIMatrixComparator(fields)
    diffs = comparator.compare(df1, df2)

    result_df = diffs
    if args.summary:
        result_df = summarize_diffs(diffs, args.matrix1, args.matrix2)

    if result_df.empty:
        print("No differences found")
    else:
        if args.output:
            output_path = Path(args.output)
            # write results to an optional Excel file

            try:
                result_df.to_excel(output_path, index=False)
            except ValueError as exc:
                if "sheet is too large" in str(exc):
                    output_path = output_path.with_suffix(".csv")
                    result_df.to_csv(output_path, index=False)
                else:
                    raise  # pragma: no cover
            print(f"Differences written to {output_path}")
        else:
          print(diffs.to_string(index=False))  # pragma: no cover


    if args.ppd:
        # optionally verify that all MOP documents exist in the PPD file
        checker = PPDChecker(args.ppd)
        ppd_ids = checker.load_ids()
        all_ids = collect_document_ids(
            pd.concat([df1, df2], ignore_index=True),
            ["MOP_design", "MOP_test", "CAF_Comments"],
        )
        # any referenced IDs not found in the PPD are reported
        missing = sorted(all_ids - ppd_ids)
        if missing:
            print("Missing in PPD:", ", ".join(missing))
        else:
            print("All referenced MOP documents are present in the PPD file")


if __name__ == "__main__":
    main()
