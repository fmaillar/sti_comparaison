"""Tests for :mod:`compare_sti`."""

# pylint: disable=missing-module-docstring,wrong-import-order,wrong-import-position,
# pylint: disable=import-outside-toplevel,missing-function-docstring,unused-argument,
# pylint: disable=protected-access,line-too-long,trailing-newlines

import builtins
from pathlib import Path
from types import SimpleNamespace

import yaml

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_sti as cs


def make_config(tmp_path):
    """Create a minimal YAML configuration for testing."""
    cfg = {
        "fields_to_compare": ["Requirement", "MOP_design"],
        "matrices": [
            {
                "name": "A",
                "file": str(tmp_path / "a.xlsx"),
                "sti_sheet": "Sheet1",
                "column_mapping": {"A": "Reference", "B": "Requirement", "C": "MOP_design"},
                "sheets": {"Sheet1": {"header_row": 0}},
            },
            {
                "name": "B",
                "file": str(tmp_path / "b.xlsx"),
                "sti_sheet": "Sheet1",
                "column_mapping": {"A": "Reference", "B": "Requirement", "C": "MOP_design"},
                "sheets": {"Sheet1": {"header_row": 0}},
            },
        ],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def test_sti_config(tmp_path):
    path = make_config(tmp_path)
    cfg = cs.STIConfig(str(path))
    assert cfg.fields_to_compare == ["Requirement", "MOP_design"]
    assert cfg.matrix_definition("A")["file"].endswith("a.xlsx")
    with pytest.raises(KeyError):
        cfg.matrix_definition("missing")


def test_sti_matrix_load(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    cfg = cs.STIConfig(str(path))
    defn = cs.STIMatrixDefinition.from_config(cfg, "A")

    def fake_read_excel(*args, **kwargs):
        assert Path(args[0]) == Path(defn.file)
        assert kwargs["sheet_name"] == "Sheet1"
        data = {
            "A": [1],
            "B": ["req"],
            "C": ["blah DID1234567890"],
        }
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    matrix = cs.STIMatrix(defn, cfg.fields_to_compare)
    df = matrix.load()
    assert set(df.columns) == {"Reference", "Requirement", "MOP_design"}
    assert df.loc[0, "MOP_design"] == {"DID1234567890"}


def test_sti_matrix_load_missing_column(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    cfg = cs.STIConfig(str(path))
    defn = cs.STIMatrixDefinition.from_config(cfg, "A")

    def fake_read_excel(*args, **kwargs):
        data = {"A": [1], "B": ["req"]}
        return pd.DataFrame(data)

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    matrix = cs.STIMatrix(defn, cfg.fields_to_compare)
    df = matrix.load()
    # Missing column should be filled with NA
    assert df.loc[0, "MOP_design"] == set()


def test_ppd_checker(monkeypatch, tmp_path):
    ppd_file = tmp_path / "ppd.xlsx"
    def fake_read_excel(path):
        assert Path(path) == ppd_file
        return pd.DataFrame({"Référence ALSTOM": ["A", None, "B"]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    checker = cs.PPDChecker(str(ppd_file))
    assert checker.load_ids() == {"A", "B"}

    def fake_read_excel_fail(path):
        return pd.DataFrame({})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel_fail)
    with pytest.raises(KeyError):
        checker.load_ids()


def test_values_equal():
    cmp = cs.STIMatrixComparator([])
    assert cmp._values_equal({"a"}, {"a"})
    assert not cmp._values_equal({"a"}, None)
    assert cmp._values_equal(pd.NA, pd.NA)
    assert cmp._values_equal(1, "1")


def test_compare():
    cmp = cs.STIMatrixComparator(["Requirement", "MOP_design"])
    df1 = pd.DataFrame(
        {
            "Reference": [1, 2],
            "Requirement": ["A", "B"],
            "MOP_design": [{"ID1"}, set()],
        }
    )
    df2 = pd.DataFrame(
        {
            "Reference": [1, 3],
            "Requirement": ["A", "C"],
            "MOP_design": [{"ID1"}, {"ID2"}],
        }
    )
    diffs = cmp.compare(df1, df2)
    assert len(diffs) == 3


def test_compare_no_difference():
    cmp = cs.STIMatrixComparator(["Requirement"])
    df = pd.DataFrame({"Reference": [1], "Requirement": ["A"]})
    result = cmp.compare(df, df.copy())
    assert result.empty


def test_collect_document_ids():
    df = pd.DataFrame({"A": [{"X"}, {"Y"}]})
    ids = cs.collect_document_ids(df, ["A"])
    assert ids == {"X", "Y"}


def test_main(monkeypatch, tmp_path):
    path = make_config(tmp_path)

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=None,
        )

    fake_df = pd.DataFrame({})

    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: fake_df)
    monkeypatch.setattr(cs.STIMatrixComparator, "compare", lambda self, a, b: fake_df)

    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert "No differences found" in out[0]


def test_main_output_and_ppd(monkeypatch, tmp_path):
    path = make_config(tmp_path)
    output = tmp_path / "out.xlsx"
    ids_df = pd.DataFrame({"Référence ALSTOM": ["ID1"]})

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=str(output),
            ppd=str(tmp_path / "ppd.xlsx"),
        )

    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: ids_df)
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: pd.DataFrame({"Reference": [1], "Requirement": ["A"], "MOP_design": [set()] }))
    monkeypatch.setattr(cs.STIMatrixComparator, "compare", lambda self, a, b: pd.DataFrame({"Reference": [1], "field": ["Requirement"], "value_1": ["A"], "value_2": ["B"]}))
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda self, *a, **k: None)
    monkeypatch.setattr(cs.PPDChecker, "load_ids", lambda self: {"ID1"})

    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))

    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert any("Differences written" in line for line in out)


def test_main_missing_ppd(monkeypatch, tmp_path):
    path = make_config(tmp_path)
    ids_df = pd.DataFrame({"Référence ALSTOM": []})

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=str(tmp_path / "ppd.xlsx"),
        )

    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: ids_df)
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: pd.DataFrame({"Reference": [1], "Requirement": ["A"], "MOP_design": [{"ID1"}] }))
    monkeypatch.setattr(cs.STIMatrixComparator, "compare", lambda self, a, b: pd.DataFrame())
    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert any("Missing in PPD" in line for line in out)


def test_run_module(monkeypatch, tmp_path):
    path = make_config(tmp_path)

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=None,
        )
    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda *a, **k: None)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    import runpy
    runpy.run_module("compare_sti", run_name="__main__")
