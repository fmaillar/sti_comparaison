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
    assert set(df.columns) == {
        "Reference",
        "Requirement",
        "MOP_design",
        "MOP_design_raw",
    }
    assert df.loc[0, "MOP_design"] == {"DID1234567890"}
    assert df.loc[0, "MOP_design_raw"] == "blah DID1234567890"


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
    assert pd.isna(df.loc[0, "MOP_design_raw"])


def test_sti_matrix_load_relative_path(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    cfg_dict = yaml.safe_load(path.read_text())
    cfg_dict["matrices"][0]["file"] = "sub/a.xlsx"
    path.write_text(yaml.dump(cfg_dict))
    cfg = cs.STIConfig(str(path))
    defn = cs.STIMatrixDefinition.from_config(cfg, "A")

    expected = defn.base_dir / "sub/a.xlsx"

    def fake_read_excel(pth, *args, **kwargs):
        assert Path(pth) == expected.resolve()
        return pd.DataFrame({"A": [1], "B": ["req"], "C": [""]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    matrix = cs.STIMatrix(defn, cfg.fields_to_compare)
    matrix.load()


def test_sti_matrix_load_outside_base_dir(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    cfg_dict = yaml.safe_load(path.read_text())
    cfg_dict["matrices"][0]["file"] = str(tmp_path.parent / "evil.xlsx")
    path.write_text(yaml.dump(cfg_dict))
    cfg = cs.STIConfig(str(path))
    defn = cs.STIMatrixDefinition.from_config(cfg, "A")
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame())
    matrix = cs.STIMatrix(defn, cfg.fields_to_compare)
    with pytest.raises(ValueError):
        matrix.load()


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


def test_compare_raw_difference():
    cmp = cs.STIMatrixComparator(["MOP_design"])
    df1 = pd.DataFrame({
        "Reference": [1],
        "MOP_design": [{"ID1"}],
        "MOP_design_raw": ["a ID1"],
    })
    df2 = pd.DataFrame({
        "Reference": [1],
        "MOP_design": [{"ID2"}],
        "MOP_design_raw": ["b ID2"],
    })
    diffs = cmp.compare(df1, df2)
    assert diffs.loc[0, "Différence"] == "a ID1 -> b ID2"


def test_compare_no_difference():
    cmp = cs.STIMatrixComparator(["Requirement"])
    df = pd.DataFrame({"Reference": [1], "Requirement": ["A"]})
    result = cmp.compare(df, df.copy())
    assert result.empty


def test_collect_document_ids():
    df = pd.DataFrame({"A": [{"X"}, {"Y"}]})
    ids = cs.collect_document_ids(df, ["A"])
    assert ids == {"X", "Y"}


def test_main_print_diffs(monkeypatch, tmp_path):
    path = make_config(tmp_path)

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=None,
        )

    diff_df = pd.DataFrame(
        {"Reference": [1], "field": ["Requirement"], "value_1": ["A"], "value_2": ["B"]}
    )

    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: pd.DataFrame())
    monkeypatch.setattr(cs.STIMatrixComparator, "compare", lambda self, a, b: diff_df)

    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert any("Reference" in line for line in out)


def test_main(monkeypatch, tmp_path):
    path = make_config(tmp_path)

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=None,
            summary=False,
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
            summary=False,
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


def test_main_output_too_large(monkeypatch, tmp_path):
    path = make_config(tmp_path)
    output = tmp_path / "out.xlsx"

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=str(output),
            ppd=None,
            summary=False,
        )

    # ignore file loading
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        cs.STIMatrix,
        "load",
        lambda self: pd.DataFrame({"Reference": [1], "Requirement": ["A"]}),
    )
    monkeypatch.setattr(
        cs.STIMatrixComparator,
        "compare",
        lambda self, a, b: pd.DataFrame(
            {"Reference": [1], "field": ["Requirement"], "value_1": ["A"], "value_2": ["B"]}
        ),
    )

    def fake_to_excel(self, *a, **k):
        raise ValueError("This sheet is too large!")

    path_holder = {}

    def fake_to_csv(self, path, index=False):
        path_holder["path"] = path

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))

    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert path_holder["path"] == output.with_suffix(".csv")
    assert any("Differences written" in line for line in out)


def test_main_prints_differences(monkeypatch, tmp_path):
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
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: pd.DataFrame({"Reference": [1]}))
    monkeypatch.setattr(
        cs.STIMatrixComparator,
        "compare",
        lambda self, a, b: pd.DataFrame({
            "Reference": [1],
            "field": ["Requirement"],
            "value_1": ["A"],
            "value_2": ["B"],
            "Différence": ["A -> B"],
        }),
    )

    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert any("A -> B" in line for line in out)


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
            summary=False,
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
            summary=False,
        )
    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pd.DataFrame, "to_excel", lambda *a, **k: None)
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    import runpy
    runpy.run_module("compare_sti", run_name="__main__")


def test_summarize_diffs():
    diffs = pd.DataFrame(
        {
            "Reference": [1, 2, 3],
            "field": ["Requirement", "Requirement", "MOP_design"],
            "value_1": ["A", pd.NA, {"X"}],
            "value_2": ["B", "B", pd.NA],
        }
    )
    summary = cs.summarize_diffs(diffs, "A", "B")
    assert len(summary) == 3
    assert set(summary["Etat"]) == {"Différents", "Absent dans A", "Absent dans B"}


def test_summarize_diffs_empty():
    df = pd.DataFrame(columns=["Reference", "field", "value_1", "value_2"])
    result = cs.summarize_diffs(df, "A", "B")
    assert result.empty


def test_main_summary(monkeypatch, tmp_path):
    path = make_config(tmp_path)

    def fake_parse_args():
        return SimpleNamespace(
            matrix1="A",
            matrix2="B",
            config=str(path),
            output=None,
            ppd=None,
            summary=True,
        )

    monkeypatch.setattr(cs.argparse.ArgumentParser, "parse_args", staticmethod(fake_parse_args))
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cs.STIMatrix, "load", lambda self: pd.DataFrame({"Reference": [1], "Requirement": ["A"], "MOP_design": [set()] }))
    monkeypatch.setattr(cs.STIMatrixComparator, "compare", lambda self, a, b: pd.DataFrame({"Reference": [1], "field": ["Requirement"], "value_1": ["A"], "value_2": ["B"]}))
    out = []
    monkeypatch.setattr(builtins, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    cs.main()
    assert any("Différents" in line for line in out)
