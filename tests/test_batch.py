import csv
import json

import pytest

from arxiv_reproducer.batch import (
    COLUMNS,
    BatchRow,
    read_id_file,
    summarize,
    write_summary_csv,
    write_summary_md,
)


def rows_for_tests():
    return [
        BatchRow(
            arxiv_id="2301.11111",
            title="A Good Paper",
            status="completed",
            verdict="REPRODUCED",
            confidence=90,
            target_result="Tc = 2.269",
            iterations=12,
            wall_clock_seconds=300.0,
            estimated_cost_usd=2.5,
            report="runs/2301.11111/x/REPORT.md",
        ),
        BatchRow(
            arxiv_id="2301.22222",
            status="fetch_error",
            error="arXiv returned HTTP 404",
        ),
        BatchRow(
            arxiv_id="2301.33333",
            title="Pipes | and | newlines",
            status="completed",
            verdict="NOT REPRODUCED",
            confidence=60,
            target_result="a | b\nc",
            estimated_cost_usd=1.25,
        ),
    ]


class TestReadIdFile:
    def test_skips_blanks_and_comments(self, tmp_path):
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text(
            "# physics batch\n"
            "2301.11111\n"
            "\n"
            "2301.22222  # the ising one\n"
            "hep-th/9901001\n"
        )
        assert read_id_file(ids_file) == ["2301.11111", "2301.22222", "hep-th/9901001"]

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="could not read batch file"):
            read_id_file(tmp_path / "nope.txt")

    def test_empty_file_is_a_clear_error(self, tmp_path):
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("# only comments\n\n")
        with pytest.raises(ValueError, match="no arXiv IDs"):
            read_id_file(ids_file)


class TestFromManifest:
    def test_reads_run_json_fields(self, tmp_path):
        (tmp_path / "REPORT.md").write_text("# report")
        manifest = tmp_path / "run.json"
        manifest.write_text(
            json.dumps(
                {
                    "arxiv_id": "2301.11111",
                    "title": "T",
                    "status": "completed",
                    "verdict": "REPRODUCED",
                    "confidence": 85,
                    "target_result": "Tc",
                    "iterations": 7,
                    "wall_clock_seconds": 42.5,
                    "estimated_cost_usd": 1.5,
                    "error": None,
                }
            )
        )
        row = BatchRow.from_manifest(manifest)
        assert row.arxiv_id == "2301.11111"
        assert row.verdict == "REPRODUCED"
        assert row.confidence == 85
        assert row.estimated_cost_usd == 1.5
        assert row.report == str(tmp_path / "REPORT.md")

    def test_missing_report_leaves_report_none(self, tmp_path):
        manifest = tmp_path / "run.json"
        manifest.write_text(json.dumps({"arxiv_id": "x", "status": "error"}))
        row = BatchRow.from_manifest(manifest)
        assert row.report is None
        assert row.status == "error"


class TestSummaryFiles:
    def test_csv_roundtrips_all_columns(self, tmp_path):
        path = tmp_path / "summary.csv"
        write_summary_csv(rows_for_tests(), path)
        with path.open(newline="") as handle:
            parsed = list(csv.DictReader(handle))
        assert [row["arxiv_id"] for row in parsed] == ["2301.11111", "2301.22222", "2301.33333"]
        assert list(parsed[0].keys()) == COLUMNS
        assert parsed[0]["verdict"] == "REPRODUCED"
        assert parsed[0]["confidence"] == "90"
        assert parsed[1]["verdict"] == ""  # None → empty cell
        assert parsed[1]["error"] == "arXiv returned HTTP 404"

    def test_markdown_table_escapes_pipes_and_newlines(self, tmp_path):
        path = tmp_path / "summary.md"
        write_summary_md(rows_for_tests(), path)
        text = path.read_text()
        lines = text.splitlines()
        assert lines[0].startswith("| arXiv ID | Verdict | Confidence |")
        assert len(lines) == 2 + 3  # header + separator + one row per paper
        assert "a \\| b c" in text  # pipe escaped, newline flattened
        assert "90%" in text
        assert "—" in text  # missing values rendered as em dash


class TestSummarize:
    def test_counts_and_total_cost(self):
        line = summarize(rows_for_tests())
        assert line.startswith("3 papers")
        assert "1 REPRODUCED" in line
        assert "1 NOT REPRODUCED" in line
        assert "1 without a verdict" in line
        assert "$3.75" in line

    def test_singular_paper(self):
        line = summarize([BatchRow(arxiv_id="x")])
        assert line.startswith("1 paper ·")
