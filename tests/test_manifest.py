import json

import pytest

from arxiv_reproducer.manifest import (
    SCHEMA_VERSION,
    section_snippet,
    verdict_from_report,
    write_manifest,
)


class TestVerdictExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("## Verdict\n\nREPRODUCED — errors under 1%.", "REPRODUCED"),
            ("## Verdict\n\nPARTIALLY REPRODUCED: 2 of 3 values match.", "PARTIALLY REPRODUCED"),
            ("## Verdict\n\nNOT REPRODUCED. The method is underspecified.", "NOT REPRODUCED"),
            ("The agent crashed before writing a verdict.", None),
        ],
    )
    def test_extracts_the_right_verdict(self, text, expected):
        assert verdict_from_report(text) == expected

    def test_partially_is_not_mistaken_for_bare_reproduced(self):
        assert verdict_from_report("Verdict: PARTIALLY REPRODUCED") == "PARTIALLY REPRODUCED"

    def test_lowercase_prose_does_not_count(self):
        assert verdict_from_report("we have not reproduced anything") is None


class TestSectionSnippet:
    REPORT = (
        "# Report\n\n"
        "## Target Result\n\n"
        "Figure 3: critical temperature Tc = 2.269 for the 2D Ising model.\n"
        "More detail here.\n\n"
        "## Method Summary\n\nMonte Carlo.\n"
    )

    def test_extracts_first_line_of_section(self):
        snippet = section_snippet(self.REPORT, "Target Result")
        assert snippet is not None and snippet.startswith("Figure 3: critical temperature")

    def test_missing_section_returns_none(self):
        assert section_snippet(self.REPORT, "No Such Heading") is None

    def test_empty_section_returns_none(self):
        assert section_snippet("## Target Result\n\n## Next\nx\n", "Target Result") is None


class TestWriteManifest:
    def test_writes_valid_versioned_json(self, tmp_path):
        path = write_manifest(tmp_path, {"arxiv_id": "2301.12345", "verdict": None})
        assert path == tmp_path / "run.json"
        data = json.loads(path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["arxiv_id"] == "2301.12345"
        assert data["verdict"] is None
