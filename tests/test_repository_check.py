import json
import tempfile
from pathlib import Path

from sciai.common.repository_check import (
    REQUIRED_IGNORE_RULES,
    _broken_local_links,
    _candidate_json_files,
    _gitignore_errors,
)


class TestRepositoryCheck:
    def test_candidate_json_files_exclude_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracked = root / "reports" / "summary.json"
            ignored = root / "data" / "raw" / "private.json"
            tracked.parent.mkdir(parents=True)
            ignored.parent.mkdir(parents=True)
            tracked.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            ignored.write_text("not-json", encoding="utf-8")

            assert _candidate_json_files(root) == [tracked]

    def test_broken_local_links_ignore_external_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "README.md"
            target = root / "docs" / "guide.md"
            target.parent.mkdir()
            target.write_text("# Guide\n", encoding="utf-8")
            document.write_text(
                "[guide](docs/guide.md) [section](docs/guide.md#part) "
                "[external](https://example.com) [missing](docs/missing.md)\n",
                encoding="utf-8",
            )

            assert _broken_local_links(document, root) == ["docs/missing.md"]

    def test_gitignore_rejects_unscoped_runs_directory(self) -> None:
        valid_rules = "\n".join(REQUIRED_IGNORE_RULES)

        assert _gitignore_errors(valid_rules) == []
        assert "Overbroad .gitignore rule: runs/" in _gitignore_errors(
            f"{valid_rules}\nruns/\n"
        )
