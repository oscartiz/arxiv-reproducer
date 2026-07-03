from datetime import datetime

from arxiv_reproducer.runs import new_run_dir, safe_dir_name


class TestSafeDirName:
    def test_old_style_ids_lose_the_slash(self):
        assert safe_dir_name("hep-th/9901001") == "hep-th_9901001"

    def test_new_style_ids_unchanged(self):
        assert safe_dir_name("2301.12345v2") == "2301.12345v2"


class TestNewRunDir:
    def test_creates_timestamped_directory(self, tmp_path):
        run_dir = new_run_dir(tmp_path, now=datetime(2026, 7, 3, 14, 22, 5))
        assert run_dir == tmp_path / "20260703-142205"
        assert run_dir.is_dir()

    def test_same_second_reruns_get_distinct_dirs(self, tmp_path):
        now = datetime(2026, 7, 3, 14, 22, 5)
        first = new_run_dir(tmp_path, now=now)
        second = new_run_dir(tmp_path, now=now)
        third = new_run_dir(tmp_path, now=now)
        assert len({first, second, third}) == 3
        assert first.is_dir() and second.is_dir() and third.is_dir()

    def test_prior_run_contents_are_never_touched(self, tmp_path):
        now = datetime(2026, 7, 3, 14, 22, 5)
        first = new_run_dir(tmp_path, now=now)
        (first / "REPORT.md").write_text("# precious prior verdict")
        second = new_run_dir(tmp_path, now=now)
        assert (first / "REPORT.md").read_text() == "# precious prior verdict"
        assert not (second / "REPORT.md").exists()

    def test_latest_symlink_tracks_newest_run(self, tmp_path):
        new_run_dir(tmp_path, now=datetime(2026, 7, 3, 10, 0, 0))
        second = new_run_dir(tmp_path, now=datetime(2026, 7, 3, 11, 0, 0))
        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == second
