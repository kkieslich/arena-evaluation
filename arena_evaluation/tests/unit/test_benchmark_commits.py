"""The commits a run records in its manifest: Arena's checkout and the packages it depends on (`state.capture_commits`)."""

from __future__ import annotations

import logging
import pathlib
import subprocess

import pytest
import yaml

from arena_evaluation.benchmark import state
from arena_evaluation.benchmark.state import RUN_PACKAGES, Manifest, RunDir, capture_commits

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "protocol.file.allow=always", "-c", "init.defaultBranch=main"]


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo(path: pathlib.Path, files: dict[str, str]) -> str:
    """A git repo at `path` with `files` in one commit; its commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> dict[str, object]:
    """An Arena checkout with humansim and arena_evaluation as submodules, as the container's /opt/arena_ws/src/Arena."""
    _repo(tmp_path / "origin" / "humansim", {"arena_humansim/arena_humansim/__init__.py": ""})
    _repo(tmp_path / "origin" / "arena_evaluation", {"arena_evaluation/arena_evaluation/__init__.py": ""})
    arena = (tmp_path / "ws" / "src" / "Arena").resolve()
    _repo(arena, {"README.md": "arena\n"})
    for sub in ("humansim", "arena_evaluation"):
        _git(arena, "submodule", "add", "-q", str(tmp_path / "origin" / sub), sub)
    _git(arena, "commit", "-q", "-m", "pins")
    origins = {"arena_humansim": arena / "humansim/arena_humansim/arena_humansim/__init__.py", "arena_evaluation": arena / "arena_evaluation/arena_evaluation/arena_evaluation/__init__.py"}
    return {"arena": arena, "install": tmp_path / "ws" / "install" / "arena_evaluation" / "share" / "arena_evaluation", "origins": origins}


def test_a_run_records_arena_and_the_submodules_it_runs(workspace: dict) -> None:
    arena: pathlib.Path = workspace["arena"]
    found = capture_commits(pathlib.Path("/nonexistent"), workspace["origins"].get)
    assert set(found) == {"arena", *(name for name, _, _ in RUN_PACKAGES)}
    assert found["arena"] == {"path": str(arena), "commit": _git(arena, "rev-parse", "HEAD"), "dirty": False}
    for sub in ("humansim", "arena_evaluation"):
        head = _git(arena / sub, "rev-parse", "HEAD")
        assert found[sub] == {"path": str(arena / sub), "commit": head, "dirty": False, "pinned": head}
    assert yaml.safe_load(yaml.safe_dump(found)) == found


def test_a_moved_or_edited_submodule_shows_against_its_pin(workspace: dict) -> None:
    arena: pathlib.Path = workspace["arena"]
    pin = _git(arena / "humansim", "rev-parse", "HEAD")
    (arena / "humansim" / "fix.py").write_text("x = 1\n")
    _git(arena / "humansim", "add", "fix.py")
    _git(arena / "humansim", "commit", "-q", "-m", "fix")
    (arena / "humansim" / "wip.py").write_text("y = 2\n")
    found = capture_commits(pathlib.Path("/nonexistent"), workspace["origins"].get)
    assert found["humansim"]["commit"] == _git(arena / "humansim", "rev-parse", "HEAD") != pin
    assert found["humansim"]["pinned"] == pin and found["humansim"]["dirty"] is True
    assert found["arena"]["dirty"] is True


def test_arena_is_found_from_the_install_when_arena_evaluation_runs_from_elsewhere(workspace: dict, tmp_path: pathlib.Path) -> None:
    arena: pathlib.Path = workspace["arena"]
    loose = tmp_path.resolve() / "elsewhere" / "arena_evaluation" / "__init__.py"
    loose.parent.mkdir(parents=True)
    loose.touch()
    origins = {**workspace["origins"], "arena_evaluation": loose}
    found = capture_commits(workspace["install"], origins.get)
    assert found["arena"]["path"] == str(arena) and found["arena"]["commit"] == _git(arena, "rev-parse", "HEAD")
    # the module lies in no checkout: the Arena submodule's checkout stands in, with its pin
    assert found["arena_evaluation"]["path"] == str(arena / "arena_evaluation")
    assert found["arena_evaluation"]["pinned"] == found["arena_evaluation"]["commit"]


def test_what_cannot_be_read_is_null_with_a_reason(tmp_path: pathlib.Path) -> None:
    tmp_path = tmp_path.resolve()
    loose = tmp_path / "site" / "arena_humansim" / "__init__.py"
    loose.parent.mkdir(parents=True)
    loose.touch()
    found = capture_commits(tmp_path / "install", {"arena_humansim": loose}.get)
    for name in found:
        assert found[name]["commit"] is None and found[name]["dirty"] is None and found[name]["path"] is None
        assert found[name]["reason"]
    assert "no git checkout above" in found["arena"]["reason"]
    assert f"module arena_humansim at {loose}" in found["humansim"]["reason"]
    assert "module arena_evaluation not installed" in found["arena_evaluation"]["reason"]
    # a checkout git cannot read (a worktree whose git dir is not mounted)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: /not/mounted/.git/worktrees/broken\n")
    (broken / "arena_humansim").mkdir()
    (broken / "arena_humansim" / "__init__.py").touch()
    found = capture_commits(tmp_path / "install", {"arena_humansim": broken / "arena_humansim" / "__init__.py"}.get)
    assert found["humansim"] == {"path": str(broken), "commit": None, "dirty": None, "reason": f"git rev-parse HEAD failed in {broken}"}


def _manifest(**overrides: object) -> Manifest:
    base: dict[str, object] = {
        "run_id": "shared-run",
        "created_at": "2026-11-10T10:00:00+00:00",
        "arena_git_sha": "abc123",
        "arena_git_dirty": False,
        "cli_args": [],
        "env_n": 1,
        "headless": True,
        "config_hash": "0" * 40,
        "simulator": "flatsim",
        "scale_episodes": 1.0,
        "suite_name": "basic",
        "contest_name": "basic",
        "suite": {"stages": [{"name": "s1", "episodes": 1}]},
        "contest": [{"name": "c1"}],
        "steps": [{"key": "c1/s1", "episodes_planned": 1}],
    }
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


COMMITS = {"arena": {"path": "/a", "commit": "a" * 40, "dirty": False}, "humansim": {"path": "/a/humansim", "commit": "b" * 40, "dirty": False, "pinned": "b" * 40}}


def test_the_manifest_keeps_the_commits_and_reads_older_manifests_without_them() -> None:
    manifest = _manifest(commits=COMMITS)
    assert Manifest.from_yaml(manifest.to_yaml()).commits == COMMITS
    assert yaml.safe_load(manifest.to_yaml())["commits"]["humansim"]["commit"] == "b" * 40
    old = yaml.safe_load(_manifest().to_yaml())
    del old["commits"]
    assert Manifest.from_yaml(yaml.safe_dump(old)).commits is None


class _Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_a_lane_running_other_commits_is_logged(tmp_path: pathlib.Path) -> None:
    log = logging.getLogger(state.__name__)
    handler = _Messages()
    log.addHandler(handler)
    try:
        RunDir.create(tmp_path, "shared-run", _manifest(commits=COMMITS), lane="p1")
        RunDir.create(tmp_path, "shared-run", _manifest(commits=COMMITS), lane="p2")
        assert handler.messages == []
        moved = {**COMMITS, "humansim": {**COMMITS["humansim"], "commit": "c" * 40}}
        joined = RunDir.create(tmp_path, "shared-run", _manifest(commits=moved), lane="p3")
    finally:
        log.removeHandler(handler)
    (message,) = handler.messages
    assert "lane 'p3' runs" in message and "c" * 40 in message
    assert joined.manifest.commits == COMMITS
