import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from expr_tracker import _native


@pytest.fixture
def viewer(tmp_path):
    log = tmp_path / "project" / "run"
    log.mkdir(parents=True)
    (log / "metrics.jsonl").write_text(
        '{"_step":0,"_time":1,"loss":1,"huge":9007199254740993}\n'
        '{"_step":1,"_time":2,"loss":NaN}\n'
    )
    parts = log / "metrics.jsonl.parts"
    parts.mkdir()
    (parts / "00000001.jsonl").write_text(
        '{"_step":2,"_time":3,"loss":0.5}\n'
        '{"_step":0,"_time":4,"accuracy":0.9}\n'
    )
    (log / "config.json").write_text('{"lr":0.001}')
    (log / "spans.jsonl").write_text(
        '{"_step":0,"name":"first","start":1,"dur_ms":2}\n'
        '{"_step":0,"name":"second","start":2,"dur_ms":3}\n'
        '{"_step":2,"name":"third","start":3,"dur_ms":4}\n'
    )
    assets = Path(__file__).parents[1] / "src" / "expr_tracker" / "_web"
    assert (assets / "index.html").is_file(), "Build web assets before viewer tests"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from expr_tracker import _native;"
            "_native.serve(Path(sys.argv[1]),Path(sys.argv[2]),'127.0.0.1',int(sys.argv[3]))",
            str(tmp_path),
            str(assets),
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Viewer exited: {stdout.decode()}\n{stderr.decode()}")
            try:
                with urlopen(base + "/api/runs", timeout=1) as response:
                    runs = json.load(response)
                break
            except URLError:
                if time.monotonic() >= deadline:
                    pytest.fail("Viewer did not become responsive")
                time.sleep(0.05)
        yield base, runs[0]["id"], log
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)


def fetch(base, endpoint):
    with urlopen(base + endpoint, timeout=10) as response:
        return json.load(response)


def test_packaged_native_server_serves_assets_and_segmented_history(viewer):
    base, run, _ = viewer
    with urlopen(base, timeout=10) as response:
        assert "Experiment tracker" in response.read().decode()
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    detail = fetch(base, f"/api/run?run={run}")
    assert detail["rows"] == 3
    assert detail["metrics"] == ["accuracy", "huge", "loss"]
    assert detail["config"]["config.json"]["lr"] == 0.001
    history = fetch(
        base, f"/api/history?run={run}&metrics=%5B%22loss%22%5D&max_points=64"
    )
    assert history["x"] == [0, 1, 2]
    assert history["series"][0]["values"] == [1, None, 0.5]
    assert history["series"][0]["nonfinite"] == 1
    rows = fetch(base, f"/api/records?run={run}")["records"]
    patched = next(row for row in rows if row["_step"] == 0)
    assert patched["loss"] == 1
    assert patched["accuracy"] == 0.9
    assert patched["huge"] == {"$et.integer": "9007199254740993"}


def test_viewer_follows_append_and_rotation_without_reading_torn_rows(viewer):
    base, run, log = viewer
    fetch(base, f"/api/run?run={run}")
    active = log / "metrics.jsonl.parts" / "00000001.jsonl"
    with active.open("a") as output:
        output.write('{"_step":3,"_time":5,"loss":0.25')
    assert fetch(base, f"/api/run?run={run}")["rows"] == 3
    with active.open("a") as output:
        output.write("}\n")
    assert fetch(base, f"/api/run?run={run}")["rows"] == 4
    (active.parent / "00000002.jsonl").write_text(
        '{"_step":4,"_time":6,"loss":0.125}\n'
    )
    assert fetch(base, f"/api/run?run={run}")["rows"] == 5
    active.write_text('{"_step":2,"_time":3,"loss":0.7}\n')
    assert fetch(base, f"/api/run?run={run}")["rows"] == 4


def test_record_pagination_and_explicit_source_selection(viewer):
    base, run, _ = viewer
    first = fetch(base, f"/api/records?run={run}&limit=1")
    assert len(first["records"]) == 1
    assert first["next"] is not None
    second = fetch(base, f"/api/records?run={run}&limit=1&before={first['next']}")
    assert first["records"][0]["_step"] != second["records"][0]["_step"]
    with pytest.raises(HTTPError) as denied:
        fetch(base, f"/api/records?run={run}&file=../../outside.jsonl")
    assert denied.value.code == 404


def test_viewer_rejects_invalid_query_bounds_and_cross_origin(viewer):
    base, run, _ = viewer
    with pytest.raises(HTTPError) as invalid:
        fetch(
            base, f"/api/history?run={run}&metrics=%5B%22loss%22%5D&from_step=2&to_step=1"
        )
    assert invalid.value.code == 400
    request = Request(base + "/api/runs", headers={"Origin": "https://outside.example"})
    with pytest.raises(HTTPError) as denied, urlopen(request, timeout=10):
        pass
    assert denied.value.code == 403


def test_native_viewer_rejects_network_binding(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        _native.serve(tmp_path, tmp_path, "0.0.0.0", 9110)


def test_span_events_are_not_merged_and_step_ranges_apply(viewer):
    base, run, _ = viewer
    rows = fetch(base, f"/api/records?run={run}&file=spans.jsonl")["records"]
    assert len(rows) == 3
    ranged = fetch(
        base,
        f"/api/records?run={run}&file=spans.jsonl&from_step=0&to_step=1",
    )["records"]
    assert {row["name"] for row in ranged} == {"first", "second"}


def test_server_directory_browser_lists_only_directories_and_canonicalizes(viewer):
    base, _, log = viewer
    root = log.parents[1]
    (root / ".hidden").mkdir()
    (root / "ordinary.txt").write_text("not a directory")
    listing = fetch(base, "/api/directories")
    assert listing["path"] == str(root.resolve())
    assert listing["parent"] == str(root.parent.resolve())
    assert [entry["name"] for entry in listing["directories"]] == [".hidden", "project"]
    nested = fetch(base, "/api/directories?" + urlencode({"path": "project/../project/run"}))
    assert nested["path"] == str(log.resolve())
    assert fetch(base, "/api/root")["path"] == str(root.resolve())


def test_request_scoped_roots_do_not_conflict_across_concurrent_pages(viewer, tmp_path_factory):
    base, original, log = viewer
    other = tmp_path_factory.mktemp("other-viewer-root")
    # Match the relative name in the original root to catch run-ID collisions.
    other_run = other / "project" / "run"
    other_run.mkdir(parents=True)
    (other_run / "metrics.jsonl").write_text('{"_step":9,"_time":10,"other":42}\n')
    (other_run / "config.json").write_text('{"other":true}')
    root_query = urlencode({"root": str(other)})
    remote = fetch(base, "/api/runs?" + root_query)[0]
    assert remote["id"] != original
    assert fetch(base, "/api/directories?" + urlencode({"path": str(other)}))["path"] == str(other)

    def query(at):
        if at % 2:
            args = {"root": str(other), "run": remote["id"]}
            expected = "other"
        else:
            args = {"run": original}
            expected = "loss"
        assert expected in fetch(base, "/api/run?" + urlencode(args))["metrics"]
        assert fetch(base, "/api/records?" + urlencode(args))["records"]
        history = fetch(base, "/api/history?" + urlencode({**args, "metrics": json.dumps([expected])}))
        assert history["series"][0]["metric"] == expected
        fetch(base, "/api/runs" + ("?" + root_query if at % 2 else ""))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(query, range(32)))
    assert fetch(base, "/api/root")["path"] == str(log.parents[1])
    with pytest.raises(HTTPError) as wrong_scope:
        fetch(base, "/api/run?" + urlencode({"run": remote["id"]}))
    assert wrong_scope.value.code == 404


@pytest.mark.parametrize("endpoint,param", [("/api/directories", "path"), ("/api/root", "root"), ("/api/runs", "root")])
def test_server_directory_errors_are_explicit(viewer, endpoint, param):
    base, _, log = viewer
    for path, code in [("", 400), (str(log / "absent"), 404), (str(log / "metrics.jsonl"), 400)]:
        with pytest.raises(HTTPError) as error:
            fetch(base, endpoint + "?" + urlencode({param: path}))
        assert error.value.code == code
        assert json.load(error.value)["error"]


@pytest.mark.parametrize("endpoint", ["/api/directories", "/api/root", "/api/runs"])
@pytest.mark.parametrize("headers", [{"Host": "untrusted.example"}, {"Origin": "https://untrusted.example"}])
def test_directory_endpoints_preserve_host_and_origin_guards(viewer, endpoint, headers):
    base, _, _ = viewer
    with pytest.raises(HTTPError) as denied, urlopen(Request(base + endpoint, headers=headers), timeout=10):
        pass
    assert denied.value.code == 403


def test_directory_aliases_and_escaped_segments(viewer, tmp_path_factory):
    base, run, log = viewer
    outside = tmp_path_factory.mktemp("viewer-segment-outside")
    alias = log.parents[1] / "alias"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable")
    listing = fetch(base, "/api/directories")
    assert "alias" in [entry["name"] for entry in listing["directories"]]
    assert fetch(base, "/api/directories?" + urlencode({"path": str(alias)}))["path"] == str(outside)
    segment = outside / "segment.jsonl"
    segment.write_text('{"_step":5,"outside":true}\n')
    escaped = log / "metrics.jsonl.parts" / "00000002.jsonl"
    escaped.symlink_to(segment)
    with pytest.raises(HTTPError) as denied:
        fetch(base, "/api/run?" + urlencode({"run": run}))
    assert denied.value.code == 403


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_directory_permission_denied_is_not_an_empty_listing(viewer):
    if os.geteuid() == 0:
        pytest.skip("Root bypasses directory permission checks")
    base, _, log = viewer
    denied_path = log.parents[1] / "unreadable"
    denied_path.mkdir()
    denied_path.chmod(0)
    try:
        with pytest.raises(HTTPError) as error:
            fetch(base, "/api/directories?" + urlencode({"path": str(denied_path)}))
        assert error.value.code == 403
        assert "Cannot access server directory" in json.load(error.value)["error"]
    finally:
        denied_path.chmod(0o700)
