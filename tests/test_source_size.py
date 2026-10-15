"""Keep handwritten source, regression suites and benchmarks readable."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src", "rust", "tests", "benchmarks", "web/src", "web/e2e")
SOURCE_SUFFIXES = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".sh"}
MAX_SOURCE_LINES = 800


def test_authored_source_files_stay_within_line_limit():
    oversized = []
    for directory in SOURCE_ROOTS:
        for path in sorted((ROOT / directory).rglob("*")):
            if (
                not path.is_file()
                or path.suffix not in SOURCE_SUFFIXES
                or "_web" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            lines = len(path.read_text(encoding="utf-8").splitlines())
            if lines > MAX_SOURCE_LINES:
                oversized.append(f"{path.relative_to(ROOT)}: {lines} lines")
    assert not oversized, (
        f"Handwritten source files must not exceed {MAX_SOURCE_LINES} physical "
        "lines, including comments and blanks:\n" + "\n".join(oversized)
    )
