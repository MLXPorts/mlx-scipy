import re
from pathlib import Path


_BAD_NUMPY_IMPORT = re.compile(
    r"^\s*("
    r"import\s+numpy\b|"
    r"from\s+numpy\b|"
    r"cimport\s+numpy\b|"
    r"from\s+numpy\s+cimport\b|"
    r"from\s+numpy\.[\w\.]+\s+cimport\b"
    r")",
    re.MULTILINE,
)


def _iter_source_files(root: Path):
    exts = {".py", ".pyx", ".pxd", ".pxi", ".rst"}
    for p in root.rglob("*"):
        if p.name.endswith(".pyx.in"):
            yield p
            continue
        if p.suffix in exts:
            yield p


def test_repo_has_no_numpy_import_statements():
    root = Path(__file__).resolve().parents[1]
    offenders = []

    for path in _iter_source_files(root):
        # Our local shim is expected to exist; it must *not* import external numpy.
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = _BAD_NUMPY_IMPORT.search(text)
        if not m:
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        offenders.append(f"{path}:{lineno}: {m.group(0).strip()}")

    assert offenders == []

