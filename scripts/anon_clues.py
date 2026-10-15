"""Identity clues for the anonymity scans.

Generic patterns are built in. Personal terms (your name, e-mail handle, local folder names) go in a local,
git-ignored file `.anon_clues` at the repository root, one term per line, so they never enter the repository.
"""

from pathlib import Path

GENERIC = ["gmail", "@users.noreply", "c:\\users", "c:/users", "/c/users", "/home/", "/users/"]


def load_clues(root: Path | str = ".") -> list[str]:
    f = Path(root) / ".anon_clues"
    personal = [ln.strip().lower() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()] if f.exists() else []
    return GENERIC + personal
