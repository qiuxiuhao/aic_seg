"""Backward-compatible v3 submission entry point."""

from tools.infer_submission import main


if __name__ == "__main__":
    raise SystemExit(main(default_version="v3"))
