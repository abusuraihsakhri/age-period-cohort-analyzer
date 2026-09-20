#!/usr/bin/env python3
"""Compatibility launcher for the repository test suite."""

__test__ = False

if __name__ == "__main__":
    try:
        import pytest
    except ImportError as exc:
        raise SystemExit("pytest is required to run the test suite") from exc
    raise SystemExit(pytest.main(["-q", "tests/test_apc_analyzer.py"]))
