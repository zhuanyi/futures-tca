"""Smoke test: verify the package imports and the seed constant is set."""

from futures_tca import REPORT_SEED, __version__


def test_version():
    assert isinstance(__version__, str) and __version__


def test_report_seed():
    assert REPORT_SEED == 20261101
