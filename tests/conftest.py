"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Enable a command line flag for running tests decorated with @runllms."""
    parser.addoption(
        "--runllms",
        action="store_true",
        dest="runllms",
        default=False,
        help="Run tests with real LLMs",
    )
