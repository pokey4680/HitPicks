"""Shared test fixtures."""

import pytest
import sqlite3

from bts import cache


@pytest.fixture
def conn():
    """In-memory SQLite connection with BTS tables."""
    c = sqlite3.connect(":memory:")
    cache._create_tables(c)
    yield c
    c.close()
