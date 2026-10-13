"""
The single place in the project that talks to MySQL.

WHY THIS FILE EXISTS
--------------------
"One door in, one door out." Notebooks, the Streamlit app, and (later) the
FastAPI service all call `get_engine()` from here. If we migrate to PostgreSQL
in a future module, only this file changes.

KEY IDEA: an "engine"
---------------------
SQLAlchemy's Engine is a reusable connection factory / pool. Creating it is
cheap; creating a brand-new raw connection for every query is slow. We build it
once and cache it.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import DB

logger = logging.getLogger(__name__)

_ENGINE = None  # module-level cache


class DatabaseUnavailable(RuntimeError):
    """Raised when MySQL cannot be reached.

    The Streamlit app catches this and falls back to the cleaned CSV files, so a
    demo never dies just because the MySQL service is stopped.
    """


def get_engine(force_new: bool = False):
    """Return a cached SQLAlchemy engine pointed at the project database."""
    global _ENGINE
    if _ENGINE is not None and not force_new:
        return _ENGINE

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise DatabaseUnavailable(
            "SQLAlchemy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        _ENGINE = create_engine(
            DB.url,
            pool_pre_ping=True,  # silently drop dead connections instead of erroring
            pool_recycle=3600,   # MySQL closes idle connections after ~8h; refresh first
            future=True,
        )
    except Exception as exc:  # noqa: BLE001 - bad URL, missing driver, etc.
        raise DatabaseUnavailable(f"Could not build MySQL engine: {exc}") from exc

    return _ENGINE


def get_server_engine():
    """Engine that connects to the MySQL *server* without selecting a database.

    Used only by `scripts/02_setup_database.py` to run CREATE DATABASE.
    """
    from sqlalchemy import create_engine

    return create_engine(DB.server_url, future=True, isolation_level="AUTOCOMMIT")


def test_connection() -> tuple[bool, str]:
    """Try a trivial query. Returns (ok, human-readable message).

    Never raises - designed to be displayed directly in the Streamlit sidebar.
    """
    try:
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"Connected to MySQL database '{DB.database}' on {DB.host}:{DB.port}"
    except Exception as exc:  # noqa: BLE001
        return False, f"MySQL not reachable ({type(exc).__name__}): {exc}"


def read_sql(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Run a SELECT statement and hand back a pandas DataFrame.

    Parameters are passed separately (never glued into the string) which is how
    you avoid SQL injection.
    """
    from sqlalchemy import text

    engine = get_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception as exc:  # noqa: BLE001
        raise DatabaseUnavailable(f"Query failed: {exc}") from exc


def table_exists(table_name: str) -> bool:
    """True if `table_name` is present in the configured database."""
    try:
        from sqlalchemy import inspect

        return inspect(get_engine()).has_table(table_name)
    except Exception:  # noqa: BLE001
        return False


def row_count(table_name: str) -> int:
    """Number of rows in a table, or -1 if it cannot be read."""
    try:
        df = read_sql(f"SELECT COUNT(*) AS n FROM `{table_name}`")
        return int(df.iloc[0]["n"])
    except Exception:  # noqa: BLE001
        return -1


def split_sql_statements(sql_text: str) -> list[str]:
    """Split a SQL script into individual statements.

    Naively splitting on ";" is WRONG and caused a real bug in this project: a
    comment reading `-- Age arrives as a band; `age` is the midpoint.` chopped a
    CREATE TABLE in half, so every table after it silently failed to be created.

    This walks the text character by character and only treats ";" as a
    separator when it is not inside:
      * a line comment  (-- ... or # ...)
      * a block comment (/* ... */)
      * a quoted string ('...' or "...")
      * a backquoted identifier (`...`)
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql_text)
    quote: str | None = None      # active quote character, if any
    line_comment = False
    block_comment = False

    while i < n:
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < n else ""

        if line_comment:
            buf.append(ch)
            if ch == "\n":
                line_comment = False
        elif block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 1
                block_comment = False
        elif quote:
            buf.append(ch)
            if ch == "\\" and nxt:          # escaped character inside a string
                buf.append(nxt)
                i += 1
            elif ch == quote:
                quote = None
        elif ch == "-" and nxt == "-":
            buf.append(ch)
            line_comment = True
        elif ch == "#":
            buf.append(ch)
            line_comment = True
        elif ch == "/" and nxt == "*":
            buf.append(ch)
            buf.append(nxt)
            i += 1
            block_comment = True
        elif ch in ("'", '"', "`"):
            buf.append(ch)
            quote = ch
        elif ch == ";":
            statements.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    # Drop anything that is only comments or whitespace.
    real: list[str] = []
    for stmt in statements:
        body = "\n".join(
            line for line in stmt.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ).strip()
        if body:
            real.append(stmt)
    return real


def _first_real_line(stmt: str) -> str:
    """The first non-comment line of a statement, for error messages."""
    for line in stmt.splitlines():
        text_line = line.strip()
        if text_line and not text_line.startswith("--"):
            return text_line[:70]
    return stmt.strip()[:70]


def execute_script(sql_text: str) -> int:
    """Run a multi-statement SQL script (e.g. our schema.sql).

    MySQL's driver executes one statement per call, so the script is split
    first. Each statement runs in its own transaction: DDL auto-commits in MySQL
    anyway, so a shared transaction gives a false sense of atomicity while
    hiding which statement actually failed.

    Raises RuntimeError naming the failing statement, so the error you see
    identifies the problem instead of only its downstream symptom.
    """
    from sqlalchemy import text

    statements = split_sql_statements(sql_text)
    engine = get_engine()
    executed = 0
    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            executed += 1
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"SQL statement {executed + 1} of {len(statements)} failed.\n"
                f"  statement starts: {_first_real_line(stmt)}\n"
                f"  database said:    {exc}"
            ) from exc
    return executed
