"""
Checks that schema.sql is valid and loadable — without needing MySQL running.

RUN IT LIKE THIS:
    python scripts/check_schema.py

WHY THIS EXISTS
---------------
A real bug shipped in this project: a SQL comment containing a semicolon —

    -- Age arrives as a 10-year band; `age` is the band midpoint.

— caused the script splitter to chop CREATE TABLE patients in half. That
statement failed, and every table after it was never created. The symptom
appeared much later, in a different script, as:

    Table 'hospital_analytics.medications' doesn't exist

The splitter is now comment-aware, and this script guards against the whole
class of problem: it verifies every table parses, and that the columns the
Python loader writes match the columns the schema declares.

Run it after any edit to schema.sql. It needs no database connection.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hospital_analytics.db import split_sql_statements  # noqa: E402
from hospital_analytics.ingestion.load_to_mysql import (  # noqa: E402
    MEDICATION_DB_COLUMNS,
    PATIENT_DB_COLUMNS,
    REFERENCE_DB_COLUMNS,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "hospital_analytics" / "sql" / "schema.sql"
)

EXPECTED_TABLES = ["disease_reference", "patients", "medications", "data_load_log"]


def schema_columns(statement: str) -> set[str]:
    """Column names declared in one CREATE TABLE statement.

    Comments are stripped FIRST. A leading comment block can contain "(", and
    slicing from the first bracket would then read the prose as column
    definitions and report a wrong count.
    """
    lines = [
        line for line in statement.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    statement = "\n".join(lines)
    body = statement[statement.index("(") + 1:statement.rindex(")")]
    names: set[str] = set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        if re.match(r"(PRIMARY|KEY|CONSTRAINT|FOREIGN|UNIQUE|INDEX|ON|REFERENCES)\b", line, re.I):
            continue
        match = re.match(r"`?(\w+)`?\s+\w", line)
        if match:
            names.add(match.group(1))
    return names


def main() -> int:
    problems: list[str] = []
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    print("=" * 66)
    print("SCHEMA CHECK")
    print("=" * 66)
    print(f"  file: {SCHEMA_PATH}")
    print()

    # --- 1. no semicolons hiding in comments -------------------------------
    for i, line in enumerate(sql.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("--") and ";" in stripped:
            problems.append(
                f"line {i}: a comment contains ';' — this used to split "
                f"statements in half.\n      {stripped[:70]}"
            )

    # --- 2. every expected table parses as a whole statement ---------------
    statements = split_sql_statements(sql)
    print(f"  statements parsed: {len(statements)}")
    print()

    found: dict[str, str] = {}
    for stmt in statements:
        match = re.search(r"CREATE TABLE (?:IF NOT EXISTS )?`?(\w+)`?", stmt)
        if match:
            found[match.group(1)] = stmt

    for table in EXPECTED_TABLES:
        if table not in found:
            problems.append(f"table '{table}' is missing or did not parse")
            print(f"  MISSING  {table}")
            continue
        stmt = found[table]
        if stmt.count("(") != stmt.count(")"):
            problems.append(f"table '{table}': unbalanced parentheses — truncated?")
            print(f"  BROKEN   {table}")
            continue
        cols = schema_columns(stmt)
        print(f"  OK       {table:<20} {len(cols)} columns")

    # --- 3. the loader's columns must exist in the schema ------------------
    print()
    print("  Python loader vs schema:")
    pairs = [
        ("patients", PATIENT_DB_COLUMNS),
        ("medications", MEDICATION_DB_COLUMNS),
        ("disease_reference", REFERENCE_DB_COLUMNS),
    ]
    for table, wanted in pairs:
        if table not in found:
            continue
        declared = schema_columns(found[table])
        missing = [c for c in wanted if c not in declared]
        if missing:
            problems.append(
                f"table '{table}': the loader writes columns the schema does not "
                f"declare: {missing}"
            )
            print(f"    MISMATCH {table}: {missing}")
        else:
            print(f"    OK       {table:<20} all {len(wanted)} loader columns present")

    # --- verdict -----------------------------------------------------------
    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Schema is valid. Safe to run:  python scripts/02_setup_database.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
