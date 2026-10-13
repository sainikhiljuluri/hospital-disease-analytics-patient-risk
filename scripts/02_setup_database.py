"""
STEP 2 OF 3 - create the MySQL database and its tables.

RUN IT LIKE THIS:
    python scripts/02_setup_database.py

WHAT IT DOES
    1. Connects to your MySQL SERVER (not to a specific database yet).
    2. Runs CREATE DATABASE IF NOT EXISTS hospital_analytics.
    3. Connects to that database and runs src/hospital_analytics/sql/schema.sql,
       which creates 4 tables.

Safe to run more than once - every statement uses IF NOT EXISTS, so nothing is
overwritten and no data is lost.

IF IT FAILS, it is almost always one of these:
    * MySQL is not running        -> start the MySQL service
    * Wrong password              -> fix MYSQL_PASSWORD in your .env file
    * .env file does not exist    -> copy .env.example to .env and edit it
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from hospital_analytics.config import DB  # noqa: E402
from hospital_analytics.db import (  # noqa: E402
    execute_script,
    get_server_engine,
    table_exists,
    test_connection,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "hospital_analytics" / "sql" / "schema.sql"
)


def main() -> int:
    print("=" * 66)
    print("MySQL setup")
    print("=" * 66)
    print(f"  host     : {DB.host}:{DB.port}")
    print(f"  user     : {DB.user}")
    print(f"  database : {DB.database}")
    print(f"  password : {'(set)' if DB.password else '(EMPTY - is that right?)'}")
    print()

    # --- 1. create the database -------------------------------------------
    print("Creating database if it does not exist...")
    try:
        engine = get_server_engine()
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{DB.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
        print(f"  OK - database '{DB.database}' is ready.")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        print()
        print("Troubleshooting:")
        print("  1. Is MySQL running?")
        print("       Windows : open 'Services' and check MySQL80 is Running")
        print("       macOS   : brew services list")
        print("       Linux   : sudo systemctl status mysql")
        print("  2. Are the credentials in your .env file correct?")
        print("  3. Try connecting by hand:  mysql -u root -p")
        return 1

    # --- 2. create the tables ---------------------------------------------
    print()
    print("Creating tables from schema.sql...")
    try:
        n = execute_script(SCHEMA_PATH.read_text(encoding="utf-8"))
        print(f"  OK - executed {n} SQL statements.")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        print()
        print("The message above names the statement that failed. If it mentions")
        print("a syntax error, check your MySQL version is 5.7 or newer:")
        print("    mysql --version")
        return 1

    # --- 3. VERIFY the tables really exist ---------------------------------
    # Do not trust "no error" as proof of success. Check for each table by name.
    # A partial schema is worse than a failed one, because the failure surfaces
    # later as a confusing error in a different script.
    print()
    print("Verifying tables...")
    expected = ["disease_reference", "patients", "medications", "data_load_log"]
    missing = [t for t in expected if not table_exists(t)]

    for table in expected:
        print(f"  {'OK     ' if table not in missing else 'MISSING'} {table}")

    if missing:
        print()
        print(f"  {len(missing)} table(s) were not created.")
        print("  Re-run this script. If it fails again, the error message above")
        print("  names the exact SQL statement that failed.")
        return 1

    print()
    ok, msg = test_connection()
    print(("  " if ok else "  NOT OK: ") + msg)
    print()
    print("Next step:  python scripts/03_load_data.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
