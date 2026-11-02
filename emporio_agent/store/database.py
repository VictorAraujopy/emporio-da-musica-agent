"""The store's SQLite database: built from the CSV files when missing, opened read-only."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from emporio_agent.config import PROJECT_ROOT
from emporio_agent.store.seed import seed_database

DATABASE_FILE = PROJECT_ROOT / "storage" / "store.db"
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def ensure_database(database_file: Path = DATABASE_FILE, data_dir: Path = DATA_DIR) -> None:
    """Build the database from the CSV files if its file does not exist yet."""
    if database_file.exists():
        return
    build_database(database_file, data_dir)


def build_database(database_file: Path, data_dir: Path) -> None:
    """Create and seed the database file, deleting it if any step fails."""
    database_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_and_seed(database_file, data_dir)
    except BaseException:
        database_file.unlink(missing_ok=True)
        raise


def create_and_seed(database_file: Path, data_dir: Path) -> None:
    """Create the tables in a new database file and insert the CSV rows into them."""
    with closing(sqlite3.connect(database_file)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
        connection.executescript(schema_sql)
        seed_database(connection, data_dir)
        connection.commit()


@contextmanager
def open_read_only(database_file: Path = DATABASE_FILE) -> Iterator[sqlite3.Connection]:
    """Open the existing database for reading only, and close it when the block ends."""
    database_uri = f"{database_file.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
