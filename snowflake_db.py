from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import snowflake.connector as sf_connector
except Exception:  # pragma: no cover - import availability differs by environment
    sf_connector = None


REQUIRED_SNOWFLAKE_ENV_VARS: Tuple[str, ...] = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)
DEFAULT_SNOWFLAKE_TABLE = "ATTENDANCE_RAW"
_SIMPLE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


@dataclass(frozen=True)
class SnowflakeCredentials:
    """Environment-driven Snowflake connection settings."""

    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None
    authenticator: Optional[str] = None

    @classmethod
    def missing_env_vars(cls) -> List[str]:
        return [name for name in REQUIRED_SNOWFLAKE_ENV_VARS if not os.getenv(name)]

    @classmethod
    def from_env(cls) -> Optional["SnowflakeCredentials"]:
        missing = cls.missing_env_vars()
        if missing:
            return None

        return cls(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            role=os.getenv("SNOWFLAKE_ROLE"),
            authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR"),
        )


def _normalize_identifier(identifier: str) -> str:
    """
    Snowflake-specific identifier handling:
    - Unquoted identifiers are uppercased by Snowflake.
    - Non-simple names are safely quoted.
    """
    token = str(identifier or "").strip()
    if not token:
        raise ValueError("Snowflake identifier cannot be empty.")

    if _SIMPLE_IDENTIFIER_PATTERN.match(token):
        return token.upper()

    escaped = token.replace('"', '""')
    return f'"{escaped}"'


def _normalize_value(value: Any) -> Any:
    """Convert DataFrame cell values into JSON-serializable values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value

    return value


@contextmanager
def snowflake_connection(credentials: SnowflakeCredentials):
    """Open and close a Snowflake connection safely via context manager."""
    if sf_connector is None:
        raise ImportError("snowflake-connector-python is not installed.")

    connect_kwargs: Dict[str, Any] = {
        "account": credentials.account,
        "user": credentials.user,
        "password": credentials.password,
        "warehouse": credentials.warehouse,
        "database": credentials.database,
        "schema": credentials.schema,
    }

    if credentials.role:
        connect_kwargs["role"] = credentials.role
    if credentials.authenticator:
        connect_kwargs["authenticator"] = credentials.authenticator

    conn = sf_connector.connect(**connect_kwargs)
    try:
        yield conn
    finally:
        conn.close()


class SnowflakeAttendanceRepository:
    """Persistence adapter for attendance data in Snowflake."""

    def __init__(self, table_name: Optional[str] = None):
        # Snowflake-specific: keep table identifier uppercase-safe by default.
        self.table_name = table_name or os.getenv("SNOWFLAKE_TABLE", DEFAULT_SNOWFLAKE_TABLE)

    def missing_env_vars(self) -> List[str]:
        return SnowflakeCredentials.missing_env_vars()

    def is_configured(self) -> bool:
        return len(self.missing_env_vars()) == 0

    def is_enabled(self) -> bool:
        return self.is_configured() and sf_connector is not None

    def assert_ready(self) -> SnowflakeCredentials:
        missing = self.missing_env_vars()
        if missing:
            raise RuntimeError(
                "Snowflake backend requested but required environment variables are missing: "
                + ", ".join(missing)
            )

        if sf_connector is None:
            raise RuntimeError(
                "Snowflake backend requested but snowflake-connector-python is not installed."
            )

        creds = SnowflakeCredentials.from_env()
        if creds is None:
            raise RuntimeError("Snowflake credentials are not available from environment variables.")

        return creds

    def _qualified_table_name(self, credentials: SnowflakeCredentials) -> str:
        database = _normalize_identifier(credentials.database)
        schema = _normalize_identifier(credentials.schema)
        table = _normalize_identifier(self.table_name)
        return f"{database}.{schema}.{table}"

    def get_table_ddl(self) -> str:
        """Return required Snowflake table DDL for this repository."""
        table = _normalize_identifier(self.table_name)
        return (
            f"CREATE TABLE IF NOT EXISTS {table} (\n"
            "    ROW_HASH STRING NOT NULL,\n"
            "    ROW_DATA VARIANT NOT NULL,\n"
            "    SOURCE_FILE STRING,\n"
            "    INGESTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),\n"
            "    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),\n"
            "    PRIMARY KEY (ROW_HASH)\n"
            ");"
        )

    def ensure_table_exists(self) -> None:
        creds = self.assert_ready()
        table_name = self._qualified_table_name(creds)
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            "    ROW_HASH STRING NOT NULL,\n"
            "    ROW_DATA VARIANT NOT NULL,\n"
            "    SOURCE_FILE STRING,\n"
            "    INGESTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),\n"
            "    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),\n"
            "    PRIMARY KEY (ROW_HASH)\n"
            ")"
        )

        with snowflake_connection(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)

    def _prepare_rows(self, df: pd.DataFrame, source_file: str) -> List[Tuple[str, str, str]]:
        rows: List[Tuple[str, str, str]] = []

        for _, row in df.iterrows():
            payload = {str(col): _normalize_value(value) for col, value in row.items()}
            payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            row_hash = hashlib.md5(payload_json.encode("utf-8")).hexdigest()
            rows.append((row_hash, payload_json, source_file))

        return rows

    def upsert_dataframe(self, df: pd.DataFrame, source_file: str = "uploaded_file") -> int:
        """
        Snowflake-specific merge strategy:
        - Load payloads into a temporary table.
        - Merge by deterministic row hash to avoid duplicates.
        """
        if df is None or df.empty:
            return 0

        creds = self.assert_ready()
        self.ensure_table_exists()
        target_table = self._qualified_table_name(creds)
        temp_table = _normalize_identifier(f"TMP_{self.table_name}_UPLOAD")
        rows = self._prepare_rows(df, source_file)

        create_temp_sql = (
            f"CREATE TEMPORARY TABLE IF NOT EXISTS {temp_table} ("
            "ROW_HASH STRING, ROW_DATA VARIANT, SOURCE_FILE STRING)"
        )
        truncate_temp_sql = f"TRUNCATE TABLE {temp_table}"
        insert_sql = (
            f"INSERT INTO {temp_table} (ROW_HASH, ROW_DATA, SOURCE_FILE) "
            "SELECT %s, PARSE_JSON(%s), %s"
        )
        merge_sql = (
            f"MERGE INTO {target_table} AS TGT "
            f"USING {temp_table} AS SRC "
            "ON TGT.ROW_HASH = SRC.ROW_HASH "
            "WHEN MATCHED THEN UPDATE SET "
            "    ROW_DATA = SRC.ROW_DATA, "
            "    SOURCE_FILE = SRC.SOURCE_FILE, "
            "    UPDATED_AT = CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT "
            "    (ROW_HASH, ROW_DATA, SOURCE_FILE, INGESTED_AT, UPDATED_AT) "
            "VALUES "
            "    (SRC.ROW_HASH, SRC.ROW_DATA, SRC.SOURCE_FILE, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())"
        )

        with snowflake_connection(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(create_temp_sql)
                cur.execute(truncate_temp_sql)
                cur.executemany(insert_sql, rows)
                cur.execute(merge_sql)

        return len(rows)

    def fetch_all_rows(self) -> pd.DataFrame:
        """Read the full attendance payload from Snowflake into a pandas DataFrame."""
        creds = self.assert_ready()
        self.ensure_table_exists()
        table_name = self._qualified_table_name(creds)

        query = (
            f"SELECT ROW_DATA "
            f"FROM {table_name} "
            "ORDER BY INGESTED_AT"
        )

        records: List[Dict[str, Any]] = []
        with snowflake_connection(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                for (payload,) in cur.fetchall():
                    if payload is None:
                        continue

                    if isinstance(payload, dict):
                        records.append(payload)
                        continue

                    if isinstance(payload, str):
                        try:
                            parsed = json.loads(payload)
                            if isinstance(parsed, dict):
                                records.append(parsed)
                        except json.JSONDecodeError:
                            continue
                        continue

                    try:
                        payload_dict = dict(payload)
                    except Exception:
                        continue
                    records.append(payload_dict)

        return pd.DataFrame(records)

    def has_data(self) -> bool:
        creds = self.assert_ready()
        self.ensure_table_exists()
        table_name = self._qualified_table_name(creds)

        query = f"SELECT COUNT(*) FROM {table_name}"
        with snowflake_connection(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row_count = cur.fetchone()[0]

        return int(row_count) > 0

    def get_source_signature(self) -> str:
        """Build cache signature from Snowflake table state."""
        creds = self.assert_ready()
        self.ensure_table_exists()
        table_name = self._qualified_table_name(creds)

        query = (
            "SELECT "
            "    COUNT(*) AS ROW_COUNT, "
            "    COALESCE(MAX(UPDATED_AT), MAX(INGESTED_AT)) AS LAST_CHANGE "
            f"FROM {table_name}"
        )

        with snowflake_connection(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row_count, last_change = cur.fetchone()

        if not row_count:
            return "snowflake-empty"

        timestamp_text = str(last_change) if last_change is not None else "no-timestamp"
        return f"snowflake-{int(row_count)}-{timestamp_text}"


def get_snowflake_table_creation_script(table_name: Optional[str] = None) -> str:
    """Public helper to expose the required Snowflake DDL."""
    repository = SnowflakeAttendanceRepository(table_name=table_name)
    return repository.get_table_ddl()


def get_required_snowflake_env_vars() -> Tuple[str, ...]:
    return REQUIRED_SNOWFLAKE_ENV_VARS
