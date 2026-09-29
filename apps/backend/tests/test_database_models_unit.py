"""Unit tests for the Milestone 8 ORM models and Alembic migration
(docs/25_Database_Schema.md, ADR-0003), verifiable WITHOUT a live
PostgreSQL connection: SQLAlchemy metadata introspection and Alembic's
own offline SQL-generation mode (`alembic upgrade/downgrade --sql`,
which compiles real dialect-specific DDL text without connecting to any
database).

Live CRUD/constraint-enforcement/transaction tests against a real
PostgreSQL server could NOT be run in this environment -- no live server
is reachable. Installed client/container tools do not establish server
availability; this is verified by the application's connection check.
This is explicitly reported here and in the Milestone 8 completion
report, per that milestone's own fallback instruction, rather than
silently substituting SQLite or claiming untested behavior works.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "apps" / "backend"

EXPECTED_TABLES = {
    "customers", "bank_accounts", "payees", "cheque_issuance", "transactions",
    "reference_signatures", "processed_cheque_history", "users", "cheques",
    "ocr_results", "validation_results", "fraud_results", "signature_results",
    "duplicate_results", "anomaly_results", "risk_assessments", "decisions",
    "manual_review_cases", "audit_logs",
}

# docs/25 S8 table list -- the canonical 14 tables this milestone's
# instructions require, using exactly these names.
DOCS25_CANONICAL_TABLES = {
    "customers", "bank_accounts", "cheques", "ocr_results", "validation_results",
    "fraud_results", "signature_results", "duplicate_results", "anomaly_results",
    "risk_assessments", "decisions", "manual_review_cases", "users", "audit_logs",
}


@pytest.fixture(scope="module")
def metadata():
    import sys as _sys

    if str(BACKEND_DIR) not in _sys.path:
        _sys.path.insert(0, str(BACKEND_DIR))
    from app.core.database import Base
    import app.models  # noqa: F401 -- registers every table

    return Base.metadata


def test_all_expected_tables_registered(metadata):
    assert set(metadata.tables.keys()) == EXPECTED_TABLES


def test_all_docs25_canonical_table_names_present(metadata):
    """Every one of docs/25's own 14 documented table names must be
    used verbatim (this milestone's explicit instruction #3)."""
    missing = DOCS25_CANONICAL_TABLES - set(metadata.tables.keys())
    assert missing == set()


def test_cheques_table_has_amount_check_constraint(metadata):
    table = metadata.tables["cheques"]
    check_names = {c.name for c in table.constraints if hasattr(c, "sqltext")}
    assert "chk_cheque_amount_non_negative" in check_names


def test_decisions_table_has_enum_check_constraint(metadata):
    table = metadata.tables["decisions"]
    check_names = {c.name for c in table.constraints if hasattr(c, "sqltext")}
    assert "chk_decision_value" in check_names


def test_risk_and_fraud_score_range_constraints_present(metadata):
    fraud_checks = {c.name for c in metadata.tables["fraud_results"].constraints if hasattr(c, "sqltext")}
    risk_checks = {c.name for c in metadata.tables["risk_assessments"].constraints if hasattr(c, "sqltext")}
    assert "chk_fraud_score_range" in fraud_checks
    assert "chk_risk_score_range" in risk_checks


def test_cheques_foreign_key_to_bank_accounts(metadata):
    table = metadata.tables["cheques"]
    fk_targets = {fk.target_fullname for fk in table.foreign_keys}
    assert "bank_accounts.account_id" in fk_targets


def test_result_tables_have_unique_cheque_id_for_one_to_one_relationship(metadata):
    """docs/25 S4: cheques 1:1 ocr_results/validation_results/fraud_results/
    signature_results/anomaly_results/risk_assessments/decisions."""
    for table_name in ("ocr_results", "validation_results", "fraud_results", "signature_results", "anomaly_results", "risk_assessments", "decisions"):
        table = metadata.tables[table_name]
        cheque_id_col = table.columns["cheque_id"]
        assert cheque_id_col.foreign_keys, f"{table_name}.cheque_id should be a FK to cheques.cheque_id"
        assert cheque_id_col.unique, f"{table_name}.cheque_id must be UNIQUE for a 1:1 relationship"


def test_duplicate_results_cheque_id_not_unique_for_one_to_many(metadata):
    """duplicate_results is documented as cheques 1:N (docs/25 S4)."""
    table = metadata.tables["duplicate_results"]
    assert table.columns["cheque_id"].unique is not True


def test_audit_logs_metadata_column_maps_to_reserved_python_attribute(metadata):
    """docs/27 S13's canonical field is literally named `metadata`; since
    that name is reserved by SQLAlchemy's DeclarativeBase, this verifies
    the DB column is still named "metadata" even though the Python
    attribute is `event_metadata`."""
    table = metadata.tables["audit_logs"]
    assert "metadata" in table.columns


def test_manual_review_cases_reviewer_decision_check_constraint(metadata):
    table = metadata.tables["manual_review_cases"]
    check_names = {c.name for c in table.constraints if hasattr(c, "sqltext")}
    assert "chk_reviewer_decision_value" in check_names


# ----------------------------------------------------------------------
# Alembic migration -- verified via offline SQL generation (no live DB
# connection required; Alembic's own documented offline mode).
# ----------------------------------------------------------------------

def _run_alembic(args: list[str]) -> subprocess.CompletedProcess:
    python = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    return subprocess.run(
        [str(python), "-m", "alembic", *args],
        cwd=str(BACKEND_DIR), capture_output=True, text=True, timeout=30,
    )


def test_alembic_upgrade_offline_generates_complete_sql():
    result = _run_alembic(["upgrade", "head", "--sql"])
    assert result.returncode == 0, result.stderr
    sql = result.stdout
    for table_name in EXPECTED_TABLES:
        assert f"CREATE TABLE {table_name} " in sql, f"missing CREATE TABLE for {table_name}"
    assert sql.strip().endswith("COMMIT;") or "COMMIT;" in sql


def test_alembic_downgrade_offline_generates_complete_sql():
    result = _run_alembic(["downgrade", "f8c784d77f6b:base", "--sql"])
    assert result.returncode == 0, result.stderr
    sql = result.stdout
    for table_name in EXPECTED_TABLES:
        assert f"DROP TABLE {table_name}" in sql, f"missing DROP TABLE for {table_name}"


def test_no_live_postgresql_available_in_this_environment():
    """Documents, rather than hides, the real environment limitation
    this milestone's tests operate under."""
    from app.core.database import check_database_connection

    assert check_database_connection() is False
