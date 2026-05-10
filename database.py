import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def init_database(db_path: Path):
    """Initialize SQLite database for caching and generated artifacts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_cache (
            step_id TEXT PRIMARY KEY,
            step_name TEXT NOT NULL,
            output_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stl_conversions (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            category TEXT,
            stl_formula TEXT NOT NULL,
            value_sum TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS json_artifacts (
            artifact_name TEXT PRIMARY KEY,
            artifact_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            output_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (cache_key, fingerprint)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_scenario_manifest (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            scenic_rules_repo TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            implementation_harness TEXT NOT NULL,
            implementation_provider TEXT,
            implementation_model TEXT NOT NULL,
            implementation_reasoning TEXT,
            entries_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_column_if_missing(cursor, "rule_scenario_manifest", "implementation_provider TEXT")
    _add_column_if_missing(cursor, "rule_scenario_manifest", "implementation_reasoning TEXT")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_rule_scenarios (
            branch TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            category TEXT,
            scenario_name TEXT,
            scenario_path TEXT,
            status TEXT,
            generated_at TEXT,
            implementation_harness TEXT,
            implementation_provider TEXT,
            implementation_model TEXT,
            implementation_reasoning TEXT,
            entry_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _add_column_if_missing(cursor, "generated_rule_scenarios", "implementation_provider TEXT")
    _add_column_if_missing(cursor, "generated_rule_scenarios", "implementation_reasoning TEXT")

    conn.commit()
    return conn


def _add_column_if_missing(cursor, table_name: str, column_sql: str) -> None:
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc):
            raise


def _fingerprint(payload):
    """Stable short hash of any JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cached_api_call(db_path: Path, model: str, cache_key, fn, fingerprint_payload):
    """Cache API results keyed by (step, fingerprint of prompt+model)."""
    fp = _fingerprint({"model": model, "payload": fingerprint_payload})
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT output_json FROM api_cache WHERE cache_key = ? AND fingerprint = ?",
            (cache_key, fp),
        ).fetchone()
        if row:
            print(f"  [cache hit: {cache_key}.{fp}]")
            return json.loads(row[0])
    finally:
        conn.close()

    result = fn()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO api_cache
                (cache_key, fingerprint, output_json, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (cache_key, fp, json.dumps(result, indent=2)))
        conn.commit()
    finally:
        conn.close()
    return result


def cache_step_to_db(conn, step_id, step_name, output_data):
    """Cache pipeline step results to SQLite database."""
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_cache (step_id, step_name, output_json)
        VALUES (?, ?, ?)
    """, (step_id, step_name, json.dumps(output_data)))
    conn.commit()


def retrieve_step_from_db(conn, step_id):
    """Retrieve cached pipeline step from SQLite database."""
    row = conn.execute(
        "SELECT output_json FROM pipeline_cache WHERE step_id = ?",
        (step_id,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def store_json_artifact(conn, artifact_name: str, artifact_data: dict) -> None:
    """Store a complete JSON artifact in SQLite."""
    conn.execute("""
        INSERT OR REPLACE INTO json_artifacts
            (artifact_name, artifact_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (artifact_name, json.dumps(artifact_data, indent=2)))
    conn.commit()


def retrieve_json_artifact(conn, artifact_name: str) -> dict | None:
    """Return a JSON artifact from SQLite, if present."""
    row = conn.execute(
        "SELECT artifact_json FROM json_artifacts WHERE artifact_name = ?",
        (artifact_name,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def store_rule_scenario_manifest(
    conn,
    manifest: dict,
    scenic_rules_path: Path,
    implementation_harness: str,
    implementation_provider: str,
    implementation_model: str,
    implementation_reasoning: str,
) -> None:
    """Store generated-rule scenario manifest as both a JSON artifact and rows."""
    entries = manifest.get("entries", [])
    stored_harness = manifest.get("implementation_harness", implementation_harness)
    stored_provider = manifest.get("implementation_provider", implementation_provider)
    stored_model = manifest.get("implementation_model", implementation_model)
    stored_reasoning = manifest.get("implementation_reasoning", implementation_reasoning)
    generated_at = manifest.get("generated_at", datetime.now().isoformat(timespec="seconds"))

    store_json_artifact(conn, "generated_rule_scenarios", manifest)

    conn.execute("""
        INSERT OR REPLACE INTO rule_scenario_manifest
            (id, scenic_rules_repo, generated_at, implementation_harness,
             implementation_provider, implementation_model, implementation_reasoning,
             entries_json, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        manifest.get("scenic_rules_repo", str(scenic_rules_path)),
        generated_at,
        stored_harness,
        stored_provider,
        stored_model,
        stored_reasoning,
        json.dumps(entries, indent=2),
    ))

    conn.execute("DELETE FROM generated_rule_scenarios")
    conn.executemany("""
        INSERT OR REPLACE INTO generated_rule_scenarios
            (branch, rule_text, category, scenario_name, scenario_path, status,
             generated_at, implementation_harness, implementation_provider,
             implementation_model, implementation_reasoning, entry_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, [
        (
            entry.get("branch") or f"entry_{idx}",
            entry.get("rule", ""),
            entry.get("category", ""),
            entry.get("scenario_name", ""),
            entry.get("scenario_path", ""),
            entry.get("status", ""),
            generated_at,
            stored_harness,
            stored_provider,
            stored_model,
            stored_reasoning,
            json.dumps(entry, indent=2),
        )
        for idx, entry in enumerate(entries)
    ])
    conn.commit()
