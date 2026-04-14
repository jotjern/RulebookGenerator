"""Simple Flask server to expose SQLite database as JSON API."""
import json
import sqlite3
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = Path(__file__).parent.parent / "rulebook_pipeline.db"


def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/api/pipeline-status", methods=["GET"])
def get_pipeline_status():
    """Get all cached pipeline steps."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT step_id, step_name, created_at FROM pipeline_cache
        ORDER BY created_at DESC
    """)
    steps = []
    for row in cursor.fetchall():
        steps.append({
            "step_id": row[0],
            "step_name": row[1],
            "created_at": row[2],
        })
    conn.close()
    return jsonify({
        "status": "success",
        "total_steps": len(steps),
        "steps": steps,
    })


@app.route("/api/stl-conversions", methods=["GET"])
def get_stl_conversions():
    """Get all STL conversions from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT rule_id, rule_text, category, stl_formula, value_sum, created_at
        FROM stl_conversions
        ORDER BY created_at DESC
        LIMIT 100
    """)
    conversions = []
    for row in cursor.fetchall():
        conversions.append({
            "rule_id": row[0],
            "rule_text": row[1],
            "category": row[2],
            "stl_formula": row[3],
            "value_sum": row[4],
            "created_at": row[5],
        })
    conn.close()
    return jsonify({
        "status": "success",
        "total_conversions": len(conversions),
        "conversions": conversions,
    })


@app.route("/api/step/<step_id>", methods=["GET"])
def get_step_details(step_id):
    """Get detailed output for a specific pipeline step."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT step_name, output_json, created_at FROM pipeline_cache
        WHERE step_id = ?
    """, (step_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"status": "error", "message": "Step not found"}), 404

    return jsonify({
        "status": "success",
        "step_id": step_id,
        "step_name": row[0],
        "created_at": row[2],
        "data": json.loads(row[1]),
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get overall statistics from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get counts
    cursor.execute("SELECT COUNT(*) FROM pipeline_cache")
    pipeline_steps = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM stl_conversions")
    total_conversions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT category) FROM stl_conversions
        WHERE category IS NOT NULL
    """)
    distinct_categories = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "status": "success",
        "pipeline_steps_cached": pipeline_steps,
        "total_stl_conversions": total_conversions,
        "distinct_categories": distinct_categories,
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    db_exists = DB_PATH.exists()
    return jsonify({
        "status": "healthy",
        "database_exists": db_exists,
        "database_path": str(DB_PATH),
    })


if __name__ == "__main__":
    print(f"Starting server... Database: {DB_PATH}")
    app.run(debug=True, port=5000, host="127.0.0.1")
