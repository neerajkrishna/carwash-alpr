"""
database.py — PostgreSQL schema and persistence layer
======================================================
Connects to PostgreSQL using the DATABASE_URL environment variable.
Creates the detections table on first run if it doesn't exist.

Table: detections
  - id            : auto-increment primary key
  - camera_name   : name of the camera that made the detection
  - camera_type   : "primary" (plate+colour+brand) or "secondary" (plate only)
  - plate         : licence plate text, or NULL if not read
  - colour        : vehicle colour, NULL for secondary cameras
  - colour_conf   : colour confidence 0.0-1.0, NULL for secondary cameras
  - brand         : brand name, NULL for secondary cameras
  - brand_conf    : brand confidence 0.0-1.0, NULL for secondary cameras
  - detected_at   : timestamp when the vehicle crossed the entry line
  - video_time    : timestamp string within the video (MM:SS)

Environment variable:
  DATABASE_URL   postgresql://user:password@host:5432/dbname
"""

import os
import logging
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

logger = logging.getLogger("database")

# ── Connection pool ────────────────────────────────────────────────────────────
# Uses a thread-safe connection pool so multiple camera pipelines can write
# concurrently without blocking each other.
DATABASE_URL = os.environ.get("DATABASE_URL", "")
_pool: pool.ThreadedConnectionPool | None = None


def init_db():
    """
    Initialize the connection pool and create the detections table if it does not exist.
    Must be called once at server startup before any other database functions.
    """
    global _pool

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set.\n"
            "Example: export DATABASE_URL=postgresql://user:pass@localhost:5432/alpr"
        )

    logger.info("Connecting to PostgreSQL...")
    _pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=DATABASE_URL)

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id           SERIAL PRIMARY KEY,
                    camera_name  TEXT        NOT NULL,
                    camera_type  TEXT        NOT NULL DEFAULT 'primary',
                    plate        TEXT,
                    colour       TEXT,
                    colour_conf  REAL,
                    brand        TEXT,
                    brand_conf   REAL,
                    detected_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    video_time   TEXT,
                    UNIQUE (plate, camera_name)
                );
            """)
            # Index on camera_name and detected_at for fast dashboard queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_detections_camera
                ON detections (camera_name, detected_at DESC);
            """)
            # Ensure the unique constraint exists on pre-existing tables
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE detections ADD CONSTRAINT uq_plate_camera UNIQUE (plate, camera_name);
                EXCEPTION WHEN duplicate_table OR duplicate_object THEN NULL;
                END $$;
            """)
            # Clear previous session data so the UI starts fresh on every run
            cur.execute("TRUNCATE TABLE detections RESTART IDENTITY;")
        conn.commit()

    logger.info("Database ready — previous session data cleared.")


@contextmanager
def _get_conn():
    """Context manager that borrows a connection from the pool and returns it after use."""
    if _pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings. Returns 999 if lengths differ by more than 2."""
    if abs(len(a) - len(b)) > 2:
        return 999
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def insert_detection(
    camera_name: str,
    camera_type: str,
    plate: str | None,
    colour: str | None,
    colour_conf: float | None,
    brand: str | None,
    brand_conf: float | None,
    video_time: str | None = None,
) -> int:
    """
    Insert a finalized vehicle detection into the database.

    For secondary cameras, colour/brand/conf values should be passed as None.
    Returns the new row's id.

    Example:
        insert_detection(
            camera_name="Gate-A-Primary",
            camera_type="primary",
            plate="AB123CD",
            colour="White",
            colour_conf=0.91,
            brand="Toyota",
            brand_conf=0.62,
            video_time="01:24",
        )
    """
    # Normalise empty strings to None for cleaner DB records
    plate  = plate  if plate  and plate  != "—" else None
    colour = colour if colour and colour != "?" else None
    brand  = brand  if brand  and brand  != "—" else None

    with _get_conn() as conn:
        with conn.cursor() as cur:
            row_id = None

            # Check for a prefix-match plate from the same camera.
            # e.g. existing "ADI28" and new "ADI283" — one is a truncated read of the other.
            if plate:
                # Step 1: prefix match (one plate is a truncated read of the other)
                cur.execute("""
                    SELECT id, plate FROM detections
                    WHERE camera_name = %s AND plate IS NOT NULL
                      AND (%s LIKE plate || '%%' OR plate LIKE %s || '%%')
                    ORDER BY detected_at DESC
                    LIMIT 2;
                """, (camera_name, plate, plate))
                match = cur.fetchone()
                if match:
                    existing_id, existing_plate = match
                    if len(plate) > len(existing_plate):
                        cur.execute("""
                            UPDATE detections SET plate = %s, detected_at = NOW()
                            WHERE id = %s RETURNING id;
                        """, (plate, existing_id))
                        row_id = cur.fetchone()[0]
                    else:
                        row_id = existing_id

                # Step 2: fuzzy match — check only the 2 most recent records for this camera
                if row_id is None:
                    cur.execute("""
                        SELECT id, plate FROM detections
                        WHERE camera_name = %s AND plate IS NOT NULL
                          AND ABS(length(plate) - length(%s)) <= 2
                        ORDER BY detected_at DESC
                        LIMIT 2;
                    """, (camera_name, plate))
                    for existing_id, existing_plate in cur.fetchall():
                        if _edit_distance(plate, existing_plate) <= 2:
                            if len(plate) >= len(existing_plate):
                                cur.execute("""
                                    UPDATE detections SET plate = %s, detected_at = NOW()
                                    WHERE id = %s RETURNING id;
                                """, (plate, existing_id))
                                row_id = cur.fetchone()[0]
                            else:
                                row_id = existing_id
                            break

            if row_id is None:
                cur.execute("""
                    INSERT INTO detections
                        (camera_name, camera_type, plate, colour, colour_conf,
                         brand, brand_conf, video_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (plate, camera_name) DO UPDATE SET
                        colour      = CASE
                                        WHEN EXCLUDED.colour_conf IS NOT NULL
                                         AND (detections.colour_conf IS NULL
                                              OR EXCLUDED.colour_conf > detections.colour_conf)
                                        THEN EXCLUDED.colour
                                        ELSE detections.colour
                                      END,
                        colour_conf = GREATEST(detections.colour_conf, EXCLUDED.colour_conf),
                        brand       = CASE
                                        WHEN EXCLUDED.brand_conf IS NOT NULL
                                         AND (detections.brand_conf IS NULL
                                              OR EXCLUDED.brand_conf > detections.brand_conf)
                                        THEN EXCLUDED.brand
                                        ELSE detections.brand
                                      END,
                        brand_conf  = GREATEST(detections.brand_conf, EXCLUDED.brand_conf),
                        detected_at = NOW()
                    RETURNING id;
                """, (camera_name, camera_type, plate, colour, colour_conf,
                      brand, brand_conf, video_time))
                row_id = cur.fetchone()[0]

        conn.commit()

    logger.debug(f"Upserted detection id={row_id} camera={camera_name} plate={plate}")
    return row_id


def fetch_recent(camera_name: str | None = None, limit: int = 100) -> list[dict]:
    """
    Fetch the most recent detections, optionally filtered by camera_name.
    Returns a list of dicts with all columns.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if camera_name:
                cur.execute("""
                    SELECT * FROM detections
                    WHERE camera_name = %s
                    ORDER BY detected_at DESC
                    LIMIT %s;
                """, (camera_name, limit))
            else:
                cur.execute("""
                    SELECT * FROM detections
                    ORDER BY detected_at DESC
                    LIMIT %s;
                """, (limit,))
            return [dict(r) for r in cur.fetchall()]


def fetch_stats() -> dict:
    """
    Return aggregate stats per camera: total detections, unique plates.
    Useful for the dashboard summary panel.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    camera_name,
                    camera_type,
                    COUNT(*) AS total,
                    COUNT(DISTINCT plate) FILTER (WHERE plate IS NOT NULL) AS unique_plates
                FROM detections
                GROUP BY camera_name, camera_type
                ORDER BY camera_name;
            """)
            return [dict(r) for r in cur.fetchall()]
