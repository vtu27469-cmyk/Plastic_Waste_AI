import sqlite3
from datetime import datetime

DATABASE = "plastic_detection.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            plastic_count INTEGER NOT NULL,
            detected_objects TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_detection(image_name, plastic_count, detected_objects, confidence):
    connection = get_connection()

    cursor = connection.execute("""
        INSERT INTO detections
        (image_name, plastic_count, detected_objects, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_name,
        plastic_count,
        detected_objects,
        confidence,
        datetime.now().isoformat()
    ))

    detection_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return detection_id


def get_all_detections():
    connection = get_connection()

    rows = connection.execute("""
        SELECT *
        FROM detections
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]