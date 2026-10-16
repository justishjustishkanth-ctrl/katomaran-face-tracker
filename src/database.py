import sqlite3
from pathlib import Path


class Database:
    def __init__(self, db_path):
        if str(db_path) == ":memory:":
            self.connection = sqlite3.connect(":memory:")
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path))

        self.connection.execute("PRAGMA foreign_keys = ON")
        self.create_tables()

    def create_tables(self):
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_seen_at TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL
                        CHECK(event_type IN ('entry', 'exit')),
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    FOREIGN KEY(face_id) REFERENCES faces(face_id)
                )
            """)

    def get_faces(self):
        return self.connection.execute(
            "SELECT face_id, embedding FROM faces ORDER BY face_id"
        ).fetchall()

    def add_face(self, first_seen_at, embedding):
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO faces (first_seen_at, embedding) VALUES (?, ?)",
                (first_seen_at, sqlite3.Binary(embedding)),
            )
        return cursor.lastrowid

    def unique_visitor_count(self):
        return self.connection.execute(
            "SELECT COUNT(*) FROM faces"
        ).fetchone()[0]

    def close(self):
        self.connection.close()

    def add_event(self, face_id, track_id, event_type, timestamp, image_path):
        """Save an entry or exit event."""
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO events
                    (face_id, track_id, event_type, timestamp, image_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (face_id, track_id, event_type, timestamp, image_path),
            )
        return cursor.lastrowid

    def add_event(self, face_id, track_id, event_type, timestamp, image_path):
        if event_type not in ("entry", "exit"):
            raise ValueError("event_type must be 'entry' or 'exit'")

        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO events (
                    face_id, track_id, event_type, timestamp, image_path
                )
                SELECT ?, ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM events
                    WHERE track_id = ? AND event_type = ?
                )
                """,
                (
                    face_id, track_id, event_type, timestamp, image_path,
                    track_id, event_type
                ),
            )

        return cursor.lastrowid if cursor.rowcount else None
