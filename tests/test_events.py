from src.database import Database


def test_register_and_count_visitors(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    try:
        first_id = db.add_face(
            "2026-10-16T10:00:00+00:00", b"face1"
        )
        second_id = db.add_face(
            "2026-10-16T10:01:00+00:00", b"face2"
        )

        assert first_id != second_id
        assert db.unique_visitor_count() == 2
    finally:
        db.close()


def test_entry_and_exit_events(tmp_path):
    db = Database(str(tmp_path / "events.db"))

    try:
        face_id = db.add_face(
            "2026-10-16T10:00:00+00:00", b"face1"
        )

        db.add_event(
            face_id, 10, "entry",
            "2026-10-16T10:00:00+00:00",
            "logs/entries/test.jpg",
        )
        db.add_event(
            face_id, 10, "exit",
            "2026-10-16T10:05:00+00:00",
            "logs/exits/test.jpg",
        )

        events = db.connection.execute(
            "SELECT event_type FROM events ORDER BY event_id"
        ).fetchall()

        assert [row[0] for row in events] == ["entry", "exit"]
    finally:
        db.close()


def test_duplicate_event_is_not_inserted(tmp_path):
    db = Database(str(tmp_path / "duplicate.db"))

    try:
        face_id = db.add_face(
            "2026-10-16T10:00:00+00:00", b"face1"
        )

        first = db.add_event(
            face_id, 20, "entry",
            "2026-10-16T10:00:00+00:00",
            "logs/entries/test.jpg",
        )
        duplicate = db.add_event(
            face_id, 20, "entry",
            "2026-10-16T10:00:01+00:00",
            "logs/entries/test_duplicate.jpg",
        )

        count = db.connection.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE track_id = 20 AND event_type = 'entry'"
        ).fetchone()[0]

        assert first is not None
        assert duplicate is None
        assert count == 1
    finally:
        db.close()
