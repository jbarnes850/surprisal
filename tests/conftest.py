import pytest
from surprisal.db import Database


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    yield db
    db.close()
