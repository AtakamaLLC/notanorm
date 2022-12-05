import os

import pytest

from notanorm import SqliteDb

PYTEST_REG = False


@pytest.fixture
def db_sqlite():
    db = SqliteDb(":memory:")
    yield db
    db.close()


@pytest.fixture
def db_sqlite_noup():
    class SqliteDbNoUp(SqliteDb):
        @property
        def _upsert_sql(self, **_):
            raise AttributeError

    db = SqliteDbNoUp(":memory:")

    assert not hasattr(db, "_upsert_sql")

    yield db

    db.close()


@pytest.fixture
def db_mysql_noup():
    from notanorm import MySqlDb

    class MySqlDbNoUp(MySqlDb):
        @property
        def _upsert_sql(self):
            raise AttributeError

    db = get_mysql_db(MySqlDbNoUp)

    assert not hasattr(db, "_upsert_sql")

    yield db

    db.close()


@pytest.fixture
def db_sqlite_notmem(tmp_path):
    db = SqliteDb(str(tmp_path / "db"))
    yield db
    db.close()


def get_mysql_db(typ):
    db = typ(read_default_file=os.path.expanduser("~/.my.cnf"))
    db.query("DROP DATABASE IF EXISTS test_db")
    db.query("CREATE DATABASE test_db")
    db.query("USE test_db")

    return typ(read_default_file=os.path.expanduser("~/.my.cnf"), db="test_db")


def cleanup_mysql_db(db):
    db._DbBase__closed = False
    db.query("SET SESSION TRANSACTION READ WRITE;")
    db.query("DROP DATABASE test_db")
    db.close()


@pytest.fixture
def db_mysql():
    from notanorm import MySqlDb

    db = get_mysql_db(MySqlDb)
    yield db
    cleanup_mysql_db(db)


@pytest.fixture
def db_mysql_notmem(db_mysql):
    yield db_mysql


@pytest.fixture(name="db")
def db_fixture(request, db_name):
    yield request.getfixturevalue("db_" + db_name)


@pytest.fixture(name="db_sqlup", params=["", "_noup"])
def db_sqlup_fixture(request, db_name):
    yield request.getfixturevalue("db_" + db_name + request.param)


@pytest.fixture(name="db_notmem")
def db_notmem_fixture(request, db_name):
    yield request.getfixturevalue("db_" + db_name + "_notmem")


def pytest_generate_tests(metafunc):
    """Converts user-argument --db to fixture parameters."""

    global PYTEST_REG  # pylint: disable=global-statement
    if not PYTEST_REG:
        if any(db in metafunc.fixturenames for db in ("db", "db_notmem", "db_sqlup")):
            db_names = metafunc.config.getoption("db", [])
            db_names = db_names or ["sqlite"]
            for mark in metafunc.definition.own_markers:
                if mark.name == "db":
                    db_names = set(mark.args).intersection(set(db_names))
                    break
            db_names = sorted(db_names)  # xdist compat
            metafunc.parametrize("db_name", db_names, scope="function")
