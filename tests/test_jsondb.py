import gc
import os
from unittest.mock import patch

import pytest

from notanorm import errors as err, open_db, model_from_ddl, JsonDb
from tests.test_notanorm import _persist_schema


def test_no_ddl(db_jsondb):
    db = db_jsondb
    db.insert("t", x=5, y="yo")
    db.insert("t", x=5, y=9)
    db.insert("z", x=b"bytes")

    with pytest.raises(err.UnknownPrimaryError):
        db.update("t", x=5, y=9, z=6)

    assert db.model() == model_from_ddl(
        "create table t (x, y); create table z(x)", "sqlite"
    )

    # this can take a different code path with caching, so check again
    with pytest.raises(err.UnknownPrimaryError):
        db.update("t", x=5, y=9, z=6)


def test_retry_fileop(db_jsondb_notmem):
    db = db_jsondb_notmem
    _persist_schema(db)
    prev = open
    first = True

    def one_err_dump(*a, **k):
        nonlocal first
        if first:
            first = False
            raise PermissionError
        return prev(*a, **k)

    # jsondb attempts to paves over windows permission errors that occur with many processes
    with patch("builtins.open", one_err_dump):
        db.insert("foo", tx="hi")
        db.close()

    uri = db.uri
    db = open_db(uri)
    row = db.select("foo")[0]

    assert row.tx == "hi"


def test_retry_fails_eventually(db_jsondb_notmem, tmp_path):
    db = db_jsondb_notmem
    _persist_schema(db)

    def perm_err(*a, **k):
        raise PermissionError

    # jsondb attempts to pave over windows permission errors that occur with many processes
    with pytest.raises(PermissionError):
        with patch("os.replace", perm_err):
            db.insert("foo", tx="hi")
            db.close()

    # tmp db is cleaned up even tho permission error happened
    assert os.listdir(tmp_path) == []


def test_finalize_on_del(tmp_path):
    # jsondb serializes commits on close/__del__
    db = JsonDb(str(tmp_path / "db"))
    db.insert("foo", tx="hi")
    uri = db.uri
    del db
    gc.collect()
    db = open_db(uri)
    assert db.select("foo")[0].tx == "hi"