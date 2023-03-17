# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import copy
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from typing import Generator, Iterator, cast
from unittest.mock import MagicMock, patch

import pytest

import notanorm.errors
import notanorm.errors as err
from notanorm import SqliteDb, DbRow, DbBase, DbType, DbIndex, And, Or, Op, DbIndexField
from notanorm.connparse import open_db, parse_db_uri
from notanorm.jsondb import JsonDb

log = logging.getLogger(__name__)


def test_db_basic(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")
    assert db.query("select bar from foo")[0].bar == "hi"
    assert (
        db.query("select bar from foo where bar=%s" % db.placeholder, "hi")[0].bar
        == "hi"
    )
    assert not db.query("select bar from foo where bar='zz' and 1=1 and 2=2")
    assert not db.query("select bar from foo where bar=%s" % db.placeholder, "ho")


def test_db_delete(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.insert("foo", bar="ho")
    db.delete("foo", bar="hi")
    assert not db.select("foo", bar="hi")
    assert db.select("foo")[0].bar == "ho"


def test_db_col_as(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")
    assert db.query("select bar as x from foo")[0].x == "hi"


def test_db_version(db):
    assert db.version()


def test_db_count(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    assert db.count("foo") == 1
    assert db.count("foo", bar="hi") == 1
    assert db.count("foo", {"bar": "hi"}) == 1
    assert db.count("foo", {"bar": "ho"}) == 0


def test_db_sum(db):
    create_and_fill_test_db(db, 4, "x", bar="integer primary key")

    assert db.count("x") == 4
    assert db.sum("x", "bar") == sum([0, 1, 2, 3])


def test_db_select(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    assert db.select("foo", bar="hi")[0].bar == "hi"
    assert db.select("foo", bar=None) == []
    assert db.select("foo", {"bar": "hi"})[0].bar == "hi"
    assert db.select("foo", {"bar": "ho"}) == []


def test_db_select_not_both_where(db):
    create_and_fill_test_db(db, 2)
    with pytest.raises(ValueError):
        assert db.select("foo", ["bar"], {"bar": 1}, baz=1) == []


def test_db_row_obj__dict__(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    ret = db.select_one("foo")
    assert ret.__dict__ == {"bar": "hi"}
    assert ret._asdict() == {"bar": "hi"}
    assert ret.copy() == ret._asdict()
    assert copy.copy(ret) == ret._asdict()


def test_db_row_obj_case(db):
    db.query("create table foo (Bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    ret = db.select_one("foo")
    assert ret["bar"] == "hi"
    assert ret["BAR"] == "hi"
    assert ret.bar == "hi"
    assert ret.BaR == "hi"
    assert "Bar" in ret.keys()
    assert "bar" not in ret.keys()
    assert "bar" in ret


def skip_json(db):
    if isinstance(db, JsonDb):
        pytest.skip("not supported")


def test_db_order(db):
    db.query("create table foo (bar integer)")
    for i in range(10):
        db.insert("foo", bar=i)

    fwd = db.select("foo", order_by=["bar"])

    assert db.select("foo", order_by="bar desc") == list(reversed(fwd))
    assert next(iter(db.select("foo", order_by="bar desc"))).bar == 9


def test_db_op_gt(db):
    db.query("create table foo (bar integer)")
    db.insert("foo", bar=3)
    db.insert("foo", {"bar": 4})
    db.insert("foo", bar=5)

    assert db.select_one("foo", bar=notanorm.Op(">", 4)).bar == 5

    assert db.select_one("foo", bar=notanorm.Op("<", 4)).bar == 3

    assert {r.bar for r in db.select("foo", bar=notanorm.Op(">=", 4))} == {4, 5}

    assert {r.bar for r in db.select("foo", bar=notanorm.Op("<=", 4))} == {3, 4}


def test_op_internals() -> None:
    assert notanorm.Op(">", 4) == notanorm.Op(">", 4)
    assert notanorm.Op(">", 4) != notanorm.Op("=", 4)
    assert notanorm.Op(">", 4) != notanorm.Op(">", 5)
    assert notanorm.Op(">", 4) != object()

    assert repr(notanorm.Op(">", 4)) == "Op('>', 4)"


def test_db_select_gen_ex(db):
    db.query("create table foo (bar integer)")
    db.insert("foo", bar=1)
    db.insert("foo", bar=2)

    # works normally
    generator = db.select_gen("foo", order_by="bar")
    assert next(iter(generator)).bar == 1

    # it's a generator
    generator = db.select_gen("foox", order_by="bar")
    assert isinstance(generator, Generator)

    # raises error correctly
    with pytest.raises(notanorm.errors.TableNotFoundError):
        for _ in generator:
            raise ValueError

    # errors pass up correctly
    generator = db.select_gen("foo", order_by="bar")
    with pytest.raises(ValueError):
        for _ in generator:
            raise ValueError

    class Foo:
        def __init__(self, bar=None):
            self.bar = bar
            assert False, "not a good foo"

    # bad class
    db.register_class("foo", Foo)
    generator = db.select_gen("foo", order_by="bar")
    with pytest.raises(AssertionError):
        for _ in generator:
            pass


def test_db_select_gen_close_fetch(db):
    create_and_fill_test_db(db, 5)
    with pytest.raises(ValueError):
        generator = db.select_gen("foo")
        one = generator
        next(one)
        # cursor has 5 rows, raising here should close the cursor
        raise ValueError

    # mysql and other db's will fail here if we don't close the cursor
    assert db.select_any_one("foo")


def test_db_select_any_one(db):
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 0, "oth")
    assert db.select_any_one("foo").bar is not None

    assert db.select_any_one("oth") is None
    assert db.select_one("oth") is None


def test_db_tab_not_found(db):
    db.query("create table foo (bar integer)")
    with pytest.raises(notanorm.errors.TableNotFoundError):
        db.select("foox")
    with pytest.raises(notanorm.errors.TableNotFoundError):
        db.execute("drop table foox")


def test_db_row_obj_iter(db):
    db.query("create table foo (Bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    ret = db.select_one("foo")
    for k in ret:
        assert k == "Bar"

    assert "Bar" in ret


def test_db_row_obj_integer_access(db):
    db.query("create table foo (a text, b text, c text)")
    db.insert("foo", a="a", b="b", c="c")

    ret = db.select_one("foo")

    assert ret[0] == "a"
    assert ret[1] == "b"
    assert ret[2] == "c"

    assert len(list(ret.keys())) == 3
    assert len(list(ret.values())) == 3
    assert len(list(ret.items())) == 3


def test_db_class(db):
    db.query("create table foo (bar text)")
    db.query("insert into foo (bar) values (%s)" % db.placeholder, "hi")

    class Foo:
        def __init__(self, bar=None):
            self.bar = bar

    obj = db.select_one("foo", bar="hi", __class=Foo)

    assert type(obj) == Foo

    db.register_class("foo", Foo)

    obj = db.select_one("foo", bar="hi")

    assert obj.bar == "hi"
    assert type(obj) == Foo


def test_db_select_in(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.insert("foo", bar="ho")

    res = [DbRow({"bar": "hi"}), DbRow({"bar": "ho"})]

    assert db.select("foo", ["bar"], {"bar": ["hi", "ho"]}) == res
    assert db.select("foo", bar=["hi", "ho"]) == res


def test_db_select_explicit_field_map(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar="hi")
    db.insert("foo", bar="ho")

    res = [DbRow({"x": "hi"}), DbRow({"x": "ho"})]

    assert db.select("foo", _fields={"x": "bar"}, bar=["hi", "ho"]) == res


def test_db_select_join(db):
    skip_json(db)
    db.query("create table foo (col text, d text)")
    db.query("create table baz (col text, d text)")
    db.insert("foo", col="hi", d="foo")
    db.insert("baz", col="hi", d="baz")
    res = db.select(
        "foo inner join baz on foo.col=baz.col", ["baz.d"], {"foo.col": "hi"}
    )
    assert res[0].d == "baz"


def test_db_update_and_select(db):
    print("sqlite3 version", sqlite3.sqlite_version)

    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # infers where clause from primary key
    db.update("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # alternate interface where the first argument is a where clause dict
    db.update("foo", {"bar": "hi"}, baz="up2")

    assert db.select("foo")[0].baz == "up2"

    # alternate interface where the first argument is a where clause dict (reversed primary)
    db.update("foo", {"baz": "up2"}, bar="yo")

    assert db.select("foo")[0].bar == "yo"

    # alternate interface where the first argument is a where clause dict and second is a update dict
    db.update("foo", {"baz": "up2"}, {"baz": "hi"})

    assert db.select("foo")[0].baz == "hi"

    # alternate interface where the select is explicit
    assert db.select("foo", ["baz"])[0].baz, "hi"


def test_db_upsert(db_sqlup):
    db = db_sqlup
    db.query("create table foo (bar varchar(32) not null primary key, baz text)")
    db.insert("foo", bar="hi", baz="ho")

    assert db.select("foo", bar="hi")[0].bar == "hi"

    # updates
    db.upsert("foo", bar="hi", baz="up")

    assert db.select("foo", bar="hi")[0].baz == "up"

    # inserts
    ret = db.upsert("foo", bar="lo", baz="down")

    if db_sqlup.uri_name == "sqlite":
        assert ret.lastrowid

    assert db.select("foo", bar="hi")[0].baz == "up"
    assert db.select("foo", bar="lo")[0].baz == "down"

    # no-op
    db.upsert("foo", bar="hi")

    # update everything
    db.upsert_all("foo", baz="all")

    assert db.select("foo", bar="lo")[0].baz == "all"
    assert db.select("foo", bar="hi")[0].baz == "all"

    # where clause doen't update, only inserts
    db.upsert("foo", {"bar": "new"}, baz="baznew")

    assert db.select("foo", bar="new")[0].baz == "baznew"

    db.upsert("foo", {"bar": "new"}, _insert_only={"baz": "bazdef"})

    assert db.select("foo", bar="new")[0].baz == "baznew"

    db.upsert("foo", bar="n2", baz="uponly", _insert_only={"baz": "i2"})

    assert db.select("foo", bar="n2")[0].baz == "i2"

    db.upsert("foo", bar="n2", baz="uponly", _insert_only={"baz": "i2"})

    assert db.select("foo", bar="n2")[0].baz == "uponly"


def test_db_insert_no_vals(db):
    db.query("create table foo (bar integer default 1, baz double default 2.2)")
    db.insert("foo")
    assert db.select_one("foo").bar == 1
    assert db.select_one("foo").baz == 2.2


def test_db_insert_lrid(db):
    db.query("create table foo (bar integer auto_increment primary key)")
    ret = db.insert("foo")
    assert ret.lastrowid


def test_db_upsert_lrid(db):
    db.query("create table foo (bar integer auto_increment primary key, baz integer)")
    ret = db.upsert("foo", bar=1, baz=2)
    assert ret.lastrowid


def test_db_update_none_val(db):
    db.query("create table foo (bar integer, baz integer)")
    db.insert("foo", bar=None, baz=2)
    db.insert("foo", bar=2, baz=2)
    assert db.count("foo", bar=None) == 1, "count w none"
    db.update("foo", {"bar": None}, baz=3)
    assert db.select_one("foo", bar=None).baz == 3, "update w none"


def test_db_update_no_tab(db):
    db.query("create table foo (bar integer, baz integer)")
    with pytest.raises(err.TableNotFoundError):
        db.update("wrongtab", {"bar": 1}, baz=2)


def test_tab_exists(db):
    db.query("create table foo (bar integer)")
    with pytest.raises(err.TableExistsError):
        db.query("create table foo (bar integer)")


def test_no_primary(db):
    db.query("create table foo (bar text)")
    with pytest.raises(err.UnknownPrimaryError):
        db.upsert("foo", bar=1)


def test_db_upsert_non_null(db):
    db.query(
        "create table foo (bar varchar(32) not null primary key, baz text, bop text)"
    )
    db.insert("foo", bar="hi", baz="ho", bop="keep")

    # updates baz... but not bop to none
    db.upsert_non_null("foo", bar="hi", baz="up", bop=None)

    assert db.select_one("foo").baz == "up"
    assert db.select_one("foo").bop == "keep"


def test_multi_close(db):
    db.close()
    db.close()

    # noinspection PyMissingConstructor
    class VeryClose(SqliteDb):
        def __init__(self):
            self.close()

    # always safe to call close, even if not initialized
    VeryClose()


@pytest.mark.db("sqlite")
def test_safedb_inmemorydb(db):
    # test that in memory db's are relatively safe
    db.query("create table foo (bar primary key)")
    db.query("insert into foo (bar) values (?)", 0)

    def updater(_i):
        db.query("update foo set bar = bar + ?", 1)

    pool = ThreadPool(processes=100)

    pool.map(updater, range(100))

    assert db.query("select bar from foo")[0].bar == 100


def test_transactions_any_exc(db):
    class TestExc(Exception):
        pass

    db.query("CREATE table foo (bar integer primary key)")
    db.insert("foo", bar=5)
    with pytest.raises(TestExc):
        with db.transaction() as db:
            db.delete_all("foo")
            raise TestExc()
    assert db.select("foo")[0].bar == 5


def test_transactions_deadlock(db):
    def trans_thread(orig_db):
        with orig_db.transaction() as ins_db:
            for ins in range(50, 100):
                ins_db.insert("foo", bar=ins)

    db.query("CREATE table foo (bar integer primary key)")

    thread = threading.Thread(target=trans_thread, args=(db,), daemon=True)
    thread.start()
    for i in range(50):
        db.insert("foo", bar=i)
    thread.join()


def test_select_gen_not_lock(db: DbBase):
    db.query("CREATE table foo (bar integer primary key)")

    with db.transaction():
        for i in range(500):
            db.insert("foo", bar=i)

    thread_result = []

    ev1 = threading.Event()
    ev2 = threading.Event()

    def slow_q():
        nonlocal thread_result
        for row in db.select_gen("foo", order_by="bar desc"):
            thread_result.append(row.bar)
            ev1.set()
            ev2.wait()

    thread = threading.Thread(target=slow_q, daemon=True)
    thread.start()

    ev1.wait()
    with db.transaction():
        for i in range(500, 600):
            db.insert("foo", bar=i)

    ev2.set()

    start = time.time()
    fast_result = []
    for ent in db.select_gen("foo", order_by="bar desc"):
        fast_result.append(ent.bar)
    end = time.time()

    thread.join()

    assert thread_result == list(reversed(range(500)))
    assert fast_result == list(reversed(range(600)))
    # this prevents a slow machine from falsely succeeding
    assert (end - start) < 1


def test_readonly_fail(db, db_name: str):
    db.query("create table foo (bar text, baz integer default 0)")
    db.insert("foo", bar="y1")

    if db_name == "sqlite":
        db.query("PRAGMA query_only=ON;")
    elif db_name == "mysql":
        db.query("SET SESSION TRANSACTION READ ONLY;")
    elif db_name == "jsondb":
        db.read_only = True

    assert db.select("foo")

    with pytest.raises(err.DbReadOnlyError):
        db.insert("foo", bar="y2")

    with pytest.raises(err.DbReadOnlyError):
        db.update("foo", {"bar": "y1"}, baz=2)

    with pytest.raises(err.DbReadOnlyError):
        db.delete("foo", {"bar": "y1"})

    with pytest.raises(err.DbReadOnlyError):
        db.drop("foo")

    with pytest.raises(err.DbReadOnlyError):
        db.rename("foo", "xxx")


def test_missing_column(db):
    db.query("create table foo (bar text)")
    with pytest.raises(err.NoColumnError):
        db.insert("foo", nocol="y2")


def test_timeout_rational(db_notmem):
    db = db_notmem
    assert db.max_reconnect_attempts > 1
    assert db.timeout > 1
    assert db.timeout < 60


def test_db_cursor_can_exec(db):
    # this really shouldn't be a guarantee, imo!
    con = db._conn().cursor()
    con.execute("create table foo(bar integer)")


def test_db_more_than_one(db):
    db.query("create table foo (bar text)")
    db.insert("foo", bar=1)
    db.insert("foo", bar=1)
    with pytest.raises(err.MoreThanOneError):
        assert db.select_one("foo")
    with pytest.raises(err.MoreThanOneError):
        assert db.select_one("foo", bar=1)


def test_db_integ_foreign(db):
    skip_json(db)
    if isinstance(db, SqliteDb):
        db.query("pragma foreign_keys=on;")
    db.query("create table foo (bar integer primary key)")
    db.query("create table zop (bar integer, foreign key (bar) references foo (bar))")
    db.insert("foo", bar=1)
    db.insert("zop", bar=1)
    with pytest.raises(err.IntegrityError):
        db.insert("zop", bar=2)


def test_db_integ_prim(db):
    db.query("create table foo (bar integer primary key)")
    db.insert("foo", bar=1)
    with pytest.raises(err.IntegrityError):
        db.insert("foo", bar=1)


def test_db_integ_notnul(db):
    db.query("create table foo (bar integer not null)")
    db.insert("foo", bar=1)
    with pytest.raises(err.IntegrityError):
        db.insert("foo", bar=None)

    db.query("create table oth (bar integer not null default 4)")
    db.insert("oth")
    assert db.select("oth")[0].bar == 4


def test_db_annoying_col_names(db):
    db.query('create table "group" (bar integer primary key, "group" integer)')
    db.insert("group", bar=1, group=1)
    db.update("group", bar=1, group=1)
    db.upsert("group", bar=1, group=1)
    db.select("group", group=1)


@pytest.mark.db("mysql")
def test_mysql_op_error(db):
    # test that connection errors don't happen when you do stuf like this
    with pytest.raises(notanorm.errors.OperationalError):
        db.query("create table foo (bar text primary key);")


def test_syntax_error(db):
    with pytest.raises(notanorm.errors.OperationalError):
        db.query("create table fo()o (bar text primary key);")


@pytest.mark.db("sqlite")
def test_no_extra_close(db):
    db.query("create table foo (bar integer primary key);")
    db.insert("foo", bar=1)

    mok = MagicMock()

    def newx(*_, **__):
        # mock cursor
        ret = MagicMock()
        ret.fetchall = lambda: []
        ret.fetchone = lambda: None
        ret.close = mok
        return ret

    db.execute = newx
    db.select("foo")
    list(db.select_gen("foo"))
    mok.close.assert_not_called()


@pytest.mark.db("sqlite")
def test_any_col(db):
    db.query("create table foo (bar whatever primary key);")
    db.insert("foo", bar=1)
    assert db.model()["foo"].columns[0].typ == DbType.ANY


def test_uri_parse():
    from notanorm import MySqlDb

    typ, args, kws = parse_db_uri("sqlite:file.db")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {}

    # escaping works
    typ, args, kws = parse_db_uri("mysql:host=whatever,password=\\,\\=::yo")
    assert typ == MySqlDb
    assert kws == {"host": "whatever", "password": ",=::yo"}

    typ, args, kws = parse_db_uri("sqlite://file.db?timeout=5.1")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {"timeout": 5.1}

    typ, args, kws = parse_db_uri("mysql:host=localhost,port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri("mysql:localhost,port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri("mysql://localhost?port=45")

    assert typ == MySqlDb
    assert kws == {"host": "localhost", "port": 45}

    typ, args, kws = parse_db_uri(
        "mysql://localhost?use_unicode=false&autocommit=true&buffered=FaLsE&compress=TrUe"
    )
    assert typ == MySqlDb
    assert kws == {
        "host": "localhost",
        "use_unicode": False,
        "autocommit": True,
        "buffered": False,
        "compress": True,
    }

    typ, args, kws = parse_db_uri("sqlite://file.db?check_same_thread=false")
    assert typ == SqliteDb
    assert args == ["file.db"]
    assert kws == {"check_same_thread": False}

    with pytest.raises(ValueError):
        parse_db_uri("sqlite://file.db?check_same_thread=not_a_bool")


def test_open_db():
    db = open_db("sqlite://:memory:")
    db.execute("create table foo (bar)")


def test_cap_exec(db):
    with db.capture_sql(execute=True) as stmts:
        db.execute("create table foo(inty integer)")
        db.insert("foo", inty=4)

    assert stmts[0] == ("create table foo(inty integer)", ())
    if db.uri_name == "sqlite":
        assert stmts[1] == ('insert into "foo"("inty") values (?)', (4,))


def test_exec_script(db):
    db.executescript(
        """
        create table foo (x integer);
        create table bar (y integer);
    """
    )
    db.insert("foo", x=1)
    db.insert("bar", y=2)


def create_and_fill_test_db(db, num, tab="foo", **fds):
    if not fds:
        fds = {
            "bar": "integer primary key",
            "baz": "integer not null",
            "cnt": "integer default 0",
        }
    fd_sql = ",".join(db.quote_key(fd) + " " + typ for fd, typ in fds.items())
    db.query(f"CREATE table {db.quote_key(tab)} ({fd_sql})")
    for ins in range(num):
        vals = {
            nm: ins
            if typ in ("integer primary key", "integer")
            else 0
            if typ == "integer not null"
            else None
            for nm, typ in fds.items()
        }
        db.insert(tab, **vals)


@pytest.mark.db("sqlite")
def test_sqlite_unsafe_gen(db_notmem):
    db = db_notmem
    create_and_fill_test_db(db, 5)
    db.generator_guard = True
    with pytest.raises(err.UnsafeGeneratorError):
        for row in db.select_gen("foo"):
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)

    # now with embedded generators
    for row in db.select_gen("foo"):
        for row2 in db.select_gen("foo"):
            pass
        with pytest.raises(err.UnsafeGeneratorError):
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)

    # safe to update now
    db.insert("foo", bar=5, baz=6)

    # ok, select inside select
    for _ in db.select_gen("foo"):
        db.select("foo")

    for _ in db.select_gen("foo"):
        list(db.select_gen("foo"))

    # no memory leaks
    assert db._in_gen_size() == 0


@pytest.mark.db("sqlite")
def test_sqlite_guard_thread(db_notmem):
    db = db_notmem
    create_and_fill_test_db(db, 5)
    db.generator_guard = True
    cool = False
    event = threading.Event()

    def updatey():
        nonlocal cool
        try:
            db.upsert("foo", bar=row.bar, baz=row.baz + 1)
            cool = True
        finally:
            event.set()

    for row in db.select_gen("foo"):
        threading.Thread(target=updatey, daemon=True).start()
        assert event.wait(3)
        break

    assert cool


def test_subq(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(db.select("foo", bar=db.subq("oth", ["bar"], bar=[1, 3]), baz=0)) == 2


def test_subq_wildcards(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    assert (
        len(db.select(db.subq("foo", {"bing": "bar", "baz": "baz"}, bar=[1, 3]), baz=0))
        == 2
    )


def test_nested_subq(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(
        db.select(
            db.subq(db.subq("foo", bar=db.subq("oth", ["bar"], bar=[1, 3])), baz=0)
        )
    )


def test_subq_limited_fields_join(db):
    skip_json(db)
    # use weird field name on purpose
    create_and_fill_test_db(db, 5, select="integer primary key", baz="integer")
    create_and_fill_test_db(db, 5, "oth")

    # no limit
    db.select(db.join(db.subq("foo", select=[1, 3]), "oth", baz="baz"))

    # subq limited to just 'select', so we can't join on 'baz', even tho you can without the limit
    with pytest.raises(err.UnknownColumnError):
        db.select(
            db.join(db.subq("foo", _fields=["select"], select=[1, 3]), "oth", baz="baz")
        )

    # 'select' as a field name works tho
    db.select(
        db.join(db.subq("foo", _fields=["select"], select=[1, 3]), "oth", select="bar")
    )


def test_joinq_ambig_unknown_col_join(db):
    skip_json(db)
    create_and_fill_test_db(db, 5, "a")
    create_and_fill_test_db(db, 5, "b")
    create_and_fill_test_db(db, 5, "c")

    with pytest.raises(err.UnknownColumnError):
        # nested join... selecting "bar" automagically selects "bar" from the underlying table
        # but it's ambiguous, so it should not succeed
        db.select(db.join(db.join("a", "b", bar="bar"), "c", bar="bar"))

    # being specific is fine
    db.select(db.join(db.join("a", "b", bar="bar"), "c", a__bar="bar"))

    # unknown col error 'bzz'
    with pytest.raises(err.UnknownColumnError):
        db.select(db.join(db.join("a", "b", bar="bar"), "c", bzz="bar"))


def test_joinq_non_ambig_col(db):
    skip_json(db)
    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")
    create_and_fill_test_db(db, 5, "c", cid="integer primary key", did="integer")

    assert db.select(db.join(db.join("a", "b", bid="bid"), "c", cid="cid"))


def test_joinq_model_changes(db_sqlite_notmem):
    db = db_sqlite_notmem

    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")

    assert db.select(db.join("a", "b", bid="bid"))

    # other process has changed things!
    db._conn().execute("drop table b")
    db._conn().execute("create table b (b_id integer primary_key, c_id integer)")
    db._conn().execute("insert into b values (1, 1)")

    with pytest.raises(err.OperationalError):
        db.select(db.join("a", "b", bid="b_id"))

    # explicit table names is always ok
    assert db.select(db.join("a", "b", a__bid="b__b_id"), ["a.aid", "b.b_id"])

    # asterisk is cool too
    assert db.select(db.join("a", "b", a__bid="b__b_id"), ["a.*"])[0].aid == 1

    # or you can clear the cache
    db.clear_model_cache()

    db.select(db.join("a", "b", bid="b_id"))


def test_select_star_in_child(db_sqlite_notmem: DbBase) -> None:
    db = db_sqlite_notmem

    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")

    log.debug("No fields")
    res = db.select(db.join("a", "b", bid="bid"))
    assert res[2].aid
    log.debug("%s", res)

    log.debug("Single field")
    res = db.select(db.join("a", "b", bid="bid"), _fields=["a.aid"])
    log.debug("%s", res)
    assert res[2].aid
    assert "cid" not in res[0]

    log.debug("Single in join field")
    res = db.select(db.join("a", "b", bid="bid", _fields=["a.aid"]))
    log.debug("%s", res)
    assert res[2].aid
    assert "cid" not in res[0]

    log.debug("Wildcard field")
    res = db.select(db.subq(db.join("a", "b", bid="bid"), _fields=["a.*"]))
    log.debug("%s", res)
    assert res[2].bid

    log.debug("Wildcard in join fields")
    res = db.select(db.subq(db.join("a", "b", bid="bid", _fields=["a.*"])))
    log.debug("%s", res)
    assert res[2].bid


def test_joinq_left(db):
    skip_json(db)
    create_and_fill_test_db(db, 5, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 3, "b", bid="integer primary key", cid="integer")

    assert "left" in db.left_join("a", "b", bid="bid").sql

    rows = db.select(db.left_join("a", "b", bid="bid"))

    assert len(rows) == 5
    assert len(list(row for row in rows if row.cid is not None)) == 3


def test_joinq_right(db_mysql):
    # sqlite does not support right joins
    db = db_mysql
    create_and_fill_test_db(db, 3, "a", aid="integer primary key", bid="integer")
    create_and_fill_test_db(db, 5, "b", bid="integer primary key", cid="integer")

    assert "right" in db.right_join("a", "b", bid="bid").sql

    rows = db.select(db.right_join("a", "b", bid="bid"))

    assert len(rows) == 5
    assert len(list(row for row in rows if row.aid is not None)) == 3


def test_select_subq(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    assert len(db.select(db.subq("foo", bar=[1, 3]), bar=1)) == 1


def test_join_simple(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    assert len(db.select(db.join("foo", "oth", bar="bar"), {"foo.bar": 1})) == 1
    assert db.select_one(db.join("foo", "oth", bar="bar"), bar=1) == {
        "bar": 1,
        "foo.baz": 0,
        "foo.cnt": None,
        "oth.baz": 0,
        "oth.cnt": None,
    }


def test_join_subqs(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    res = db.select(
        db.join("foo", db.subq("oth", bar=[1, 3], _alias="oth"), bar="bar"),
        {"foo.bar": 1},
    )
    assert len(res) == 1
    # bar is resolved because it's in the on clause
    # everything else is qualified, because ambig
    assert res == [
        {
            "bar": 1,
            "foo.baz": 0,
            "foo.cnt": None,
            "oth.bar": 1,
            "oth.baz": 0,
            "oth.cnt": None,
        }
    ]


def test_join_subqs_quoting(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, ";oth")
    res = db.select(
        db.join("foo", db.subq(";oth", bar=[1, 3], _alias=";oth"), bar="bar"),
        {"foo.bar": 1},
    )
    assert len(res) == 1
    # bar is resolved because it's in the on clause
    # everything else is qualified, because ambig
    assert res == [
        {
            "bar": 1,
            "foo.baz": 0,
            "foo.cnt": None,
            ";oth.bar": 1,
            ";oth.baz": 0,
            ";oth.cnt": None,
        }
    ]


def test_subqify_join(db):
    skip_json(db)
    create_and_fill_test_db(db, 5, "x", xid="integer primary key", yid="integer")
    create_and_fill_test_db(db, 5, "y", yid="integer primary key", zid="integer")
    create_and_fill_test_db(
        db, 5, "z", zid="integer primary key", oth="integer default 0"
    )
    j1 = db.join("x", "y", yid="yid")
    assert len(db.select(j1, xid=1)) == 1
    sub = db.subq(j1, xid=[1, 3])
    assert len(db.select(sub, xid=1)) == 1
    row = db.select_one(sub, xid=1)
    assert row.zid == 1
    j2 = db.join(sub, "z", zid="zid")
    assert len(db.select(j2, xid=1)) == 1


def test_multi_join_nested_left_right(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    create_and_fill_test_db(db, 5, "thrd")
    create_and_fill_test_db(db, 5, "mor")
    j1 = db.join("foo", "oth", bar="bar")
    log.debug("j1: %s", j1.sql)
    assert len(db.select(j1, {"foo.bar": 1})) == 1
    j2a = db.join(j1, "thrd", on={"foo.bar": "thrd.bar"})
    log.debug("j2a: %s", j2a.sql)
    assert len(db.select(j2a, {"foo.bar": 1})) == 1
    j2b = db.join("thrd", j1, bar="foo.bar")
    log.debug("j2b: %s", j2b.sql)
    assert len(db.select(j2b, {"thrd.bar": 1})) == 1
    j3a = db.join(j2a, "mor", on={"foo.bar": "mor.bar"})
    log.debug("j3a: %s", j3a.sql)
    assert len(db.select(j3a, {"foo.bar": 1})) == 1
    j3b = db.join("mor", j2b, bar="thrd.bar")
    log.debug("j3b: %s", j3b.sql)
    assert len(db.select(j3b, {"thrd.bar": 1})) == 1


def test_join_explicit_mappings(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    j1 = db.subq(
        db.join("foo", "oth", bar="bar", _fields={"z": "oth.bar", "x": "foo.bar"})
    )
    j2 = db.join("foo", j1, bar="x")
    assert db.select_one(j2, z=1).x == 1


def test_join_explicit_fields(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    j1 = db.subq(db.join("foo", "oth", bar="bar", _fields=["oth.bar"]))
    assert j1.fields
    j2 = db.join("foo", j1, bar="bar")
    assert j2.sql
    assert j2.where_map
    assert db.select_one(j2, bar=1).bar == 1


def test_subq_field_map(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    j1 = db.subq("foo", _fields={"xxx": "bar"}, bar=[1, 2])
    assert db.select_one(j1, xxx=1).xxx == 1


def test_join_2_subqs_same_tab(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    s1 = db.subq("foo", bar=[1, 2, 3])
    s2 = db.subq("foo", bar=[2, 3, 4])
    jn = db.join(s1, s2, bar="bar")
    assert len(db.select(jn)) == 2


def test_join_2_subqs(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    s1 = db.subq("foo", bar=[1, 2, 3])
    s2 = db.subq("foo", bar=[2, 3, 4])
    jn = db.join(s1, s2, bar="bar")
    assert len(db.select(jn)) == 2


def test_multi_join_auto_left(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    create_and_fill_test_db(db, 5, "thrd")

    j1 = db.join("foo", "oth", bar="bar")
    j2a = db.join("thrd", j1, bar="foo.bar")

    assert len(db.select(j2a, {"foo.bar": 1})) == 1
    assert len(db.select(j2a, foo__bar=1)) == 1


def test_join_fd_names(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    create_and_fill_test_db(db, 5, "oth")
    # this creates a mapping of fields
    j1 = db.join("foo", "oth", bar="bar")
    # this uses the mapping and just picks one
    row = db.select_one(j1, bar=1)
    log.debug("row: %s", row)
    assert row.bar == 1
    assert row.foo__baz == 0
    row.foo__baz = 4
    assert row.foo__baz == 4
    assert row["foo.baz"] == 4


def test_where_or(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    db.update("foo", bar=3, baz=2)
    assert len(db.select("foo", _where=[{"bar": 1}, {"baz": 2}])) == 2
    db.delete("foo", [{"bar": 1}, {"baz": 2}])
    assert len(db.select("foo")) == 3


def test_warn_dup_index(db, caplog):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    db.query("create index ix_1 on foo(bar);")
    db.query("create index ix_2 on foo(bar);")
    caplog.clear()
    # grabbing the model spits out an annoying warning
    db.model()
    assert "WARNING" in caplog.text


def test_where_complex(db):
    skip_json(db)
    create_and_fill_test_db(db, 5)
    assert (
        len(db.select("foo", _where=And([{"bar": Op(">", 1)}, {"bar": Op("<", 5)}])))
        == 3
    )
    assert (
        len(
            db.select(
                "foo",
                _where=[{"bar": 1}, And([{"bar": Op(">", 1)}, {"bar": Op("<", 5)}])],
            )
        )
        == 4
    )
    assert len(db.select("foo", _where=And([{"bar": db.subq("foo", ["bar"])}]))) == 5

    assert (
        len(db.select("foo", _where=And([{"bar": [1, 2, 3]}, {"bar": Op("<", 3)}])))
        == 2
    )
    assert (
        len(
            db.select(
                "foo",
                _where=And([{"bar": [1, 2, 3]}, Or(({"bar": [2]}, {"baz": [0]}))]),
            )
        )
        == 3
    )


def test_del_raises(db):
    create_and_fill_test_db(db, 5)
    db.delete("foo", bar=2)
    assert len(db.select("foo")) == 4
    with pytest.raises(ValueError):
        db.delete("foo")
    with pytest.raises(ValueError):
        db.delete("foo", {"bar": 3}, baz=0)


def test_db_direct_clone(db_notmem):
    db = db_notmem.clone()
    db.query("CREATE table foo (bar integer primary key)")
    db_notmem.insert("foo", bar=1)


def test_db_uri_clone(db_notmem):
    db = open_db(db_notmem.uri)
    db.query("CREATE table foo (bar integer primary key)")
    db_notmem.insert("foo", bar=1)


def test_quote_key(db: DbBase) -> None:
    from_inst = db.quote_key("key")
    assert from_inst
    assert from_inst != "key"

    # Check that quote_key is available as a classmethod on all impls
    assert type(db).quote_key("key") == from_inst


def test_quote_special(db: DbBase) -> None:
    from_inst = db.quote_keys("*")
    assert from_inst == "*"

    from_inst = db.quote_keys("events.*")
    assert from_inst == db.quote_key("events") + ".*"

    from_inst = db.quote_keys("count(*)")
    assert from_inst == "count(*)"


def test_db_larger_types(db):
    skip_json(db)
    db.query("create table foo (bar mediumblob)")
    # If mediumblob is accidentally translated to blob, the max size in mysql is
    # 2**16. If mysql is running in strict mode, the insert will fail.
    # Otherwise, the comparison will fail.
    db.query("insert into foo (bar) values (%s)" % db.placeholder, b"a" * (2**16 + 4))
    assert db.query("select bar from foo")[0].bar == b"a" * (2**16 + 4)


def test_db_larger_type_from_model(db):
    # same as test above, but doesn't rely on mysql syntax to work for every db type
    schema_model = notanorm.model_from_ddl("create table foo (bar mediumblob)", "mysql")
    db.create_model(schema_model)
    db.query("insert into foo (bar) values (%s)" % db.placeholder, b"a" * (2**16 + 4))
    assert db.query("select bar from foo")[0].bar == b"a" * (2**16 + 4)


def test_limit_rowcnt(db: DbBase):
    create_and_fill_test_db(db, 5)
    assert len(db.select("foo", _limit=3)) == 3
    assert len(db.select("foo", _limit=3, order_by="bar desc")) == 3
    assert len(list(db.select_gen("foo", _limit=1, order_by="bar desc"))) == 1
    assert len(list(db.select_gen("foo", _limit=0, order_by="bar desc"))) == 0

    if db.uri_name != "mysql":
        # mysql doesn't support limits in where subqueries.  probably you shouldn't use them then,
        # if you want stuff to be compat
        assert (
            len(db.select("foo", bar=db.subq("foo", ["bar"], bar=[1, 2, 3], _limit=2)))
            == 2
        )


def test_limit_offset(db: DbBase):
    create_and_fill_test_db(db, 5)
    assert len(db.select("foo", _limit=(1, 3))) == 3
    assert db.select("foo", _limit=(2, 2), order_by="bar")[0].bar == 2
    assert len(db.select("foo", _limit=(1, 3), order_by="bar desc")) == 3
    assert list(db.select_gen("foo", _limit=(0, 3), order_by="bar desc"))[0].bar == 4
    assert len(list(db.select_gen("foo", _limit=(4, 0), order_by="bar desc"))) == 0


def create_group_tabs(db):
    create_and_fill_test_db(db, 0, "a", f1="integer", f2="integer", f3="integer")

    db.insert("a", f1=1, f2=1, f3=2)
    db.insert("a", f1=1, f2=1, f3=2)
    db.insert("a", f1=1, f2=2, f3=3)
    db.insert("a", f1=1, f2=2, f3=3)
    db.insert("a", f1=1, f2=2, f3=3)
    db.insert("a", f1=2, f2=3, f3=1)
    db.insert("a", f1=2, f2=4, f3=2)
    db.insert("a", f1=2, f2=4, f3=2)


def test_raw_fields_group_by(db: DbBase):
    skip_json(db)
    create_group_tabs(db)
    ret = db.select(
        "a",
        {"cnt": "count(*)", "ver": "max(f3)"},
        {},
        _group_by=["f1", "f2"],
        _order_by=["cnt"],
        _limit=3,
    )
    # check limit
    assert len(ret) == 3
    # check count == f3
    assert all(r.cnt == r.ver for r in ret)
    # check order by
    assert all(e.cnt <= l.cnt for e, l in zip(ret, ret[1:]))


def test_subq_group_by(db: DbBase):
    skip_json(db)
    create_group_tabs(db)
    sub = db.subq(
        "a", {"cnt": "count(*)", "ver": "max(f3)"}, {}, _group_by=["f1", "f2"]
    )
    ret = db.select(sub, ver=2)
    assert ret == [{"cnt": 2, "ver": 2}, {"cnt": 2, "ver": 2}]


def test_group_by_subq(db: DbBase):
    skip_json(db)
    create_group_tabs(db)
    sub = db.subq("a", f1=1)
    ret = db.select(
        sub, {"cnt": "count(*)", "ver": "max(f3)", "f2": "f2"}, {}, _group_by=["f2"]
    )
    assert ret == [{"cnt": 2, "ver": 2, "f2": 1}, {"cnt": 3, "ver": 3, "f2": 2}]


def test_agg_group_by(db: DbBase):
    skip_json(db)
    create_group_tabs(db)

    # group by one col == dict with col index into counts
    assert db.count("a", _group_by=["f1"]) == {1: 5, 2: 3}

    # limit/order is ok in counts, maybe you have a lot of counts
    assert db.count("a", _group_by=["f1"], _limit=1, _order="asc") == {2: 3}
    assert db.count("a", _group_by=["f1"], _limit=1, _order="desc") == {1: 5}

    # group by 2 cols == dict with tuple index into counts
    assert db.count("a", _group_by=["f1", "f2"]) == {
        (1, 1): 2,
        (1, 2): 3,
        (2, 3): 1,
        (2, 4): 2,
    }

    # group by 2 cols, sum
    assert db.sum("a", "f3", _group_by=["f1", "f2"]) == {
        (1, 1): 4,
        (1, 2): 9,
        (2, 3): 1,
        (2, 4): 4,
    }

    # multi aggregate
    ret = db.aggregate(
        "a", {"sum": "sum(f3)", "cnt": "count(*)"}, _group_by=["f1", "f2"]
    )

    assert ret == {
        (1, 1): {"sum": 4, "cnt": 2},
        (1, 2): {"sum": 9, "cnt": 3},
        (2, 3): {"sum": 1, "cnt": 1},
        (2, 4): {"sum": 4, "cnt": 2},
    }

    # simple max is easy!
    assert db.aggregate("a", "max(f3)", f1=1) == 3

    with pytest.raises(ValueError):
        # no mix where
        db.aggregate(
            "a",
            {"sum": "sum(f3)", "cnt": "count(*)"},
            {"f2": 2},
            f1=1,
            _group_by=["f1", "f2"],
        )


def test_type_translation_mysql_dialect(db: DbBase):
    skip_json(db)
    # mysql-compatible types that can be used with sqlite
    schema = """
        CREATE table foo (
            a text,
            b longtext,
            c mediumtext,
            d integer,
            e tinyint,
            f smallint,
            g bigint,
            h int primary key
        )
        """

    schema_model = notanorm.model_from_ddl(schema, "mysql")

    db.execute(schema)
    exec_model = db.model()

    assert db.simplify_model(exec_model) == db.simplify_model(schema_model)


def test_quote_group_by(db: DbBase) -> None:
    skip_json(db)
    schema = """
        CREATE table foo (
            a integer primary key not null,
            `;evil` text not null,
        )
        """

    evil = ";evil"

    schema_model = notanorm.model_from_ddl(schema, "mysql")
    db.create_model(schema_model)

    exp = [{"a": 1, evil: "yo"}, {"a": 2, evil: "yoyo"}, {"a": 3, evil: "yoyo"}]
    for row in exp:
        db.insert("foo", row)
    assert len(db.select("foo")) == 3

    exp_agg = [{evil: "yo", ";eviler": 1}, {evil: "yoyo", ";eviler": 2}]

    # Check that all columns are properly quoted
    res = db.select("foo", {evil: evil, ";eviler": "COUNT(*)"}, {}, _group_by=";evil")
    res.sort(key=lambda x: cast(str, x[evil]))
    assert res == exp_agg

    # Even when table names are included.
    res = db.select(
        "foo", {evil: f"foo.{evil}", ";eviler": "COUNT(*)"}, {}, _group_by="foo.;evil"
    )
    res.sort(key=lambda x: cast(str, x[evil]))
    assert res == exp_agg

    # And even in subqueries.
    res = db.select(
        db.subq(
            "foo",
            {evil: f"foo.{evil}", ";eviler": "COUNT(*)"},
            {},
            _group_by="foo.;evil",
        )
    )
    res.sort(key=lambda x: cast(str, x[evil]))
    assert res == exp_agg


def test_quote_subq_alias(db: DbBase) -> None:
    skip_json(db)
    schema = """
        CREATE table foo (
            a integer primary key not null,
            `;evil` text not null,
        )
        """

    evil = ";evil"

    schema_model = notanorm.model_from_ddl(schema, "mysql")
    db.create_model(schema_model)

    exp = [{"a": 1, evil: "yo"}, {"a": 2, evil: "yoyo"}, {"a": 3, evil: "yoyo"}]
    for row in exp:
        db.insert("foo", row)
    assert len(db.select("foo")) == 3

    exp_fields = [{k: v for (k, v) in row.items() if k in (evil,)} for row in exp]

    # Check that the subq's alias is properly quoted.
    assert (
        db.select(db.subq("foo", [evil], _alias=";eviler", _order_by="a ASC"), [evil])
        == exp_fields
    )

    # Check . quoting -- the whole alias should be quoted, but the field should be smart-quoted.
    assert (
        db.select(
            db.subq("foo", [f"foo.{evil}"], _alias="foo.;eviler", _order_by="a ASC"),
            [evil],
        )
        == exp_fields
    )


def test_quote_order_by(db: DbBase) -> None:
    schema = """
        CREATE table foo (
            a integer primary key not null,
            `;evil` integer not null,
        )
        """

    evil = ";evil"

    schema_model = notanorm.model_from_ddl(schema, "mysql")
    db.create_model(schema_model)

    exp_asc = [
        {"a": 500, evil: 1},
        {"a": 2, evil: 2},
        {"a": 30, evil: 3},
    ]

    for row in exp_asc:
        db.insert("foo", row)
    assert len(db.select("foo")) == 3

    assert db.select("foo", _order_by=f"{evil} ASC") == exp_asc
    assert db.select("foo", _order_by=f"{evil}    ASC ") == exp_asc
    assert db.select("foo", _order_by=f"{evil} DESC") == list(reversed(exp_asc))
    # If ASC/DESC isn't specified, the order is implementation-defined.
    assert db.select("foo", _order_by=f"{evil}") in (exp_asc, list(reversed(exp_asc)))

    # Multiple columns.
    assert db.select("foo", _order_by=[f"{evil} ASC", "a"]) == exp_asc
    assert db.select("foo", _order_by=[f"{evil}    ASC ", "a"]) == exp_asc
    assert db.select("foo", _order_by=[f"{evil} DESC", "a"]) == list(reversed(exp_asc))
    assert db.select("foo", _order_by=[f"{evil}", "a"]) in (
        exp_asc,
        list(reversed(exp_asc)),
    )

    # Correctly smart-quotes if table is included.
    assert db.select("foo", _order_by=f"foo.{evil} ASC") == exp_asc
    assert db.select("foo", _order_by=f"foo.{evil}    ASC ") == exp_asc
    assert db.select("foo", _order_by=f"foo.{evil} DESC") == list(reversed(exp_asc))
    assert db.select("foo", _order_by=f"foo.{evil}") in (
        exp_asc,
        list(reversed(exp_asc)),
    )

    assert db.select("foo", _order_by=[f"foo.{evil} ASC", "foo.a"]) == exp_asc
    assert db.select("foo", _order_by=[f"foo.{evil}    ASC ", "foo.a"]) == exp_asc
    assert db.select("foo", _order_by=[f"foo.{evil} DESC", "foo.a"]) == list(
        reversed(exp_asc)
    )
    assert db.select("foo", _order_by=[f"foo.{evil}", "foo.a"]) in (
        exp_asc,
        list(reversed(exp_asc)),
    )

    skip_json(db)  # no subq, join

    # While we're here, let's test that it works with joins, subqueries, and all that jazz.
    exp_subq_asc = [
        {"a": 500, "foo.;evil": 1, "subq.a": 500, "subq.;evil": 1},
        {"a": 2, "foo.;evil": 2, "subq.a": 2, "subq.;evil": 2},
        {"a": 30, "foo.;evil": 3, "subq.a": 30, "subq.;evil": 3},
    ]

    exp_subq_desc = list(reversed(exp_subq_asc))

    assert (
        db.select(
            db.join(
                "foo",
                db.subq("foo", _order_by=f"foo.{evil} ASC", _alias="subq"),
                on={"a": "a"},
            ),
            _order_by=f"subq.{evil} ASC",
        )
        == exp_subq_asc
    )

    assert (
        db.select(
            db.join(
                "foo",
                db.subq("foo", _order_by=f"foo.{evil} DESC", _alias="subq"),
                on={"a": "a"},
            ),
            _order_by=f"subq.{evil} ASC",
        )
        == exp_subq_asc
    )

    assert (
        db.select(
            db.join(
                "foo",
                db.subq("foo", _order_by=f"foo.{evil} DESC", _alias="subq"),
                on={"a": "a"},
            ),
            _order_by=f"subq.{evil} DESC",
        )
        == exp_subq_desc
    )

    assert (
        db.select(
            db.join(
                "foo",
                db.subq("foo", _order_by=f"foo.{evil} ASC", _alias="subq"),
                on={"a": "a"},
            ),
            _order_by=f"subq.{evil} DESC",
        )
        == exp_subq_desc
    )


def test_clob_invalid():
    schema = """
        CREATE table no_clob (
            a clob
        )
        """

    with pytest.raises(ValueError):
        _ = notanorm.model_from_ddl(schema, "sqlite")


def test_rename_drop(db):
    db.execute("create table foo (bar int)")
    db.insert("foo", bar=1)
    db.rename("foo", "foo2")
    db.insert("foo2", bar=1)
    assert sum(r.bar for r in db.select("foo2")) == 2
    with pytest.raises(err.TableNotFoundError):
        db.drop("foo")
    db.drop("foo2")
    with pytest.raises(err.TableNotFoundError):
        db.select("foo2")
    with pytest.raises(err.TableNotFoundError):
        db.rename("foo2", "bazz")


def test_drop_index(db):
    skip_json(db)  # no "create index" stmt
    db.execute("create table foo (bar integer)")
    db.execute("create unique index ix_foo_uk on foo(bar)")
    assert db.model()["foo"].indexes.pop().name
    idx = DbIndex(fields=(DbIndexField("bar"),), unique=True)
    # simplified constructor
    idx2 = DbIndex.from_fields(["bar"], unique=True)
    assert idx == idx2
    assert not idx.name
    assert "ix_foo_uk" == db.get_index_name("foo", idx)
    db.drop_index("foo", idx)
    assert not db.model()["foo"].indexes

    # If an index can't be found, just return None as the name.
    assert db.get_index_name("foo", idx) is None


def test_drop_ddl(db):
    model = notanorm.model_from_ddl(
        """
        create table foo (bar integer);
        create unique index ix_foo_uk on foo(bar);
    """
    )
    db.create_model(model)
    idx = DbIndex(fields=(DbIndexField("bar"),), unique=True)
    nam = db.get_index_name("foo", idx)
    assert nam
    ddl = f"drop index {nam}"
    if db.uri_name == "mysql":
        ddl += " on foo"
    db.execute(ddl)
    assert not db.get_index_name("foo", idx)

    with pytest.raises(err.OperationalError):
        db.execute(ddl)

    with pytest.raises(err.OperationalError):
        db.drop_index_by_name("foo", nam)

    db.execute("drop table foo")
    with pytest.raises(err.TableNotFoundError):
        db.select("foo")

    with pytest.raises(err.TableNotFoundError):
        db.execute("drop table foo")

    with pytest.raises(err.DbError):
        db.drop_index_by_name("foo", nam)


def _sample_model() -> notanorm.DbModel:
    schema = """
        create table foo (
            pk integer primary key auto_increment,
            b integer,
            c integer not null,
            d integer
        );

        create index ix_foo_b on foo(b);
        create unique index ix_foo_c on foo(c);
        create unique index ix_foo_d on foo(d);

        create table bar (
            a integer,
            b integer
        );

        create index ix_bar_a on bar(a);
        """

    return notanorm.model_from_ddl(schema, "mysql")


class _CustomExc(Exception):
    pass


@contextmanager
def _raise_on_model_fetch(db: DbBase) -> Iterator[MagicMock]:
    with patch.object(db, "model", side_effect=_CustomExc, autospec=True) as m:
        yield m


def test_create_model_explicit_model(db: DbBase) -> None:
    schema_model = _sample_model()

    existing_model = db.model()

    # Confirm that None is treated separate from falsey values.
    assert dict(existing_model) == {}

    # Check that if we pass an existing model, we don't call model().
    with _raise_on_model_fetch(db):
        # Sanity check: patch is wired up properly.
        with pytest.raises(_CustomExc):
            db.model()

        db.create_model(schema_model, existing_model=existing_model)

    assert db.simplify_model(schema_model) == db.simplify_model(db.model())


def test_create_model_cached_model(db: DbBase) -> None:
    schema_model = _sample_model()

    # Initial call fills the model cache.
    db.create_model(schema_model, ignore_existing=True)
    assert schema_model == db._get_cached_model()
    assert db.simplify_model(db.model()) == db.simplify_model(db._get_cached_model())

    # Subsequent calls to create_model() don't re-call model().
    with _raise_on_model_fetch(db):
        # Sanity check: patch is wired up properly.
        with pytest.raises(_CustomExc):
            db.model()

        db.create_model(schema_model, ignore_existing=True)
        db.create_model(schema_model, ignore_existing=True)

    assert db.simplify_model(schema_model) == db.simplify_model(db.model())

    # If an exception is raised while executing, cache is cleared.
    # Drop an index outside of notanorm's purview.
    idx = db.get_index_name(
        "foo", next(ind for ind in db.model()["foo"].indexes if not ind.primary)
    )
    with db.capture_sql(execute=False) as sql:
        db.drop_index_by_name("foo", idx)

    assert db.model()["foo"].indexes == schema_model["foo"].indexes

    for stmt in sql:
        log.info("Secretly executing: %s", stmt)
        db._cursor(db._conn()).execute(*stmt)

    # Because of the stale cache, we don't recreate the dropped index.
    with _raise_on_model_fetch(db):
        db.create_model(schema_model, ignore_existing=True)
    assert db.model()["foo"].indexes != schema_model["foo"].indexes

    # Simulate an arbitrary exception.
    with _raise_on_model_fetch(db):
        with patch.object(db, "create_table", side_effect=err.OperationalError):
            with pytest.raises(err.OperationalError):
                db.create_model(schema_model, ignore_existing=True)

    # Try again: this time, we refetch the model and recreate the index.
    db.create_model(schema_model, ignore_existing=True)
    assert db.model()["foo"].indexes == schema_model["foo"].indexes


def test_create_model_empty_model(db: DbBase) -> None:
    # If we pass in an empty model, we don't fetch the model or do any other work.
    with _raise_on_model_fetch(db):
        db.create_model(notanorm.DbModel())


def test_create_table_and_indexes(db: DbBase) -> None:
    schema_model = _sample_model()

    # If we pass create_indexes=False, that's honored.
    for name, schema in schema_model.items():
        db.create_table(name, schema, ignore_existing=False, create_indexes=False)

    assert db.simplify_model(db.model()) != db.simplify_model(schema_model)

    # If we pass nothing, default is create_indexes=True and they do get created.
    for name, schema in schema_model.items():
        db.create_table(name, schema, ignore_existing=True)

    assert db.simplify_model(db.model()) == db.simplify_model(schema_model)


def _persist_schema(db):
    schema = """
        CREATE table foo (
            tx text,
            xin integer,
            fl double,
            xby blob,
            boo boolean
        )
        """
    db.execute_ddl(schema)


def test_db_persist(db_notmem):
    db = db_notmem
    _persist_schema(db)
    db.insert("foo", tx="hi", xin=4, fl=3.2, xby=b"dd", boo=True)
    uri = db.uri
    db.close()
    db = open_db(uri)
    row = db.select("foo")[0]
    assert row.tx == "hi"
    assert row.xin == 4
    assert row.fl == 3.2
    assert row.xby == b"dd"
    assert row.boo


def test_db_persist_exec(db_notmem):
    db = db_notmem
    _persist_schema(db)
    db.execute(
        f"insert into foo (tx, xin, fl, xby, boo) values ({db.placeholder}, 4, 3.2, {db.placeholder}, True)",
        ("hi", b"dd"),
    )
    uri = db.uri
    db.close()
    db = open_db(uri)
    row = db.select("foo")[0]
    assert row.tx == "hi"
    assert row.xin == 4
    assert row.fl == 3.2
    assert row.xby == b"dd"
    assert row.boo


def test_init_ddl():
    db = JsonDb(":memory:", ddl="create table foo (bar integer)")
    db.insert("foo", bar=4)
    db.drop("foo")
