import sys
import logging
import pytest

from notanorm import DbModel, DbCol, DbType, DbTable, DbIndex, DbBase
import notanorm.errors as err
from notanorm.model import ExplicitNone

log = logging.getLogger(__name__)

if tuple(sys.version_info[0: 2]) <= (3, 6):
    pytest.skip("sqlglot requires python 3.7 or greateer", allow_module_level=True)


# has to come below the version check above
import sqlglot.errors  # noqa
from notanorm.ddl_helper import model_from_ddl  # noqa


def test_model_ddl_cap(db):
    # creating a model using sqlite results in a model that generally works across other db's
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("auto", typ=DbType.INTEGER, autoinc=True),
                    DbCol(
                        "inty",
                        typ=DbType.INTEGER,
                        autoinc=False,
                        notnull=True,
                        default="4",
                    ),
                    DbCol("blob", typ=DbType.BLOB),
                    DbCol("blob4", typ=DbType.BLOB, size=4, fixed=False),
                    DbCol("tex", typ=DbType.TEXT, notnull=True),
                    DbCol("siz3v", typ=DbType.TEXT, size=3, fixed=False),
                    DbCol("siz3", typ=DbType.TEXT, size=3, fixed=True),
                    DbCol("flt", typ=DbType.FLOAT, default="1.1"),
                    DbCol("dbl", typ=DbType.DOUBLE, default="2.2"),
                ),
                indexes={
                    DbIndex(fields=("auto",), primary=True),
                    DbIndex(fields=("flt",), unique=True),
                },
            )
        }
    )

    # ddl / create + model are the same
    ddl = db.ddl_from_model(model)
    db.executescript(ddl)
    captured_model1 = db.model()

    db.execute("drop table foo")

    db.create_model(model)
    captured_model2 = db.model()
    assert captured_model1 == captured_model2

    extracted = model_from_ddl(ddl, db.uri_name)

    assert extracted == captured_model1


def test_execute_ddl(db: DbBase):
    mod = db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    assert db.simplify_model(db.model()) == db.simplify_model(mod)
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER


def test_execute_ddl_skip_exists(db: DbBase):
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER


def test_execute_ddl_exists(db: DbBase):
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    with pytest.raises(err.TableExistsError):
        db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql", ignore_existing=False)


def test_execute_sqlite(db: DbBase):
    db.execute_ddl("create table foo (bar integer)", "sqlite")
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER


def test_multi_key():
    mod = model_from_ddl("create table foo (bar integer, baz integer, primary key (bar, baz))")
    assert mod["foo"].indexes == {DbIndex(("bar", "baz"), primary=True), }


def test_primary_key():
    mod = model_from_ddl("create table foo (bar integer primary key, baz integer)")
    assert mod["foo"].indexes == {DbIndex(("bar", ), primary=True), }


def test_autoinc():
    mod = model_from_ddl("create table foo (bar integer auto_increment)", "mysql")
    assert mod["foo"].columns == (DbCol("bar", DbType.INTEGER, autoinc=True, size=4),)


def test_default_none():
    mod = model_from_ddl("create table foo (bar text default null)")
    assert mod["foo"].columns == (DbCol("bar", DbType.TEXT, default=ExplicitNone()),)


def test_sqlite_only():
    mod = model_from_ddl("create table foo (bar default 1)")
    assert mod["foo"].columns == (DbCol("bar", DbType.ANY, default="1"),)


def test_primary_key_auto():
    mod = model_from_ddl("create table cars(id integer auto_increment primary key, gas_level double default 1.0);", "mysql")
    assert mod["cars"].columns == (DbCol("id", DbType.INTEGER, autoinc=True, notnull=False, size=4),
                                   DbCol("gas_level", DbType.DOUBLE, default='1.0'))
    assert mod["cars"].indexes == {DbIndex(("id",), primary=True), }


def test_default_bool():
    mod = model_from_ddl("create table foo (bar boolean default TRUE)")
    assert mod["foo"].columns == (DbCol("bar", DbType.BOOLEAN, default=True),)


def test_not_null_pk():
    create = "CREATE TABLE a (id INTEGER, dd TEXT, PRIMARY KEY(id));"
    mod = model_from_ddl(create)
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=False, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex(("id",), primary=True)}


def test_explicit_not_null_pk():
    create = "CREATE TABLE a (id INTEGER NOT NULL, dd TEXT, PRIMARY KEY(id));"
    mod = model_from_ddl(create)
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=True, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex(("id",), primary=True)}


def test_unique_col():
    create = "CREATE TABLE a (id INTEGER NOT NULL, dd TEXT unique);"
    mod = model_from_ddl(create, dialect="mysql")
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=True, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex(("dd",), unique=True)}


def test_default_str():
    mod = model_from_ddl("create table foo (bar text default 'txt')")
    assert mod["foo"].columns == (DbCol("bar", DbType.TEXT, default='txt'),)


def test_err_autoinc(db):
    # for now, this restriction applies to all db's.   could move it to sqlite only, but needs testing
    model = model_from_ddl("create table foo (bar integer auto_increment, baz integer auto_increment)")
    with pytest.raises(err.SchemaError):
        db.create_model(model)


def test_detect_dialect():
    # mysql
    mod = model_from_ddl("create table foo (`bar` integer auto_increment, baz varchar(32))")
    assert mod["foo"].columns == (DbCol("bar", DbType.INTEGER, autoinc=True, size=4),
                                  DbCol("baz", DbType.TEXT, size=32))

    # sqlite
    mod = model_from_ddl("create table foo (\"bar\" integer, baz blob)")
    assert mod["foo"].columns == (DbCol("bar", DbType.INTEGER, size=4), DbCol("baz", DbType.BLOB))


def test_parser_error():
    with pytest.raises(sqlglot.errors.ParseError):
        model_from_ddl("create table foo (bar integer auto_increment, ")
