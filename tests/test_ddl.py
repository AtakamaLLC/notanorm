import sys
import logging
import pytest

from notanorm import DbModel, DbCol, DbType, DbTable, DbIndex, DbBase, DbIndexField
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
                    DbIndex(fields=(DbIndexField("auto"),), primary=True),
                    DbIndex(fields=(DbIndexField("flt"),), unique=True),
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


def test_ddl_sqlite_primary_key_autoinc(db: DbBase):
    mod = db.execute_ddl("create table foo (bar integer primary key)", "sqlite")
    assert db.simplify_model(db.model()) == db.simplify_model(mod)
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER
    assert db.simplify_model(db.model())["foo"].columns[0].autoinc


def test_execute_ddl_skip_exists(db: DbBase):
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    db.execute_ddl("""
        create table foo (bar integer auto_increment primary key);
        create table baz (bar integer auto_increment primary key);
    """, "mysql")
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER
    assert db.simplify_model(db.model())["baz"].columns[0].typ == DbType.INTEGER


def test_execute_ddl_exists(db: DbBase):
    db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql")
    with pytest.raises(err.TableExistsError):
        db.execute_ddl("create table foo (bar integer auto_increment primary key)", "mysql", ignore_existing=False)


def test_execute_sqlite(db: DbBase):
    db.execute_ddl("create table foo (bar integer)", "sqlite")
    assert db.simplify_model(db.model())["foo"].columns[0].typ == DbType.INTEGER


def test_multi_key():
    mod = model_from_ddl("create table foo (bar integer, baz integer, primary key (bar, baz))")
    assert mod["foo"].indexes == {DbIndex((DbIndexField("bar"), DbIndexField("baz")), primary=True), }


def test_prefix_index() -> None:
    prefix_idx_sql = """
        CREATE TABLE foo (bar INTEGER, txt TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(txt(10));
    """

    non_prefix_idx_sql = """
        CREATE TABLE foo (bar INTEGER, txt TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(txt);
    """

    mod1 = model_from_ddl(prefix_idx_sql, "mysql")
    with pytest.raises(err.OperationalError):
        model_from_ddl(prefix_idx_sql, "sqlite")

    mod2 = model_from_ddl(non_prefix_idx_sql, "mysql")

    assert mod1["foo"].indexes == {
        DbIndex((DbIndexField("txt", prefix_len=10),), unique=False, primary=False),
        DbIndex((DbIndexField("bar", prefix_len=None),), unique=False, primary=True),
    }

    assert mod2["foo"].indexes == {
        DbIndex((DbIndexField("txt", prefix_len=None),), unique=False, primary=False),
        DbIndex((DbIndexField("bar", prefix_len=None),), unique=False, primary=True),
    }


def test_prefix_index_multi() -> None:
    prefix_idx_sql = """
        CREATE TABLE foo (bar INTEGER, txt1 TEXT, txt2 TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(txt1(10), bar, txt2(20));
    """

    mod = model_from_ddl(prefix_idx_sql, "mysql")

    assert mod["foo"].indexes == {
        DbIndex((DbIndexField("txt1", prefix_len=10), DbIndexField("bar"), DbIndexField("txt2", prefix_len=20)), unique=False, primary=False),
        DbIndex((DbIndexField("bar", prefix_len=None),), unique=False, primary=True),
    }


def test_prefix_index_bad() -> None:
    # We don't validate that the column name makes sense.
    ddl = """
        CREATE TABLE foo (bar INTEGER, txt1 TEXT, txt2 TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(abc(10));
    """

    mod = model_from_ddl(ddl, "mysql")
    assert DbIndex((DbIndexField("abc", prefix_len=10),)) in mod["foo"].indexes

    # Looks like a prefix index, but has too many args.
    ddl = """
        CREATE TABLE foo (bar INTEGER, txt1 TEXT, txt2 TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(txt1(10, 20));
    """

    with pytest.raises(err.SchemaError):
        mod = model_from_ddl(ddl, "mysql")

    # Looks like a prefix index, but has the wrong type of arg.
    ddl = """
        CREATE TABLE foo (bar INTEGER, txt1 TEXT, txt2 TEXT, PRIMARY KEY (bar));
        CREATE INDEX prefix_idx ON foo(txt1(abc));
    """

    with pytest.raises(err.SchemaError):
        mod = model_from_ddl(ddl, "mysql")


def test_expression_indices() -> None:
    tbl_def = "CREATE TABLE foo (bar INTEGER, txt1 TEXT, txt2 TEXT, PRIMARY KEY (bar));"

    sqlite_stmts = (
        "CREATE INDEX exp_idx ON foo(lower(txt1));",
        "CREATE INDEX prefix_idx ON foo(bar * 2);",
        "CREATE INDEX exp_idx ON foo(bar, lower(txt1));",
        "CREATE INDEX prefix_idx ON foo(bar, bar * 2);",
    )

    for stmt in sqlite_stmts:
        with pytest.raises(err.SchemaError, match="Indices on expressions"):
            model_from_ddl(tbl_def + " " + stmt, "sqlite")

    mysql_stmts = (
        "CREATE INDEX exp_idx ON foo((lower(txt1)));",
        "CREATE INDEX prefix_idx ON foo((bar * 2));",
        "CREATE INDEX exp_idx ON foo(bar, (lower(txt1)));",
        "CREATE INDEX exp_idx ON foo(bar, (bar * 2));",
        "CREATE INDEX exp_idx ON foo(txt1(10), (bar * 2));",
    )
    for stmt in mysql_stmts:
        with pytest.raises(err.SchemaError, match="Unsupported type"):
            model_from_ddl(tbl_def + stmt, "mysql")


def test_simple_parse_unsupported_dialect() -> None:
    ddl = """
        CREATE TABLE foo (bar INTEGER, txt TEXT NOT NULL, PRIMARY KEY (bar));
        CREATE INDEX idx ON foo(txt);
    """

    # Check that basic parsing is supported for other sqlglot-supported dialects
    # Presto chosen because it seems an unlikely candidate for support in the near future
    mod = model_from_ddl(ddl, "presto")
    assert mod["foo"] == DbTable(
        columns=(
            DbCol("bar", DbType.INTEGER, size=4),
            DbCol("txt", DbType.TEXT, notnull=True),
        ),
        indexes={
            DbIndex((DbIndexField("bar"),), primary=True),
            DbIndex((DbIndexField("txt"),),),
        },
    )

    ddl = """
        CREATE TABLE foo (bar INTEGER, txt TEXT NOT NULL, PRIMARY KEY (bar));
        CREATE INDEX idx ON foo(lower(txt));
    """

    with pytest.raises(err.SchemaError, match="Unsupported type"):
        model_from_ddl(ddl, "presto")

    ddl = """
        CREATE TABLE foo (bar INTEGER, txt TEXT NOT NULL, PRIMARY KEY (bar));
        CREATE INDEX idx ON foo((txt));
    """

    mod = model_from_ddl(ddl, "presto")

    assert mod["foo"] == DbTable(
        columns=(
            DbCol("bar", DbType.INTEGER, size=4),
            DbCol("txt", DbType.TEXT, notnull=True),
        ),
        indexes={
            DbIndex((DbIndexField("bar"),), primary=True),
            DbIndex((DbIndexField("txt"),),),
        },
    )

    ddl = """
        CREATE TABLE foo (bar INTEGER, txt TEXT NOT NULL, PRIMARY KEY (bar));
        CREATE INDEX idx ON foo(flembo(txt));
    """

    with pytest.raises(err.SchemaError, match="Unsupported type"):
        model_from_ddl(ddl, "presto")


def test_primary_key():
    mod = model_from_ddl("create table foo (bar integer primary key, baz integer)")
    assert mod["foo"].indexes == {DbIndex((DbIndexField("bar"), ), primary=True), }


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
    assert mod["cars"].indexes == {DbIndex((DbIndexField("id"),), primary=True), }


def test_default_bool():
    mod = model_from_ddl("create table foo (bar boolean default TRUE)")
    assert mod["foo"].columns == (DbCol("bar", DbType.BOOLEAN, default=True),)


def test_not_null_pk():
    create = "CREATE TABLE a (id INTEGER, dd TEXT, PRIMARY KEY(id));"
    mod = model_from_ddl(create)
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=False, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex((DbIndexField("id"),), primary=True)}


def test_explicit_not_null_pk():
    create = "CREATE TABLE a (id INTEGER NOT NULL, dd TEXT, PRIMARY KEY(id));"
    mod = model_from_ddl(create)
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=True, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex((DbIndexField("id"),), primary=True)}


def test_unique_col():
    create = "CREATE TABLE a (id INTEGER NOT NULL, dd TEXT unique);"
    mod = model_from_ddl(create, "mysql")
    assert mod["a"].columns == (DbCol("id", DbType.INTEGER, notnull=True, size=4), DbCol("dd", DbType.TEXT))
    assert mod["a"].indexes == {DbIndex((DbIndexField("dd"),), unique=True)}


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
