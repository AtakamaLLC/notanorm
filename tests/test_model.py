# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging

from notanorm import DbModel, DbCol, DbType, DbTable, DbIndex

log = logging.getLogger(__name__)


def test_model_create(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
                    DbCol("blob", typ=DbType.BLOB),
                    DbCol("bool", typ=DbType.BOOLEAN),
                    DbCol("blob3", typ=DbType.BLOB, size=3, fixed=True),
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
    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_create_composite_pk(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("part1", typ=DbType.INTEGER, notnull=True),
                    DbCol("part2", typ=DbType.BLOB, size=16, notnull=True),
                    DbCol("blob4", typ=DbType.BLOB, size=4, fixed=False),
                    DbCol("tex", typ=DbType.TEXT, notnull=True),
                    DbCol("siz3v", typ=DbType.TEXT, size=3, fixed=False),
                    DbCol("siz3", typ=DbType.TEXT, size=3, fixed=True),
                    DbCol("flt", typ=DbType.FLOAT, default="1.1"),
                    DbCol("dbl", typ=DbType.DOUBLE, default="2.2"),
                ),
                indexes={
                    DbIndex(fields=("part1", "part2"), primary=True),
                },
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert check["foo"].indexes == model["foo"].indexes


def test_model_ddl_cross(db):
    # creating a model using sqlite results in a model that generally works across other db's
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
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
    db.create_model(model)
    extracted_model = db.model()

    db.execute("drop table foo")

    db.create_model(extracted_model)
    check = db.model()
    assert check == extracted_model


def test_model_preserve_types(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("vtex", typ=DbType.TEXT, size=3, notnull=True), DbCol("vbin", typ=DbType.BLOB, size=2)),
            )
        }
    )
    db.execute("create table foo (vtex varchar(3) not null, vbin varbinary(2))")
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_create_nopk(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("inty", typ=DbType.INTEGER),),
                indexes={DbIndex(fields=("inty",), primary=False)},
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert check == model


def test_model_cap(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("inty", typ=DbType.INTEGER),),
                indexes={DbIndex(fields=("inty",), primary=False)},
            )
        }
    )

    ddl = db.ddl_from_model(model)

    expect = """
create table foo("inty" integer);
create index "ix_foo_inty" on foo ("inty");
"""
    if db.uri_name == "sqlite":
        assert ddl.strip() == expect.strip()
    else:
        # vague assertion that we captured stuff
        assert "create table" in ddl.lower()
        assert "foo" in ddl
        assert "inty" in ddl


def test_model_cmp(db):
    model1 = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("Auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=("Auto",), primary=True)},
            )
        }
    )
    model2 = DbModel(
        {
            "FOO": DbTable(
                columns=(
                    DbCol("autO", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=("autO",), primary=True)},
            )
        }
    )

    assert model1["foo"].columns[0] == model2["FOO"].columns[0]
    assert model1 == model2
