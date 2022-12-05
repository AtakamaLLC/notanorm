# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging
import pytest

from notanorm import DbModel, DbCol, DbType, DbTable, DbIndex, DbBase, DbIndexField
from notanorm.errors import SchemaError

log = logging.getLogger(__name__)


def test_model_create_many(db: "DbBase"):
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
                    DbIndex(fields=(DbIndexField("auto"),), primary=True),
                    DbIndex(fields=(DbIndexField("flt"),), unique=True),
                },
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_intsize(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("int1", typ=DbType.INTEGER, size=1),
                    DbCol("int2", typ=DbType.INTEGER, size=2),
                    DbCol("int4", typ=DbType.INTEGER, size=4),
                    DbCol("int8", typ=DbType.INTEGER, size=8),
                ),
                indexes=set(),
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)
    if db.uri_name != "sqlite":
        # other db's support varying size integers
        assert check["foo"].columns[0].size == 1
        assert check["foo"].columns[1].size == 2
        assert check["foo"].columns[2].size == 4
        assert check["foo"].columns[3].size == 8


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
                    DbIndex(
                        fields=(DbIndexField("part1"), DbIndexField("part2")),
                        primary=True,
                    ),
                },
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert check["foo"].indexes == model["foo"].indexes


def test_model_ddl_cross(db):
    # create db mased on model, extract the model, recreate.  same db.
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
                    DbIndex(fields=(DbIndexField("auto"),), primary=True),
                    DbIndex(fields=(DbIndexField("flt"),), unique=True),
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


def test_model_prefix_index(db: DbBase) -> None:
    model = DbModel(
        foo=DbTable(
            columns=(
                DbCol("intk", typ=DbType.INTEGER, size=4),
                DbCol("tex", typ=DbType.TEXT),
            ),
            indexes={
                DbIndex(fields=(DbIndexField("tex", prefix_len=4),)),
            },
        ),
    )

    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)

    _check_index_prefix_lens(db, model["foo"], check["foo"])


def test_model_prefix_index_multi(db: DbBase) -> None:
    model = DbModel(
        foo=DbTable(
            columns=(
                DbCol("intk", typ=DbType.INTEGER, size=4),
                DbCol("tex1", typ=DbType.TEXT),
                DbCol("tex2", typ=DbType.TEXT),
            ),
            indexes={
                DbIndex(
                    fields=(
                        DbIndexField("tex1", prefix_len=4),
                        DbIndexField("intk"),
                        DbIndexField("tex2", prefix_len=7),
                    )
                ),
            },
        ),
    )

    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)

    _check_index_prefix_lens(db, model["foo"], check["foo"])


def _check_index_prefix_lens(
    db: DbBase, tbl_expected: DbTable, tbl_actual: DbTable
) -> None:
    if db.uri_name == "sqlite":
        # SQLite doesn't support prefix indices, so we expect that metadata to be dropped
        assert len(tbl_expected.indexes) == 1
        new_fields = tuple(
            ind._replace(prefix_len=None)
            for ind in next(iter(tbl_expected.indexes)).fields
        )
        expected = {DbIndex(fields=new_fields)}
    else:
        # Other DBs do though
        expected = tbl_expected.indexes

    assert tbl_actual.indexes == expected


def test_model_preserve_types(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("vtex", typ=DbType.TEXT, size=3, notnull=True),
                    DbCol("vbin", typ=DbType.BLOB, size=2),
                ),
            )
        }
    )
    db.execute("create table foo (vtex varchar(3) not null, vbin varbinary(2))")
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_primary_key(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("vtex", typ=DbType.TEXT, size=8),),
                indexes={DbIndex((DbIndexField("vtex"),), primary=True)},
            )
        }
    )
    db.create_model(model)
    check = db.model()
    # simplify model is needed, since many db's will quietly default the column to not null
    # and that's ok
    # so we "simplify" by forcing all primary keys to not-null
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_create_nopk(db: "DbBase"):
    # no primary key
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("inty", typ=DbType.INTEGER, size=4),),
                indexes={DbIndex(fields=(DbIndexField("inty"),), primary=False)},
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)
    assert not check["foo"].indexes.pop().primary


def test_model_create_indexes(db: "DbBase"):
    # no primary key
    model = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("inty", typ=DbType.INTEGER, size=4, autoinc=True),
                    DbCol("vary", typ=DbType.BLOB, size=16),
                ),
                indexes={
                    DbIndex(fields=(DbIndexField("inty"),), primary=True),
                    DbIndex(fields=(DbIndexField("vary"),), unique=True),
                },
            )
        }
    )
    db.create_model(model)
    check = db.model()
    assert db.simplify_model(check) == db.simplify_model(model)


def test_model_cap(db):
    model = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("inty", typ=DbType.INTEGER),),
                indexes={DbIndex(fields=(DbIndexField("inty"),), primary=False)},
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


def test_model_any(db):
    mod = DbModel(
        {
            "foo": DbTable(
                columns=(DbCol("any", typ=DbType.ANY),),
            )
        }
    )
    if db.uri_name != "sqlite":
        with pytest.raises(SchemaError):
            db.create_model(mod)
    else:
        db.create_model(mod)


def test_model_cmp(db):
    model1 = DbModel(
        {
            "foo": DbTable(
                columns=(
                    DbCol("Auto", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=(DbIndexField("Auto"),), primary=True)},
            )
        }
    )
    model2 = DbModel(
        {
            "FOO": DbTable(
                columns=(
                    DbCol("autO", typ=DbType.INTEGER, autoinc=True, notnull=True),
                ),
                indexes={DbIndex(fields=(DbIndexField("autO"),), primary=True)},
            )
        }
    )

    assert model1["foo"].columns[0] == model2["FOO"].columns[0]
    assert model1 == model2
