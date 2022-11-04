import sys
import logging
import pytest
from notanorm import DbModel, DbCol, DbType, DbTable, DbIndex

log = logging.getLogger(__name__)

if tuple(sys.version_info[0: 2]) <= (3, 6):
    pytest.skip("sqlglot requires python 3.7 or greateer", allow_module_level=True)


from notanorm.ddl_helper import model_from_ddl  # noqa


def test_model_ddl_cap(db):
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

    # ddl / create + model are the same
    ddl = db.ddl_from_model(model)
    for ent in ddl.split(";"):
        if ent.strip():
            db.execute(ent)
    captured_model1 = db.model()

    db.execute("drop table foo")

    db.create_model(model)
    captured_model2 = db.model()
    assert captured_model1 == captured_model2

    extracted = model_from_ddl(ddl, db.uri_name)

    assert extracted == captured_model1
