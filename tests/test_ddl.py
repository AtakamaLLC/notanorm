# pylint: disable=missing-docstring, protected-access, unused-argument, too-few-public-methods
# pylint: disable=import-outside-toplevel, unidiomatic-typecheck

import logging
import multiprocessing
import sqlite3
import threading
import time
from multiprocessing.pool import ThreadPool

import pytest

import notanorm.errors
from notanorm import SqliteDb, DbRow, DbModel, DbCol, DbType, DbTable, DbIndex, DbBase
from notanorm.connparse import parse_db_uri

import notanorm.errors as err
from notanorm.connparse import open_db

from notanorm.ddh_helper import DDLHelper

log = logging.getLogger(__name__)

# noinspection PyUnresolvedReferences
from .test_notanorm import db_fixture, db_sqlite

@pytest.fixture
def db_name():
    return "sqlite"


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

    # ddl / create + model are the same
    ddl = db.ddl_from_model(model)
    for ent in ddl.split(";"):
        db.execute(ent)
    db2 = SqliteDb(":memory:")
    db2.create_model(model)
    captured_model1 = db.model()
    captured_model2 = db2.model()
    assert captured_model1 == captured_model2

    extracted = DDLHelper(ddl, "sqlite").model()

    assert not model_diff(extracted, captured_model1)


def model_diff(a: DbModel, b: DbModel):
    diff = []

    for tab, coldef in a.items():
        coldef: DbTable
        if not b.get(tab):
            diff.append(("<", {tab}))
            continue

        bcols: DbTable = b.get(tab)
        for i, acol in enumerate(coldef.columns):
            if i >= len(bcols.columns):
                diff.append(("<", {tab: acol.name}))
                continue

            if acol != bcols.columns[i]:
                diff.append(("!", {tab: acol.name}))

        for i, bcol in enumerate(bcols.columns):
            if i >= len(bcols.columns):
                diff.append((">", {tab: bcol.name}))
                continue

        bcols: DbTable = b.get(tab)
        for adex in coldef.indexes:
            if adex not in bcols.indexes:
                diff.append(("<", {tab: adex}))
                continue

        for bdex in bcols.indexes:
            if bdex not in coldef.indexes:
                diff.append((">", {tab: bdex}))
                continue

    for tab, coldef in b.items():
        if not a.get(tab):
            diff.append((">", {tab}))

    return diff



