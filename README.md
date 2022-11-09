## Overview

Simple library that makes working with databases more convenient in python.

`notanorm` can return objects instead of rows, protects you from injection, and 
has a very simply driver interface.

Decidedly not an ORM, since ORM's are typically mega libraries with 
often confusing semantics.

## Example:


```
from notanorm import SqliteDb, open_db 
# from notanorm import MysqlDb 


fname = ":memory:"

# accepts all the same parameters as sqlite3.connect
db = SqliteDb(fname)

# this works too, params turn into kwargs
db = open_db("sqlite://:memory:")

db.uri      # get my uri
db.clone()  # make a copy of the db connection

# no special create support, just use a string
db.query("create table foo (bar text, iv integer primary key)")

# insert, select, update, and upsert are convenient and do the right thing
# preventing injection, normalizing across db types, etc.
db.insert("foo", bar="hi", iv=1)
db.insert("foo", bar="hi", iv=2)
ret = db.insert("foo", bar="ho")                   # sqlite autoincrement used
print(ret.lastrowid)                               # "3"

db.upsert("foo", bar="ho", iv=4)                   # upsert requires a primary key, updates or inserts
db.upsert("foo", bar="ho", _insert_only={"iv": 4}) # _insert_only values are never used to update rows

# can update with a where clause, or with an implied one
db.update("foo", iv=4, bar="up")                   # update, no where clause, inferred from primary key
ret = db.update("foo", {"bar": "up"}, bar="hop")   # update, primary key not needed
print(ret.rowcount, ret.lastrowid)                 # "1 4" <- 1 row updated, primary key was 4

# select_one, select
db.select_one("foo", iv=1).bar                     # hi
db.select("foo", bar="hi")[0].bar                  # hi
db.select("foo", {"bar": "hi"})[0].iv              # 1

# use a generator, for speed
[row.iv for row in db.select_gen("foo", bar="hi")]                      # [1, 2]

# use order_by
[row.iv for row in db.select_gen("foo", bar="hi", order_by="iv desc")   # [2, 1]

db.count("foo", bar="hi")                          # 2


class Foo:
    def __init__(self, bar=None, iv=None):
        self.bar = bar
        self.iv = iv
    def iv_squared(self):
        return self.iv * self.iv

# create a class during select
db.register_class("foo", Foo)
obj = db.select_one("foo", bar="hop")

print(obj.iv_squared(), obj.bar)                   # 16 hop


# inside a transaction, updates and inserts are faster, and are atomic
with db.transaction():
    db.update("foo", iv=4, bar="up")
    db.update("foo", iv=3, bar="up")

# delete a row
db.delete("foo", iv=1)

# select greater than
db.select("foo", iv=notnaorm.Op(">", 3))

# wipe the table
db.delete_all("foo")

### cross database ddl management

# this requires: pip install sqlglot

model = notanorm.model_from_ddl("create table foo(bar integer)")

# create tables and indices from the generic model
db.create_model(model)

# capture sql statements instead of executing them
with db.capture_sql(execute=False) as cap:
   ... do some stuff

print(cap)
```

## Database support

sqlite3 drivers come with python

To use mysql, `pip install mysqlclient`, or if that is not available, `pip install pymysql`
