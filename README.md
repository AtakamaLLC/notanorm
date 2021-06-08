Simple library that makes working with databases more convenient in python.

`notanorm` can return objects instead of rows, protects you from injection, and 
has a very simply driver interface.

Decidedly not an ORM, since ORM's are typically mega libraries with 
often confusing semantics.

Example:

```
from notanorm import SqliteDb 
from notanorm import MysqlDb 

fname = ":memory:"

# default options are 
db = SqliteDb(fname)

# no special create support, just use a string
db.query("create table foo (bar text, iv integer primary key)")

# insert, select, update, and upsert are convenient and do the right thing
# preventing injection, normalizing across db types, etc.

db.insert(bar="hi", iv=1)
db.insert(bar="hi", iv=2)
db.insert(bar="ho", iv=3)

db.upsert(bar="ho", iv=4)                   # primary keys are required for upserts

db.select("foo", bar="hi")[0].bar           # hi
db.select("foo", {"bar": "hi"})[0].iv       # 1

db.count("foo", bar="hi")                   # 2

class Foo:
    def __init__(self, bar=None, iv=None):
        self.bar = bar
        self.iv = iv

# create a class during select
db.register_class("foo", Foo)
obj = db.select_one("foo", bar="hi")
print(obj.bar)                              # hi
```

To use mysql, `pip install mysqlclient`.
