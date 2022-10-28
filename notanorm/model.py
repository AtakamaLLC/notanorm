"""Model definition support."""

import sqlglot

__all__ = ["DbModel"]

from sqlglot import Dialect


class DbModel:
    """Container holding a parsed table definition"""

    def __init__(self, ddl, dialect=None):
        self.__bnf = sqlglot.parse(ddl, read=dialect)

    def __eq__(self, other):
        a = self.__bnf
        b = other.__bnf
        return a == b

    def to_sql(self, dialect, **opts):
        write = dialect
        return [
            Dialect.get_or_raise(write)().generate(expression, **opts)
            for expression in self.__bnf
        ]