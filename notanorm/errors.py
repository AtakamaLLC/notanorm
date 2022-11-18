class DbError(RuntimeError):
    pass


class SchemaError(DbError):
    pass


class TableExistsError(SchemaError):
    pass


class UnknownPrimaryError(SchemaError):
    pass


class TableNotFoundError(SchemaError):
    pass


class NoColumnError(SchemaError):
    pass


class OperationalError(DbError):
    pass


class IntegrityError(DbError):
    pass


class DbConnectionError(DbError, ConnectionError):
    pass


class DbClosedError(DbConnectionError):
    pass


class DbReadOnlyError(DbError):
    pass


class ProgrammingError(DbError):
    pass


class MoreThanOneError(DbError, AssertionError):
    pass


class UnsafeGeneratorError(DbError):
    pass
