class DbError(Exception):
    pass


class SchemaError(DbError):
    pass


class TableExistsError(SchemaError):
    pass


class TableNotFoundError(SchemaError):
    pass


class OperationalError(DbError):
    pass


class IntegrityError(DbError):
    pass


class DbConnectionError(DbError, ConnectionError):
    pass
