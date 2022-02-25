import logging

logging.getLogger().setLevel(logging.DEBUG)


def pytest_configure(config):
    config.addinivalue_line("markers", "db")


def pytest_addoption(parser):
    parser.addoption("--db", action="append", default=[], help="db(s) to run tests for")
