import io
import pytest
import os
from notanorm.evil_open import evil_open, is_windows


def _test_open_read_write(tmp_path, data):
    if type(data) is bytes:
        flag = "b"
        enc = None
    else:
        flag = ""
        enc = "utf8"

    fil = tmp_path / "x"
    with evil_open(fil, "w" + flag, encoding=enc) as fh:
        fh.write(data)
        with pytest.raises(io.UnsupportedOperation):
            fh.read()

    if is_windows():
        # trad open raises the right error
        with open(fil, "a" + flag, encoding=enc) as fh:
            with pytest.raises(PermissionError):
                os.rename(fil, tmp_path / "zzz")

    with pytest.raises(FileNotFoundError):
        evil_open(tmp_path / "zzz", "r" + flag)

    with evil_open(fil, "r" + flag) as fh:
        assert fh.read() == data
        with pytest.raises(io.UnsupportedOperation):
            fh.write(data)

    with evil_open(fil, "w+" + flag, encoding=enc) as fh:
        fh.write(data)
        fh.seek(0)
        assert fh.read() == data

    with evil_open(fil, "r" + flag) as fh:
        assert fh.read() == data

    with evil_open(fil, "a" + flag, encoding=enc) as fh:
        fh.write(data)

    with evil_open(fil, "r" + flag) as fh:
        assert fh.read() == data + data

    with evil_open(fil, "r+" + flag) as fh:
        fh.seek(0)
        fh.write(data + data + data)
        fh.seek(0)
        assert fh.read() == data + data + data


def test_open_text(tmp_path):
    _test_open_read_write(tmp_path, "ok")


def test_open_binary(tmp_path):
    _test_open_read_write(tmp_path, b"ok")


def test_open_rename_while(tmp_path):
    fil = tmp_path / "x"
    with evil_open(fil, "w", encoding="utf8") as fh:
        fh.write("ok")
        os.rename(fil, tmp_path / "y")

    with open(tmp_path / "y", "r") as fh:
        assert fh.read() == "ok"


def test_os_open(tmp_path, open_func=os.open):
    fil = str(tmp_path / "x")

    h = open_func(fil, os.O_CREAT | os.O_WRONLY)
    os.write(h, b"hi")
    os.close(h)

    h = open_func(fil, os.O_RDONLY)
    assert os.read(h, 10) == b"hi"
    os.close(h)

    h = open_func(fil, os.O_TRUNC | os.O_RDWR)
    assert os.read(h, 10) == b""
    os.close(h)

    if is_windows():
        with pytest.raises(OSError):
            open_func(fil, os.O_TRUNC)

        h = open_func(fil, os.O_CREAT)
        with pytest.raises(OSError):
            os.write(h, b"hi")


if is_windows():

    def test_win_os_open(tmp_path):
        from notanorm.evil_open import os_open

        test_os_open(tmp_path, open_func=os_open)
