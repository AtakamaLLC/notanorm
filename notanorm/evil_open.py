import sys
import io
import os


def is_windows():
    return sys.platform in ("win32", "cygwin")


if is_windows():
    GENERIC_READ = 2147483648  # [CodeStyle: Windows Enum value]
    GENERIC_WRITE = 1073741824  # [CodeStyle: Windows Enum value]
    FILE_SHARE_READ = 1
    FILE_SHARE_WRITE = 2
    FILE_SHARE_DELETE = 4
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    INVALID_HANDLE_VALUE = -1
    NULL = 0

    import ctypes
    from ctypes.wintypes import HANDLE, ULONG, DWORD, BOOL, LPCSTR, LPCWSTR, LPVOID
    from pathlib import Path
    import msvcrt

    def create_file(
        path: str,
        access: int,
        share_mode: int,
        security_attrs,
        creation: int,
        flags: int,
        template,
    ) -> HANDLE:
        CreateFileW = ctypes.windll.kernel32.CreateFileW
        CreateFileW.argtypes = (
            LPCWSTR,  # LPCWSTR               lpFileName
            DWORD,  # DWORD                 dwDesiredAccess
            DWORD,  # DWORD                 dwShareMode
            LPVOID,  # LPSECURITY_ATTRIBUTES lpSecurityAttributes
            DWORD,  # DWORD                 dwCreationDisposition
            DWORD,  # DWORD                 dwFlagsAndAttributes
            HANDLE,  # HANDLE                hTemplateFile
        )
        CreateFileW.restype = HANDLE

        # these two arguments below have not been testing.
        lpSecurityAttributes = ctypes.byref(security_attrs) if security_attrs else NULL
        hTemplateFile = template if template else NULL

        if isinstance(path, Path):
            path = str(path)

        return CreateFileW(
            path,
            access,
            share_mode,
            lpSecurityAttributes,
            creation,
            flags,
            hTemplateFile,
        )

    def win32_os_fopen(path, wflags, wdisp, share):
        # get an handle using win32 API, specifyng the SHARED access!
        handle = create_file(path, wflags, share, None, wdisp, 0, None)

        if handle == INVALID_HANDLE_VALUE:  # pragma: no cover
            raise Exception("Open error code %s" % ctypes.GetLastError())

        # get a file descriptor associated to the handle
        file_descriptor = msvcrt.open_osfhandle(handle, 0)

        return file_descriptor

    def evil_open(path, mode, encoding=None):
        """Works just like open, except share mode is read/write/delete (more unix-like)"""
        if "w" in mode:
            wdisp = CREATE_ALWAYS
        elif "a" in mode:
            wdisp = OPEN_ALWAYS
        else:
            wdisp = OPEN_EXISTING

        if "r" in mode and "+" not in mode:
            wflags = GENERIC_READ
        else:
            wflags = GENERIC_READ | GENERIC_WRITE

        share = FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE

        file_descriptor = win32_os_fopen(path, wflags, wdisp, share)

        if "a" in mode:
            os.lseek(file_descriptor, 0, os.SEEK_END)

        # open the file descriptor
        fh = io.open(file_descriptor, mode, encoding=encoding)

        # sadly, the name property is not settable
        fh.path = path

        return fh

else:
    evil_open = open
