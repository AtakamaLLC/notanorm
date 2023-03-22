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
    CREATE_NEW = 1
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    TRUNCATE_EXISTING = 5
    INVALID_HANDLE_VALUE = -1
    NULL = 0

    try:
        import ctypes
        from ctypes.wintypes import HANDLE, DWORD, LPCWSTR, LPVOID
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
            CreateFileW.restype = int

            # these two arguments below have not been testing.
            lpSecurityAttributes = (
                ctypes.byref(security_attrs) if security_attrs else NULL
            )
            hTemplateFile = template if template else NULL

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
                winerr = ctypes.GetLastError()
                strerr = ctypes.FormatError(winerr)

                # oserror will translate for us (see fnf test!)
                raise OSError(0, strerr, path, winerr)

            # get a file descriptor associated to the handle
            file_descriptor = msvcrt.open_osfhandle(handle, 0)

            return file_descriptor

        def os_open(path, flags):
            """Works just like open, except share mode is read/write/delete (more unix-like)"""

            if flags & os.O_RDWR:
                wflags = GENERIC_READ | GENERIC_WRITE
            elif flags & os.O_WRONLY:
                wflags = GENERIC_WRITE
            else:
                wflags = GENERIC_READ

            if flags & os.O_CREAT:
                wdisp = OPEN_ALWAYS
                if flags & os.O_EXCL:
                    wdisp = CREATE_NEW
                elif flags & os.O_TRUNC:
                    wdisp = CREATE_ALWAYS
            else:
                wdisp = OPEN_EXISTING
                if flags & os.O_TRUNC:
                    wdisp = TRUNCATE_EXISTING

            share = FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE

            file_descriptor = win32_os_fopen(path, wflags, wdisp, share)

            if flags & os.O_APPEND:
                os.lseek(file_descriptor, 0, os.SEEK_END)

            return file_descriptor

        def evil_open(path, mode, encoding=None):
            # works just like io.open, but with a nicer share mode
            return io.open(path, mode, encoding=encoding, opener=os_open)

    except ImportError:
        evil_open = open

else:
    evil_open = open
