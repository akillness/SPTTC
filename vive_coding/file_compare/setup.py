from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["sys", "difflib", "PySide6"],
    "excludes": [],
    "include_files": []
}

setup(
    name="File Compare Tool",
    version="1.0",
    description="A tool to compare two files and show differences",
    options={"build_exe": build_exe_options},
    executables=[Executable("file_compare.py", base="Win32GUI" if sys.platform == "win32" else None)]
) 