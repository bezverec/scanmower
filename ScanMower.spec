# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['scanmower.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "PyQt5",
        "PySide6",
        "pandas",
        "scipy",
        "sklearn",
        "tensorflow",
        "torch",
        "pytest",
        "setuptools",
        "pkg_resources",
    ],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    exclude_binaries=True,
    name='ScanMower',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    # UPX exclusions: avoid compressing core runtime DLLs (stability/AV friendliness),
    # while still compressing most .pyd/.dll payload.
    upx_exclude=[
        "python313.dll",
        "tcl86t.dll",
        "tk86t.dll",
        "ucrtbase.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        # optional: if you ever include these:
        "msvcp140_1.dll",
        "concrt140.dll",
        # optional: leave ffmpeg uncompressed if UPX causes issues
        "opencv_videoio_ffmpeg*.dll",
    ],
    name='ScanMower',
)
