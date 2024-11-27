# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Damped+Driven_Oscillator_V1.2.py', 'fit_tools.py'],
    pathex=['.'],
    binaries=[],
    datas=[('CMU-PHY_Logo_268px.png', '.')],
    hiddenimports=["'PIL._tkinter_finder'"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
splash = Splash(
    'Ultrasonic_USB_Data_Acquisition_splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='Damped+Driven_Oscillator_V1.2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Damped+Driven_Oscillator.ico'],
)
