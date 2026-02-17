# ScanMower (Python)

## Description
**ScanMower** is a Windows desktop application for manual scan cropping, designed for cultural heritage / book scanning workflows:

- Create and edit a **frame** (quad or rectangle/square) per scan
- Crop the current scan using the saved frame (**Crop current scan**)
- Batch crop scans that have saved frames (**Batch crop scans**)
- Optional deskew (including **manual deskew**)
- Preserve scan parameters and metadata where applicable (ICC, DPI, etc.)
- **Color management via LittleCMS2** (through `pylcms2`), e.g. conversion to **eciRGB v2**

Frames are stored as JSON sidecar files (named **frame**) in:
`OUTPUT/_frames/<filename>.json`

---

## Prerequisites

### Runtime (Windows)
- Windows 10/11 (x64)
- Recommended input scans: TIFF (RGB, 8/16-bit per channel)

### Build requirements
- Python **3.13** (x64)
- PowerShell (for build commands)
- `pip` and `venv`
- PyInstaller (to build `.exe`)
- Optional: **UPX** (to reduce distribution size)

---

## Build from source

### 1) (PowerShell) Allow venv activation in this session
If script execution is restricted, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2) Create and activate a virtual environment
```powershell
cd C:\temp\man_crop
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 3) Install dependencies from `requirements.txt`
```powershell
pip install -r requirements.txt
```

### 4) Install PyInstaller (build tool)
```powershell
pip install pyinstaller
```

> Optional: If you do not need OpenCV GUI features, you can try `opencv-python-headless`
> (this may reduce distribution size). In that case, adjust `requirements.txt` accordingly:
```powershell
pip uninstall -y opencv-python
pip install opencv-python-headless
```

### 5) Build (recommended: onedir)
`onedir` typically starts faster than `onefile` and is easier to troubleshoot.

```powershell
pyinstaller --noconfirm --clean ScanMower.spec
```

Output:
- `dist\ScanMower\ScanMower.exe`

### 6) Build with UPX (optional)
1) Download UPX (Windows x64) and unpack it, e.g. `C:\tools\upx`
2) Build using:

```powershell
pyinstaller --noconfirm --clean --upx-dir C:\tools\upx ScanMower.spec
```

> UPX can reduce size, but in some environments it may increase false positives in antivirus/SmartScreen.
> If you encounter issues, disable UPX in the `.spec` or expand `upx_exclude`.

---

## Windows warning: “Unknown publisher”
Because the executable is not code-signed, Windows may show SmartScreen warnings such as:
- **“Windows protected your PC”**
- **“Unknown publisher”**

### How to run anyway
1) Click **More info**
2) Click **Run anyway**

---

## Screenshot
<img width="2560" height="1392" alt="scanmower_0 0 8" src="https://github.com/user-attachments/assets/c16a10fe-6083-4ef8-be0f-215ed8582a03" />


