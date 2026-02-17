#!/usr/bin/env python3
"""
ScanMower (v0.0.8.1)
Author: Jan Houserek
License: GPLv3
"""

import glob
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import tifffile

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk


APP_NAME = "ScanMower"
SCRIPT_VERSION = "2026-02-16-scanmower-python-v0.0.8.1"


# -----------------------------
# TIFF metadata (ICC/XMP/DPI)
# -----------------------------

def _tag_value_safe(page: tifffile.tifffile.TiffPage, tag_id_or_name) -> Optional[Any]:
    try:
        t = page.tags.get(tag_id_or_name, None)
        if t is None:
            return None
        return t.value
    except Exception:
        return None


def read_tiff_rgb(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with tifffile.TiffFile(path) as tf:
        page0 = tf.pages[0]
        arr = page0.asarray()
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Expected RGB TIFF (H,W,3+), got {arr.shape}")
        arr = arr[:, :, :3]

        icc = None
        for page in tf.pages[:3]:
            v = _tag_value_safe(page, 34675)
            if isinstance(v, (bytes, bytearray)) and len(v) > 0:
                icc = bytes(v)
                break
            for name in ("InterColorProfile", "ICCProfile"):
                v2 = _tag_value_safe(page, name)
                if isinstance(v2, (bytes, bytearray)) and len(v2) > 0:
                    icc = bytes(v2)
                    break
            if icc:
                break

        xmp = _tag_value_safe(page0, 700)
        if not (isinstance(xmp, (bytes, bytearray)) and len(xmp) > 0):
            xmp = _tag_value_safe(page0, "XMP")
        xmp = bytes(xmp) if isinstance(xmp, (bytes, bytearray)) and len(xmp) > 0 else None

        xres = yres = None
        unit = None
        try:
            x = _tag_value_safe(page0, "XResolution")
            y = _tag_value_safe(page0, "YResolution")
            u = _tag_value_safe(page0, "ResolutionUnit")
            if x is not None:
                xres = float(x[0]) / float(x[1]) if isinstance(x, tuple) else float(x)
            if y is not None:
                yres = float(y[0]) / float(y[1]) if isinstance(y, tuple) else float(y)
            if u is not None:
                unit = int(u)
        except Exception:
            pass

        meta = {
            "icc_profile": icc,
            "xmp": xmp,
            "x_resolution": xres,
            "y_resolution": yres,
            "resolution_unit": unit,
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
        }
        return arr, meta


def _resolutionunit_str(unit: Optional[int]) -> Optional[str]:
    if unit == 2:
        return "inch"
    if unit == 3:
        return "centimeter"
    return None


def write_tiff_rgb_uncompressed(path_out: str, rgb: np.ndarray, meta: Dict[str, Any]) -> None:
    extratags = []

    icc = meta.get("icc_profile", None)
    if isinstance(icc, (bytes, bytearray)) and len(icc) > 0:
        extratags.append((34675, "B", len(icc), bytes(icc), False))

    xmp = meta.get("xmp", None)
    if isinstance(xmp, (bytes, bytearray)) and len(xmp) > 0:
        extratags.append((700, "B", len(xmp), bytes(xmp), False))

    xres = meta.get("x_resolution")
    yres = meta.get("y_resolution")
    runit = _resolutionunit_str(meta.get("resolution_unit"))

    tifffile.imwrite(
        path_out,
        rgb,
        photometric="rgb",
        planarconfig="contig",
        compression=None,
        resolution=(float(xres), float(yres)) if (xres is not None and yres is not None) else None,
        resolutionunit=runit,
        extratags=extratags if extratags else None,
    )


# -----------------------------
# LCMS2 via pylcms2 (optional)
# -----------------------------

def _load_pylcms2():
    import pylcms2  # type: ignore
    return pylcms2


def apply_icc_transform(
    rgb: np.ndarray,
    src_icc_bytes: Optional[bytes],
    dst_icc_path: str,
    intent: str = "relative",
    bpc: bool = True
) -> Tuple[np.ndarray, bytes]:
    if not dst_icc_path:
        raise ValueError("CM enabled but destination ICC path is empty.")
    if not os.path.exists(dst_icc_path):
        raise ValueError(f"Destination ICC not found: {dst_icc_path}")
    if not src_icc_bytes:
        raise ValueError("No embedded ICC found in input (CM needs embedded source ICC).")

    pylcms2 = _load_pylcms2()

    if rgb.dtype == np.uint8:
        px_type = "RGB_8"
    elif rgb.dtype == np.uint16:
        px_type = "RGB_16"
    else:
        raise ValueError(f"Unsupported dtype for CM: {rgb.dtype}")

    intent_map = {
        "perceptual": "PERCEPTUAL",
        "relative": "RELATIVE_COLORIMETRIC",
        "absolute": "ABSOLUTE_COLORIMETRIC",
        "saturation": "SATURATION",
    }
    intent_norm = intent_map.get(intent.lower(), "RELATIVE_COLORIMETRIC")

    src_prof = pylcms2.Profile(buffer=src_icc_bytes)
    dst_prof = pylcms2.Profile(filename=dst_icc_path)

    flags = "BLACKPOINTCOMPENSATION" if bpc else "NONE"
    try:
        xform = pylcms2.Transform(src_prof, px_type, dst_prof, px_type, intent_norm, flags)
    except Exception:
        xform = pylcms2.Transform(src_prof, px_type, dst_prof, px_type, intent_norm, "NONE")

    out = np.asarray(xform.apply(rgb)).reshape(rgb.shape).astype(rgb.dtype, copy=False)
    dst_bytes = Path(dst_icc_path).read_bytes()
    return out, dst_bytes


# -----------------------------
# Geometry / warp helpers
# -----------------------------

def order_rect(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_warp_size(rect: np.ndarray) -> Tuple[int, int]:
    (tl, tr, br, bl) = rect
    widthA = float(np.linalg.norm(br - bl))
    widthB = float(np.linalg.norm(tr - tl))
    heightA = float(np.linalg.norm(tr - br))
    heightB = float(np.linalg.norm(tl - bl))
    maxW = max(int(round(widthA)), int(round(widthB)), 2)
    maxH = max(int(round(heightA)), int(round(heightB)), 2)
    return maxW, maxH


def warp_perspective_bgr(img_bgr: np.ndarray, quad4: np.ndarray) -> np.ndarray:
    rect = order_rect(quad4)
    maxW, maxH = compute_warp_size(rect)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        img_bgr, H, (maxW, maxH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    return warped


def crop_axis_aligned_bgr(img_bgr: np.ndarray, rect_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect_xyxy
    h, w = img_bgr.shape[:2]
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        raise ValueError("Rectangle too small.")
    return img_bgr[y0:y1, x0:x1].copy()


def add_padding(img_bgr: np.ndarray, top: int, right: int, bottom: int, left: int) -> np.ndarray:
    if top <= 0 and right <= 0 and bottom <= 0 and left <= 0:
        return img_bgr
    maxv = int(np.iinfo(img_bgr.dtype).max)
    return cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(maxv, maxv, maxv)
    )


def rotate_image(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    maxv = int(np.iinfo(img_bgr.dtype).max)
    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(maxv, maxv, maxv),
    )


def estimate_text_angle_deg(img_bgr: np.ndarray) -> Optional[float]:
    if img_bgr.dtype == np.uint16:
        gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) >> 8).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    scale = min(1.0, 1200.0 / max(w, h))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    ys, xs = np.where(bw > 0)
    if len(xs) < 500:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    w_rect, h_rect = rect[1]
    if w_rect < h_rect:
        angle = angle + 90.0
    return float(angle)


def derive_side_margins(page_side: str, margin_nonspine: int, margin_spine: int) -> Tuple[int, int, int, int]:
    top = margin_nonspine
    bottom = margin_nonspine
    if page_side == "recto":
        left = margin_spine
        right = margin_nonspine
    elif page_side == "verso":
        left = margin_nonspine
        right = margin_spine
    else:
        left = margin_nonspine
        right = margin_nonspine
    return top, right, bottom, left


def mirror_points_horiz(pts: np.ndarray, width_px: int) -> np.ndarray:
    out = np.asarray(pts, dtype=np.float32).copy()
    out[:, 0] = (float(width_px - 1) - out[:, 0])
    return out


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    cnt = poly.reshape(-1, 1, 2).astype(np.float32)
    return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0


# -----------------------------
# File collection / output
# -----------------------------

def collect_tiffs(folder: str) -> List[str]:
    p = Path(folder)
    files: List[str] = []
    for pat in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        files.extend(glob.glob(str(p / pat)))
    return sorted(set(files))


def default_output_dir(input_dir: str) -> str:
    return str(Path(input_dir) / "_scanmower_out")


# -----------------------------
# Auto-suggest (fast-ish translation estimation)
# -----------------------------

def _gray8_from_rgb8(rgb8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)


def _sobel_mag(gray8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


def suggest_frame_translation(
    cur_rgb8: np.ndarray,
    prev_rgb8: np.ndarray,
    prev_pts_full: np.ndarray,
    maxdim: int = 900
) -> np.ndarray:
    """
    Suggests a new quad by translating previous quad based on template matching
    on edge magnitude images (downscaled).
    """
    try:
        g_cur = _gray8_from_rgb8(cur_rgb8)
        g_prev = _gray8_from_rgb8(prev_rgb8)

        def down(g):
            h, w = g.shape[:2]
            s = min(1.0, float(maxdim) / max(h, w))
            if s < 1.0:
                g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            return g, s

        g_prev_s, s_prev = down(g_prev)
        g_cur_s, s_cur = down(g_cur)

        e_prev = _sobel_mag(g_prev_s)
        e_cur = _sobel_mag(g_cur_s)

        pts = np.asarray(prev_pts_full, dtype=np.float32).copy()
        x0 = int(np.clip(np.min(pts[:, 0]) * s_prev, 0, e_prev.shape[1] - 2))
        x1 = int(np.clip(np.max(pts[:, 0]) * s_prev, 2, e_prev.shape[1] - 1))
        y0 = int(np.clip(np.min(pts[:, 1]) * s_prev, 0, e_prev.shape[0] - 2))
        y1 = int(np.clip(np.max(pts[:, 1]) * s_prev, 2, e_prev.shape[0] - 1))

        templ = e_prev[y0:y1, x0:x1]
        if templ.size < 40 * 40:
            return pts

        res = cv2.matchTemplate(e_cur, templ, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.15:
            return pts

        new_x0, new_y0 = maxloc
        dx_s = new_x0 - x0
        dy_s = new_y0 - y0

        dx_full = float(dx_s / max(s_cur, 1e-6))
        dy_full = float(dy_s / max(s_cur, 1e-6))

        pts[:, 0] += dx_full
        pts[:, 1] += dy_full
        return pts
    except Exception:
        return np.asarray(prev_pts_full, dtype=np.float32).copy()


# -----------------------------
# Thumbnail helpers (fast + cached)
# -----------------------------

def load_thumb_rgb8(path: str, max_side: int) -> Optional[np.ndarray]:
    """
    Attempts to load a small RGB8 thumbnail efficiently.
    Prefers PIL (often faster for thumbnails), falls back to tifffile.
    Returns RGB uint8.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), resample=Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.uint8)
            return arr
    except Exception:
        pass

    try:
        rgb, _ = read_tiff_rgb(path)
        if rgb.dtype == np.uint16:
            rgb8 = (rgb >> 8).astype(np.uint8)
        else:
            rgb8 = rgb.astype(np.uint8)
        h, w = rgb8.shape[:2]
        s = min(1.0, float(max_side) / max(h, w))
        if s < 1.0:
            rgb8 = cv2.resize(rgb8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return rgb8
    except Exception:
        return None


class LRUPhotoCache:
    def __init__(self, capacity: int = 400):
        self.capacity = int(max(10, capacity))
        self._d: "OrderedDict[str, ImageTk.PhotoImage]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ImageTk.PhotoImage]:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: str, value: ImageTk.PhotoImage) -> None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def set_capacity(self, capacity: int) -> None:
        with self._lock:
            self.capacity = int(max(10, capacity))
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)


# -----------------------------
# GUI app
# -----------------------------

class ScanMowerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)

        # project
        self.input_dir: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.files: List[str] = []
        self.idx: int = 0

        # current image
        self.img_rgb: Optional[np.ndarray] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.img_h: int = 0
        self.img_w: int = 0

        # view/zoom
        self.zoom: float = 1.0
        self.fit_scale: float = 1.0
        self.view_scale: float = 1.0
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        # frame points (FULL-RES coords) — used when split is OFF
        self.points_full: Optional[np.ndarray] = None

        # split pages
        self.split_enabled_var = tk.BooleanVar(value=False)
        self.split_mode_var = tk.StringVar(value="manual")  # manual | auto
        self.split_active_page_var = tk.StringVar(value="L")  # L or R
        self.pages: Dict[str, Dict[str, Any]] = {}  # {"L": {...}, "R": {...}}

        # dragging
        self.drag_i: Optional[int] = None
        self.drag_mode: Optional[str] = None  # "corner" | "move"
        self.drag_last_xy_img: Optional[Tuple[float, float]] = None

        # autosave
        self._autosave_after_id = None
        self._autosave_delay_ms = 250
        self._initializing = True

        # last for suggest/copy
        self._last_rgb8_for_suggest: Optional[np.ndarray] = None
        self._last_points_full: Optional[np.ndarray] = None
        self._last_settings: Optional[Dict[str, Any]] = None

        # Settings
        self.start_side_var = tk.StringVar(value="recto")
        self.page_side_var = tk.StringVar(value="recto")
        self.auto_toggle_side_var = tk.BooleanVar(value=True)
        self.auto_suggest_frame_var = tk.BooleanVar(value=True)

        self.frame_mode_var = tk.StringVar(value="rect")  # default rect
        self.lock_square_var = tk.BooleanVar(value=False)

        # margins (ex-pad)
        self.margin_nonspine_var = tk.IntVar(value=0)  # default 0
        self.margin_spine_var = tk.IntVar(value=0)     # default 0

        self.manual_deskew_var = tk.DoubleVar(value=0.0)
        self.auto_deskew_var = tk.BooleanVar(value=False)
        self.auto_deskew_max_var = tk.DoubleVar(value=3.0)

        self.cm_enabled_var = tk.BooleanVar(value=True)  # default ON
        self.cm_to_var = tk.StringVar(value="")
        self.cm_intent_var = tk.StringVar(value="relative")
        self.cm_bpc_var = tk.BooleanVar(value=True)

        # thumbnails
        self.thumb_size = 160
        self.thumb_cache = LRUPhotoCache(capacity=1000)
        self.thumb_executor = ThreadPoolExecutor(max_workers=6)
        self.thumb_widgets: Dict[str, tk.Label] = {}
        self.thumb_rows: Dict[str, tk.Frame] = {}
        self._thumb_jobs_cancel = False

        # deskew preview / render caches
        self.view_angle_deg: float = 0.0
        self.rot_cx: float = 0.0
        self.rot_cy: float = 0.0
        self._auto_deskew_angle_cache: Dict[str, Optional[float]] = {}
        self._view_photo_cache: Dict[Tuple[str, int, int, float], ImageTk.PhotoImage] = {}
        self._view_photo_cache_order: List[Tuple[str, int, int, float]] = []
        self._view_photo_cache_cap: int = 8

        # build UI
        self._build_ui()
        self._bind_keys()
        self._bind_autosave_traces()
        self._initializing = False

    # ---------- paths ----------
    def _ensure_output_dirs(self):
        if not self.output_dir:
            raise ValueError("Output dir not set.")
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "_frames").mkdir(parents=True, exist_ok=True)
        (out / "out").mkdir(parents=True, exist_ok=True)

    def _frames_dir(self) -> Path:
        return Path(self.output_dir) / "_frames"

    def _out_dir(self) -> Path:
        return Path(self.output_dir) / "out"

    def _current_path(self) -> str:
        return self.files[self.idx]

    def _frame_path(self, img_path: str) -> Path:
        return self._frames_dir() / f"{Path(img_path).stem}.json"

    # ---------- autosave ----------
    def _schedule_autosave(self):
        if self._autosave_after_id is not None:
            try:
                self.root.after_cancel(self._autosave_after_id)
            except Exception:
                pass
        self._autosave_after_id = self.root.after(self._autosave_delay_ms, self._autosave_now)

    def _autosave_now(self):
        self._autosave_after_id = None
        try:
            self.save_frame(silent=True)
        except Exception:
            pass

    def _bind_autosave_traces(self):
        vars_to_trace = [
            self.start_side_var, self.page_side_var, self.auto_toggle_side_var,
            self.auto_suggest_frame_var,
            self.frame_mode_var, self.lock_square_var,
            self.margin_nonspine_var, self.margin_spine_var,
            self.manual_deskew_var, self.auto_deskew_var, self.auto_deskew_max_var,
            self.cm_enabled_var, self.cm_to_var, self.cm_intent_var, self.cm_bpc_var,
            self.split_enabled_var, self.split_mode_var, self.split_active_page_var,
        ]
        for v in vars_to_trace:
            v.trace_add("write", lambda *args: self._on_option_changed())

    def _on_option_changed(self):
        if getattr(self, "_initializing", False):
            return
        if not self.files or self.img_rgb is None:
            return

        # When in split mode, active page vars must be synced.
        if self.split_enabled_var.get():
            self._pull_active_page_to_vars()

        if self.frame_mode_var.get() == "rect":
            self._ensure_rect_points_from_any_full()
            self._apply_square_lock_full()

        if self.split_enabled_var.get():
            self._push_vars_to_active_page()

        # clear view cache on deskew changes so preview updates instantly
        # (keep only small cache; this makes rotations responsive).
        self._view_photo_cache.clear()
        self._view_photo_cache_order.clear()

        # auto deskew cache: if max changes, invalidate current file
        if self.auto_deskew_var.get():
            self._auto_deskew_angle_cache.pop(self._current_path(), None)

        self._render()
        self._schedule_autosave()

    # ---------- split helpers ----------
    def _ensure_pages_ready(self):
        if "L" not in self.pages:
            self.pages["L"] = {"quad": None}
        if "R" not in self.pages:
            self.pages["R"] = {"quad": None}

    def _active_page_key(self) -> str:
        return "R" if self.split_active_page_var.get().upper().startswith("R") else "L"

    def _push_vars_to_active_page(self):
        """Copy current editing vars (points_full etc.) into active page storage."""
        self._ensure_pages_ready()
        key = self._active_page_key()
        if self.points_full is not None:
            self.pages[key]["quad"] = np.asarray(self.points_full, dtype=np.float32).copy()
        # also store per-page side if desired (optional): keep global side for now

    def _pull_active_page_to_vars(self):
        """Load active page storage into editing vars (points_full)."""
        self._ensure_pages_ready()
        key = self._active_page_key()
        quad = self.pages.get(key, {}).get("quad", None)
        if quad is None:
            return
        self.points_full = np.asarray(quad, dtype=np.float32).copy()

    def _switch_active_page(self, key: str):
        if not self.split_enabled_var.get():
            return
        key = "R" if key.upper().startswith("R") else "L"
        self._push_vars_to_active_page()
        self.split_active_page_var.set(key)
        self._pull_active_page_to_vars()
        if self.frame_mode_var.get() == "rect":
            self._ensure_rect_points_from_any_full()
            self._apply_square_lock_full()
        self._render()
        self._schedule_autosave()

    # ---------- view scale / coordinate transforms ----------
    def _update_view_scale(self):
        if self.img_w <= 0 or self.img_h <= 0:
            self.fit_scale = 1.0
        else:
            cw = max(1, int(self.canvas.winfo_width()))
            ch = max(1, int(self.canvas.winfo_height()))
            cw = max(1, cw - 10)
            ch = max(1, ch - 10)
            self.fit_scale = min(1.0, cw / self.img_w, ch / self.img_h)
        self.view_scale = float(self.fit_scale * self.zoom)
        self.view_scale = max(0.05, min(8.0, self.view_scale))

    def _canvas_to_img(self, x: float, y: float) -> Tuple[float, float]:
        """Map canvas event coords -> image FULL coords, respecting scroll + deskew preview rotation."""
        cx = float(self.canvas.canvasx(x))
        cy = float(self.canvas.canvasy(y))

        # undo rotation around current render center (canvas coords)
        ang = float(getattr(self, "view_angle_deg", 0.0))
        rcx = float(getattr(self, "rot_cx", 0.0))
        rcy = float(getattr(self, "rot_cy", 0.0))
        if abs(ang) > 1e-6:
            rad = np.deg2rad(-ang)
            cosv = float(np.cos(rad))
            sinv = float(np.sin(rad))
            dx = cx - rcx
            dy = cy - rcy
            ux = dx * cosv - dy * sinv + rcx
            uy = dx * sinv + dy * cosv + rcy
            cx, cy = ux, uy

        s = max(1e-6, float(self.view_scale))
        return float(cx / s), float(cy / s)

    def _img_to_canvas_xy(self, x_img: float, y_img: float) -> Tuple[float, float]:
        """Map image FULL coords -> canvas coords, respecting deskew preview rotation."""
        cx = float(x_img * self.view_scale)
        cy = float(y_img * self.view_scale)

        ang = float(getattr(self, "view_angle_deg", 0.0))
        rcx = float(getattr(self, "rot_cx", 0.0))
        rcy = float(getattr(self, "rot_cy", 0.0))
        if abs(ang) > 1e-6:
            rad = np.deg2rad(ang)
            cosv = float(np.cos(rad))
            sinv = float(np.sin(rad))
            dx = cx - rcx
            dy = cy - rcy
            rx = dx * cosv - dy * sinv + rcx
            ry = dx * sinv + dy * cosv + rcy
            cx, cy = rx, ry

        return cx, cy

    def _img_to_canvas_pts(self, pts_full: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_full, dtype=np.float32)
        out = np.empty_like(pts, dtype=np.float32)
        for i in range(pts.shape[0]):
            out[i, 0], out[i, 1] = self._img_to_canvas_xy(float(pts[i, 0]), float(pts[i, 1]))
        return out

    # ---------- frame helpers (FULL coords) ----------
    def _ensure_rect_points_from_any_full(self):
        if self.points_full is None:
            return
        pts = np.asarray(self.points_full, dtype=np.float32)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return
        x0 = float(np.min(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        x1 = float(np.max(pts[:, 0]))
        y1 = float(np.max(pts[:, 1]))
        self.points_full = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    def _apply_square_lock_full(self):
        if self.frame_mode_var.get() != "rect" or not self.lock_square_var.get():
            return
        if self.points_full is None:
            return
        pts = np.asarray(self.points_full, dtype=np.float32)
        x0, y0 = pts[0]
        x1, y1 = pts[2]
        side = max(abs(x1 - x0), abs(y1 - y0))
        x1 = x0 + side if x1 >= x0 else x0 - side
        y1 = y0 + side if y1 >= y0 else y0 - side
        self.points_full = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    def _translate_points_full(self, dx: float, dy: float):
        if self.points_full is None:
            return
        pts = np.asarray(self.points_full, dtype=np.float32)
        pts[:, 0] += float(dx)
        pts[:, 1] += float(dy)
        pts[:, 0] = np.clip(pts[:, 0], 0, self.img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.img_h - 1)

        if self.frame_mode_var.get() == "rect":
            x0 = float(min(pts[0, 0], pts[2, 0]))
            y0 = float(min(pts[0, 1], pts[2, 1]))
            x1 = float(max(pts[0, 0], pts[2, 0]))
            y1 = float(max(pts[0, 1], pts[2, 1]))
            self.points_full = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            self._apply_square_lock_full()
        else:
            self.points_full = pts

    # ---------- UI ----------
    def _build_ui(self):
        panes = ttk.PanedWindow(self.root, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # LEFT: thumbnails
        left = ttk.Frame(panes, width=240)
        panes.add(left, weight=0)

        ttk.Label(left, text="Scans").pack(anchor="w", padx=6, pady=(6, 2))

        self.thumb_canvas = tk.Canvas(left, width=230, bg="#1f1f1f", highlightthickness=0)
        self.thumb_scroll = ttk.Scrollbar(left, orient="vertical", command=self.thumb_canvas.yview)
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scroll.set)

        self.thumb_scroll.pack(side="right", fill="y")
        self.thumb_canvas.pack(side="left", fill="both", expand=True)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg="#1f1f1f")
        self.thumb_window = self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")

        def _thumb_on_configure(event):
            self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))

        self.thumb_inner.bind("<Configure>", _thumb_on_configure)

        def _thumb_on_canvas_configure(event):
            # Keep the inner frame width in sync with the canvas width
            self.thumb_canvas.itemconfigure(self.thumb_window, width=event.width)

        self.thumb_canvas.bind("<Configure>", _thumb_on_canvas_configure)

        def _thumb_wheel(event):
            # Scroll thumbnails list
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            self.thumb_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        self.thumb_canvas.bind("<Enter>", lambda e: self.thumb_canvas.focus_set())
        self.thumb_canvas.bind("<MouseWheel>", _thumb_wheel)
        self.thumb_canvas.bind("<Button-4>", lambda e: self.thumb_canvas.yview_scroll(-3, "units"))
        self.thumb_canvas.bind("<Button-5>", lambda e: self.thumb_canvas.yview_scroll(+3, "units"))
        # Robust thumbnail scrolling: handle wheel events even when the cursor is over
        # child widgets inside the embedded frame (Windows often sends MouseWheel to the
        # widget under cursor, not to the canvas itself).
        def _is_descendant(widget, ancestor) -> bool:
            try:
                w = widget
                while w is not None:
                    if w == ancestor:
                        return True
                    w = getattr(w, "master", None)
            except Exception:
                return False
            return False

        def _thumb_wheel_global(event):
            try:
                w = self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
            except Exception:
                w = None
            if w is None or not _is_descendant(w, self.thumb_canvas):
                return  # let other handlers (e.g. center zoom) run

            delta = getattr(event, "delta", 0)
            if delta:
                self.thumb_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            return "break"

        def _thumb_button4_global(event):
            try:
                w = self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
            except Exception:
                w = None
            if w is None or not _is_descendant(w, self.thumb_canvas):
                return
            self.thumb_canvas.yview_scroll(-3, "units")
            return "break"

        def _thumb_button5_global(event):
            try:
                w = self.root.winfo_containing(self.root.winfo_pointerx(), self.root.winfo_pointery())
            except Exception:
                w = None
            if w is None or not _is_descendant(w, self.thumb_canvas):
                return
            self.thumb_canvas.yview_scroll(+3, "units")
            return "break"

        # Bind globally but only act when pointer is inside thumbnails pane.
        self.root.bind_all("<MouseWheel>", _thumb_wheel_global, add="+")
        self.root.bind_all("<Button-4>", _thumb_button4_global, add="+")
        self.root.bind_all("<Button-5>", _thumb_button5_global, add="+")


        # CENTER: canvas + scrollbars
        center = ttk.Frame(panes)
        panes.add(center, weight=1)

        self.canvas = tk.Canvas(center, bg="#222", highlightthickness=0)
        self.hbar = ttk.Scrollbar(center, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(center, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        # RIGHT: controls (scrollable)
        right_outer = ttk.Frame(panes)
        panes.add(right_outer, weight=0)

        self.right_canvas = tk.Canvas(right_outer, highlightthickness=0)
        self.right_scroll = ttk.Scrollbar(right_outer, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=self.right_scroll.set)

        self.right_scroll.pack(side="right", fill="y")
        self.right_canvas.pack(side="left", fill="both", expand=True)

        right = tk.Frame(self.right_canvas)
        self.right_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _right_on_configure(event):
            try:
                self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
            except Exception:
                pass

        right.bind("<Configure>", _right_on_configure)

        def _right_on_canvas_configure(event):
            # Keep the inner frame width in sync with the canvas width
            try:
                self.right_canvas.itemconfigure(self.right_window, width=event.width)
            except Exception:
                pass

        self.right_canvas.bind("<Configure>", _right_on_canvas_configure)

        def _right_wheel(event):
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            self.right_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            return "break"

        self.right_canvas.bind("<Enter>", lambda e: self.right_canvas.focus_set())
        self.right_canvas.bind("<MouseWheel>", _right_wheel)
        def _right_button4(event):
            self.right_canvas.yview_scroll(-3, "units")
            return "break"

        def _right_button5(event):
            self.right_canvas.yview_scroll(+3, "units")
            return "break"

        self.right_canvas.bind("<Button-4>", _right_button4)
        self.right_canvas.bind("<Button-5>", _right_button5)

        # --- Project ---
        top = tk.LabelFrame(right, text="Project")
        top.pack(fill="x", padx=8, pady=6)

        self.lbl_in = ttk.Label(top, text="Input: (not set)")
        self.lbl_in.pack(fill="x", padx=4, pady=2)
        self.lbl_out = ttk.Label(top, text="Output: (auto)")
        self.lbl_out.pack(fill="x", padx=4, pady=2)

        ttk.Button(top, text="Open input folder…", command=self.open_input_folder).pack(fill="x", padx=4, pady=4)

        ttk.Separator(right).pack(fill="x", padx=8, pady=8)

        # --- File (navigation + actions) ---
        file_box = tk.LabelFrame(right, text="File")
        file_box.pack(fill="x", padx=8, pady=4)

        self.status = tk.Label(file_box, text="Open input folder to start.", justify="left", anchor="w")
        self.status.pack(fill="x", padx=4, pady=4)

        nav = tk.Frame(file_box)
        nav.pack(fill="x", padx=4, pady=4)
        ttk.Button(nav, text="Prev (P)", command=self.prev).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(nav, text="Next (N)", command=self.next).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(file_box, text="Crop current scan (A)", command=self.crop_current).pack(fill="x", padx=4, pady=3)
        ttk.Button(file_box, text="Batch crop scans (B)", command=self.batch_crop).pack(fill="x", padx=4, pady=3)
        ttk.Button(file_box, text="Help", command=self.show_help).pack(fill="x", padx=4, pady=3)

        ttk.Separator(right).pack(fill="x", padx=8, pady=10)

        # --- Page split ---
        split = tk.LabelFrame(right, text="Page split")
        split.pack(fill="x", padx=8, pady=4)

        ttk.Checkbutton(split, text="Enable split (L/R)", variable=self.split_enabled_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Label(split, text="Active").grid(row=1, column=0, sticky="w", padx=4, pady=2)

        active_row = tk.Frame(split)
        active_row.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Radiobutton(active_row, text="L", value="L", variable=self.split_active_page_var, command=lambda: self._switch_active_page("L")).pack(side="left")
        ttk.Radiobutton(active_row, text="R", value="R", variable=self.split_active_page_var, command=lambda: self._switch_active_page("R")).pack(side="left")

        ttk.Label(split, text="Mode").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.OptionMenu(split, self.split_mode_var, self.split_mode_var.get(), "manual", "auto").grid(row=2, column=1, sticky="ew", padx=4, pady=2)

        split.columnconfigure(1, weight=1)
        ttk.Separator(split).grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 6))
        split_actions = tk.Frame(split)
        split_actions.grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(split_actions, text="Copy → next (Ctrl+C)", command=self.copy_split_to_next).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(split_actions, text="Copy+Mirror → next (Ctrl+M)", command=self.copy_split_mirror_to_next).pack(side="left", expand=True, fill="x", padx=2)

        split_actions2 = tk.Frame(split)
        split_actions2.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(split_actions2, text="Copy → end", command=self.copy_split_to_end).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(split_actions2, text="Copy → every 2nd → end", command=self.copy_split_every2_to_end).pack(side="left", expand=True, fill="x", padx=2)


        # --- Deskew ---
        desk = tk.LabelFrame(right, text="Deskew")
        desk.pack(fill="x", padx=8, pady=4)

        ttk.Label(desk, text="Manual deskew (deg)").grid(row=0, column=0, sticky="w", padx=4, pady=2)

        deskew_row = tk.Frame(desk)
        deskew_row.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        tk.Scale(
            deskew_row, from_=-10.0, to=10.0, resolution=0.1, orient="horizontal",
            variable=self.manual_deskew_var
        ).pack(side="top", fill="x")

        step_row = tk.Frame(deskew_row)
        step_row.pack(side="top", fill="x", pady=2)
        ttk.Button(step_row, text="-0.1°", command=lambda: self._deskew_step(-0.1)).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(step_row, text="+0.1°", command=lambda: self._deskew_step(+0.1)).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(step_row, text="0.0°", command=lambda: self.manual_deskew_var.set(0.0)).pack(
            side="left", expand=True, fill="x", padx=2
        )

        ttk.Checkbutton(desk, text="Auto deskew", variable=self.auto_deskew_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        ttk.Label(desk, text="Auto max (deg)").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(desk, from_=0.0, to=20.0, increment=0.5, textvariable=self.auto_deskew_max_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=2
        )
        desk.columnconfigure(1, weight=1)
        ttk.Separator(desk).grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 6))
        desk_actions = tk.Frame(desk)
        desk_actions.grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(desk_actions, text="Copy → next (Shift+C)", command=self.copy_deskew_to_next).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(desk_actions, text="Copy+Mirror → next (Shift+M)", command=self.copy_deskew_mirror_to_next).pack(side="left", expand=True, fill="x", padx=2)

        desk_actions2 = tk.Frame(desk)
        desk_actions2.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(desk_actions2, text="Copy → end", command=self.copy_deskew_to_end).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(desk_actions2, text="Copy → every 2nd → end", command=self.copy_deskew_every2_to_end).pack(side="left", expand=True, fill="x", padx=2)


        # --- Frames ---
        frm = tk.LabelFrame(right, text="Frames")
        frm.pack(fill="x", padx=8, pady=4)

        ttk.Label(frm, text="Start side").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.OptionMenu(frm, self.start_side_var, self.start_side_var.get(), "recto", "verso").grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(frm, text="Page side").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.OptionMenu(frm, self.page_side_var, self.page_side_var.get(), "recto", "verso", "unknown").grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Checkbutton(frm, text="Auto-toggle side (L/R alternation)", variable=self.auto_toggle_side_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        ttk.Checkbutton(frm, text="Auto-suggest frame on new scan", variable=self.auto_suggest_frame_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        ttk.Label(frm, text="Frame mode").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.OptionMenu(frm, self.frame_mode_var, self.frame_mode_var.get(), "rect", "quad").grid(
            row=4, column=1, sticky="ew", padx=4, pady=2
        )
        ttk.Checkbutton(frm, text="Lock square (rect)", variable=self.lock_square_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        frm.columnconfigure(1, weight=1)

        
        ttk.Separator(frm).grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 6))

        frm_actions = tk.Frame(frm)
        frm_actions.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(frm_actions, text="Copy → next (C)", command=self.copy_to_next).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(frm_actions, text="Copy+Mirror → next (M)", command=self.copy_mirror_to_next).pack(side="left", expand=True, fill="x", padx=2)


        frm_actions_end = tk.Frame(frm)
        frm_actions_end.grid(row=9, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(frm_actions_end, text="Copy → end", command=self.copy_frames_to_end).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(frm_actions_end, text="Copy → every 2nd → end", command=self.copy_frames_every2_to_end).pack(side="left", expand=True, fill="x", padx=2)

        frm_actions2 = tk.Frame(frm)
        frm_actions2.grid(row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(frm_actions2, text="Suggest frame (S)", command=self.suggest_frame_now).pack(side="left", expand=True, fill="x", padx=2)

# --- Margins ---
        margins = tk.LabelFrame(right, text="Margins")
        margins.pack(fill="x", padx=8, pady=4)

        ttk.Label(margins, text="Margin non-spine").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(margins, from_=0, to=2000, textvariable=self.margin_nonspine_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4, pady=2
        )
        ttk.Label(margins, text="Margin spine").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(margins, from_=0, to=2000, textvariable=self.margin_spine_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=2
        )
        margins.columnconfigure(1, weight=1)
        ttk.Separator(margins).grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 6))
        marg_actions = tk.Frame(margins)
        marg_actions.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(marg_actions, text="Copy → next (Alt+C)", command=self.copy_margins_to_next).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(marg_actions, text="Copy+Mirror → next (Alt+M)", command=self.copy_margins_mirror_to_next).pack(side="left", expand=True, fill="x", padx=2)


        marg_actions2 = tk.Frame(margins)
        marg_actions2.grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(marg_actions2, text="Copy → end", command=self.copy_margins_to_end).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(marg_actions2, text="Copy → every 2nd → end", command=self.copy_margins_every2_to_end).pack(side="left", expand=True, fill="x", padx=2)


        # --- Color management ---
        cm = tk.LabelFrame(right, text="Color management (LittleCMS2)")
        cm.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(cm, text="Enable CM", variable=self.cm_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        ttk.Label(cm, text="Dst ICC path").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(cm, textvariable=self.cm_to_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(cm, text="Browse…", command=self.browse_icc).grid(row=2, column=1, sticky="e", padx=4, pady=2)

        ttk.Label(cm, text="Intent").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.OptionMenu(
            cm, self.cm_intent_var, self.cm_intent_var.get(),
            "relative", "perceptual", "absolute", "saturation"
        ).grid(row=3, column=1, sticky="ew", padx=4, pady=2)

        ttk.Checkbutton(cm, text="BPC", variable=self.cm_bpc_var).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        cm.columnconfigure(1, weight=1)
# mouse bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # zoom wheel (Windows/mac)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Linux wheel
        self.canvas.bind("<Button-4>", lambda e: self._wheel_linux(+1, e))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_linux(-1, e))

    def _wheel_linux(self, direction: int, event):
        class E:
            delta = 120 if direction > 0 else -120
            x = event.x
            y = event.y
        self.on_mouse_wheel(E())

    def _deskew_step(self, delta: float):
        v = float(self.manual_deskew_var.get())
        v2 = round(v + float(delta), 1)
        v2 = max(-10.0, min(10.0, v2))
        self.manual_deskew_var.set(v2)

    def _bind_keys(self):
        self.root.bind("n", lambda e: self.next())
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("p", lambda e: self.prev())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("a", lambda e: self.crop_current())
        self.root.bind("b", lambda e: self.batch_crop())

        # Frames
        self.root.bind("c", lambda e: self.copy_to_next())
        self.root.bind("m", lambda e: self.copy_mirror_to_next())

        # Page split
        self.root.bind("<Control-c>", lambda e: self.copy_split_to_next())
        self.root.bind("<Control-m>", lambda e: self.copy_split_mirror_to_next())

        # Deskew
        self.root.bind("<Shift-C>", lambda e: self.copy_deskew_to_next())
        self.root.bind("<Shift-M>", lambda e: self.copy_deskew_mirror_to_next())

        # Margins
        self.root.bind("<Alt-c>", lambda e: self.copy_margins_to_next())
        self.root.bind("<Alt-m>", lambda e: self.copy_margins_mirror_to_next())

        self.root.bind("s", lambda e: self.suggest_frame_now())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # ---------- dialogs ----------
    def open_input_folder(self):
        folder = filedialog.askdirectory(title="Select input folder with TIFF scans")
        if not folder:
            return
        files = collect_tiffs(folder)
        if not files:
            messagebox.showerror("No TIFFs", "No .tif/.tiff files found in selected folder.")
            return

        self.input_dir = folder
        self.output_dir = default_output_dir(folder)
        self._ensure_output_dirs()

        self.files = files
        # Cache thumbnails up to the set size (works well for 500–1000 scans)
        self.thumb_cache.set_capacity(min(1100, max(200, len(files))))
        self.idx = 0
        self.zoom = 1.0
        self.page_side_var.set(self.start_side_var.get())

        self.lbl_in.config(text=f"Input: {self.input_dir}")
        self.lbl_out.config(text=f"Output: {self.output_dir}")

        self._last_rgb8_for_suggest = None
        self._last_points_full = None
        self._last_settings = None

        self.pages.clear()
        self.points_full = None
        self._auto_deskew_angle_cache.clear()

        self._thumb_jobs_cancel = False
        self._build_thumbnails()

        self.load_current()

    def browse_icc(self):
        f = filedialog.askopenfilename(
            title="Select destination ICC profile",
            filetypes=[("ICC profiles", "*.icc *.ICM *.icm *.ICC"), ("All files", "*.*")]
        )
        if f:
            self.cm_to_var.set(f)

    # ---------- thumbnails ----------
    def _build_thumbnails(self):
        for w in self.thumb_inner.winfo_children():
            w.destroy()
        self.thumb_widgets.clear()
        self.thumb_rows.clear()

        for i, p in enumerate(self.files):
            row = tk.Frame(self.thumb_inner, bg="#1f1f1f", highlightthickness=1, highlightbackground="#1f1f1f")
            row.pack(fill="x", padx=6, pady=6)
            self.thumb_rows[p] = row

            lbl = tk.Label(
                row,
                text=Path(p).name,
                compound="top",
                justify="center",
                bg="#1f1f1f",
                fg="#dddddd",
                wraplength=200,
            )
            lbl.pack(fill="x", padx=6, pady=6)

            def _mk(idx=i):
                return lambda e: self._jump_to_index(idx)

            lbl.bind("<Button-1>", _mk(i))
            row.bind("<Button-1>", _mk(i))

            self.thumb_widgets[p] = lbl
            self._enqueue_thumb(p)

    def _jump_to_index(self, idx: int):
        if not self.files:
            return
        self._remember_last()
        self.idx = int(np.clip(idx, 0, len(self.files) - 1))
        self.load_current()

    def _enqueue_thumb(self, path: str):
        cached = self.thumb_cache.get(path)
        if cached is not None:
            self._apply_thumb_to_widget(path, cached)
            return

        def job():
            if self._thumb_jobs_cancel:
                return
            rgb8 = load_thumb_rgb8(path, self.thumb_size)
            if rgb8 is None:
                return
            pil = Image.fromarray(rgb8)
            tkimg = ImageTk.PhotoImage(pil)

            def on_ui():
                if self._thumb_jobs_cancel:
                    return
                self.thumb_cache.put(path, tkimg)
                self._apply_thumb_to_widget(path, tkimg)

            self.root.after(0, on_ui)

        self.thumb_executor.submit(job)

    def _apply_thumb_to_widget(self, path: str, tkimg: ImageTk.PhotoImage):
        lbl = self.thumb_widgets.get(path)
        if not lbl:
            return
        lbl.image = tkimg  # prevent GC
        lbl.configure(image=tkimg)

    def _highlight_current_thumb(self):
        cur = self._current_path() if self.files else ""
        for p, row in self.thumb_rows.items():
            lbl = self.thumb_widgets.get(p)
            if lbl is None:
                continue
            if p == cur:
                row.configure(bg="#2a2a2a", highlightbackground="#6aa9ff")
                lbl.configure(bg="#2a2a2a", fg="#ffffff")
            else:
                row.configure(bg="#1f1f1f", highlightbackground="#1f1f1f")
                lbl.configure(bg="#1f1f1f", fg="#dddddd")
        self._scroll_thumb_to_current()

    def _scroll_thumb_to_current(self):
        """Ensure the current thumbnail row is visible in the left panel."""
        if not self.files:
            return
        cur = self._current_path()
        row = self.thumb_rows.get(cur)
        if row is None:
            return

        def _do():
            try:
                self.thumb_canvas.update_idletasks()
                total_h = max(1, int(self.thumb_inner.winfo_height()))
                canvas_h = max(1, int(self.thumb_canvas.winfo_height()))

                y0 = int(row.winfo_y())
                y1 = y0 + int(row.winfo_height())

                f0, _ = self.thumb_canvas.yview()
                vy0 = int(f0 * total_h)
                vy1 = vy0 + canvas_h

                if y0 < vy0:
                    self.thumb_canvas.yview_moveto(y0 / total_h)
                elif y1 > vy1:
                    self.thumb_canvas.yview_moveto(max(0.0, (y1 - canvas_h) / total_h))
            except Exception:
                pass

        self.root.after(0, _do)


    # ---------- toggling ----------
    def _toggle_side_if_enabled(self):
        if not self.auto_toggle_side_var.get():
            return
        old = self.page_side_var.get()
        if old == "recto":
            self.page_side_var.set("verso")
        elif old == "verso":
            self.page_side_var.set("recto")

    # ---------- suggest helpers ----------
    def _rgb8_for_suggest(self, rgb: np.ndarray, maxdim: int = 1200) -> np.ndarray:
        if rgb.dtype == np.uint16:
            rgb8 = (rgb >> 8).astype(np.uint8)
        else:
            rgb8 = rgb.astype(np.uint8)
        h, w = rgb8.shape[:2]
        s = min(1.0, float(maxdim) / max(h, w))
        if s < 1.0:
            rgb8 = cv2.resize(rgb8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return rgb8

    def _auto_suggest_from_last(self, mirror: bool):
        if self._last_rgb8_for_suggest is None or self._last_points_full is None:
            return
        if self.img_rgb is None:
            return

        cur_rgb8 = self._rgb8_for_suggest(self.img_rgb, maxdim=1200)
        prev_rgb8 = self._last_rgb8_for_suggest
        prev_pts = np.asarray(self._last_points_full, dtype=np.float32).copy()

        if mirror:
            prev_rgb8 = np.fliplr(prev_rgb8).copy()
            prev_pts = mirror_points_horiz(prev_pts, width_px=self.img_w)

        suggested = suggest_frame_translation(cur_rgb8, prev_rgb8, prev_pts)

        suggested[:, 0] = np.clip(suggested[:, 0], 0, self.img_w - 1)
        suggested[:, 1] = np.clip(suggested[:, 1], 0, self.img_h - 1)
        self.points_full = suggested.astype(np.float32)

        if self._last_settings is not None:
            self.margin_nonspine_var.set(int(self._last_settings.get("margin_nonspine", self.margin_nonspine_var.get())))
            self.margin_spine_var.set(int(self._last_settings.get("margin_spine", self.margin_spine_var.get())))
            self.frame_mode_var.set(self._last_settings.get("frame_mode", self.frame_mode_var.get()))
            self.lock_square_var.set(bool(self._last_settings.get("lock_square", self.lock_square_var.get())))
            self.manual_deskew_var.set(float(self._last_settings.get("manual_deskew_deg", self.manual_deskew_var.get())))
            self.auto_deskew_var.set(bool(self._last_settings.get("auto_deskew", self.auto_deskew_var.get())))
            self.auto_deskew_max_var.set(float(self._last_settings.get("auto_deskew_max_deg", self.auto_deskew_max_var.get())))
            self.cm_enabled_var.set(bool(self._last_settings.get("cm_enabled", self.cm_enabled_var.get())))
            self.cm_to_var.set(str(self._last_settings.get("cm_to_icc", self.cm_to_var.get())))
            self.cm_intent_var.set(self._last_settings.get("cm_intent", self.cm_intent_var.get()))
            self.cm_bpc_var.set(bool(self._last_settings.get("cm_bpc", self.cm_bpc_var.get())))

    # ---------- load/render ----------
    def load_current(self):
        if not self.files:
            return

        path = self._current_path()
        rgb, meta = read_tiff_rgb(path)
        self.img_rgb = rgb
        self.meta = meta
        self.img_h, self.img_w = rgb.shape[:2]

        # load frame if exists
        fpath = self._frame_path(path)
        had_frame = False
        self.points_full = None
        self.pages.clear()

        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                self.page_side_var.set(data.get("page_side", self.page_side_var.get()))
                self.frame_mode_var.set(data.get("frame_mode", self.frame_mode_var.get()))
                self.lock_square_var.set(bool(data.get("lock_square", self.lock_square_var.get())))

                # margins: new keys, but accept old pad_* if present
                self.margin_nonspine_var.set(int(data.get("margin_nonspine", data.get("pad_nonspine", self.margin_nonspine_var.get()))))
                self.margin_spine_var.set(int(data.get("margin_spine", data.get("pad_spine", self.margin_spine_var.get()))))

                self.manual_deskew_var.set(float(data.get("manual_deskew_deg", self.manual_deskew_var.get())))
                self.auto_deskew_var.set(bool(data.get("auto_deskew", self.auto_deskew_var.get())))
                self.auto_deskew_max_var.set(float(data.get("auto_deskew_max_deg", self.auto_deskew_max_var.get())))
                self.cm_enabled_var.set(bool(data.get("cm_enabled", self.cm_enabled_var.get())))
                self.cm_to_var.set(str(data.get("cm_to_icc", self.cm_to_var.get())))
                self.cm_intent_var.set(data.get("cm_intent", self.cm_intent_var.get()))
                self.cm_bpc_var.set(bool(data.get("cm_bpc", self.cm_bpc_var.get())))

                split_enabled = bool(data.get("split_enabled", False))
                self.split_enabled_var.set(split_enabled)
                self.split_mode_var.set(str(data.get("split_mode", self.split_mode_var.get())))
                self.split_active_page_var.set(str(data.get("split_active_page", self.split_active_page_var.get())))

                if split_enabled:
                    self._ensure_pages_ready()
                    pages = data.get("pages", {})
                    for k in ("L", "R"):
                        q = np.array(pages.get(k, {}).get("quad_xy", []), dtype=np.float32)
                        if q.shape == (4, 2):
                            self.pages[k]["quad"] = q
                    self._pull_active_page_to_vars()
                    had_frame = (self.pages.get("L", {}).get("quad") is not None) or (self.pages.get("R", {}).get("quad") is not None)
                else:
                    quad_full = np.array(data.get("quad_xy", []), dtype=np.float32)
                    if quad_full.shape == (4, 2):
                        self.points_full = quad_full
                        had_frame = True
            except Exception:
                had_frame = False

        if (not had_frame) and self.auto_suggest_frame_var.get():
            self._auto_suggest_from_last(mirror=False)

        if self.points_full is None:
            margin = max(80, int(round(min(self.img_h, self.img_w) * 0.03)))
            self.points_full = np.array([
                [margin, margin],
                [self.img_w - margin, margin],
                [self.img_w - margin, self.img_h - margin],
                [margin, self.img_h - margin],
            ], dtype=np.float32)

        if self.split_enabled_var.get():
            self._ensure_pages_ready()
            if self.pages["L"].get("quad") is None:
                self.pages["L"]["quad"] = self.points_full.copy()
            if self.pages["R"].get("quad") is None:
                self.pages["R"]["quad"] = self.points_full.copy()
            self._pull_active_page_to_vars()

        if self.frame_mode_var.get() == "rect":
            self._ensure_rect_points_from_any_full()
            self._apply_square_lock_full()
            if self.split_enabled_var.get():
                self._push_vars_to_active_page()

        self._render()
        self._schedule_autosave()
        self._highlight_current_thumb()

    def _get_auto_deskew_angle(self) -> float:
        """Estimate deskew angle for preview (cached per file). Returns 0 if disabled/unknown."""
        if not self.auto_deskew_var.get() or self.img_rgb is None or not self.files:
            return 0.0
        path = self._current_path()
        cached = self._auto_deskew_angle_cache.get(path, None)
        if cached is not None:
            return float(cached)

        try:
            rgb = self.img_rgb
            if rgb.dtype == np.uint16:
                rgb8 = (rgb >> 8).astype(np.uint8)
            else:
                rgb8 = rgb.astype(np.uint8)

            h, w = rgb8.shape[:2]
            maxdim = 1400
            s = min(1.0, float(maxdim) / max(h, w))
            if s < 1.0:
                rgb8 = cv2.resize(rgb8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            bgr8 = rgb8[:, :, ::-1].copy()
            ang = estimate_text_angle_deg(bgr8)
            maxdeg = float(self.auto_deskew_max_var.get())
            if ang is None or abs(float(ang)) > maxdeg:
                angv = 0.0
            else:
                angv = float(ang)
        except Exception:
            angv = 0.0

        self._auto_deskew_angle_cache[path] = angv
        return float(angv)

    def _view_photo_cache_get(self, key: Tuple[str, int, int, float]) -> Optional[ImageTk.PhotoImage]:
        v = self._view_photo_cache.get(key)
        if v is None:
            return None
        try:
            self._view_photo_cache_order.remove(key)
        except ValueError:
            pass
        self._view_photo_cache_order.append(key)
        return v

    def _view_photo_cache_put(self, key: Tuple[str, int, int, float], value: ImageTk.PhotoImage) -> None:
        if key in self._view_photo_cache:
            try:
                self._view_photo_cache_order.remove(key)
            except ValueError:
                pass
        self._view_photo_cache[key] = value
        self._view_photo_cache_order.append(key)
        while len(self._view_photo_cache_order) > self._view_photo_cache_cap:
            old = self._view_photo_cache_order.pop(0)
            self._view_photo_cache.pop(old, None)

    def _draw_one(self, pts_full: np.ndarray, label: str, outline: str):
        pts_canvas = self._img_to_canvas_pts(pts_full)

        # polygon
        self.canvas.create_polygon(pts_canvas.reshape(-1).tolist(), outline=outline, fill="", width=3)

        # corner handles + indices
        for i in range(4):
            x, y = float(pts_canvas[i, 0]), float(pts_canvas[i, 1])
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#ffcc00", outline="#000", width=2)
            self.canvas.create_text(x + 12, y, text=str(i + 1), fill="#ffcc00", anchor="w")

        # label at top edge midpoint
        mx = float((pts_canvas[0, 0] + pts_canvas[1, 0]) * 0.5)
        my = float((pts_canvas[0, 1] + pts_canvas[1, 1]) * 0.5)
        self.canvas.create_text(mx, my - 18, text=label, fill=outline, font=("Segoe UI", 12, "bold"))

    def _render(self):
        if self.img_rgb is None:
            return
        self._update_view_scale()

        if self.img_rgb.dtype == np.uint16:
            base8 = (self.img_rgb >> 8).astype(np.uint8)
        else:
            base8 = self.img_rgb.astype(np.uint8)

        out_w = max(1, int(round(self.img_w * self.view_scale)))
        out_h = max(1, int(round(self.img_h * self.view_scale)))

        # deskew preview rotation (manual + optional auto)
        md = float(self.manual_deskew_var.get())
        ad = float(self._get_auto_deskew_angle()) if self.auto_deskew_var.get() else 0.0
        angle = -(md + ad)
        angle_key = float(round(angle, 1))

        # rotation center in canvas coords (scrollregion coords)
        self.rot_cx = out_w * 0.5
        self.rot_cy = out_h * 0.5
        self.view_angle_deg = angle_key

        cache_key = (self._current_path() if self.files else "", out_w, out_h, angle_key)
        cached_photo = self._view_photo_cache_get(cache_key)
        if cached_photo is not None:
            self.tk_img = cached_photo
        else:
            prev = cv2.resize(base8, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if abs(angle_key) > 1e-6:
                M = cv2.getRotationMatrix2D((self.rot_cx, self.rot_cy), angle_key, 1.0)
                prev = cv2.warpAffine(
                    prev, M, (out_w, out_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

            pil = Image.fromarray(prev)
            self.tk_img = ImageTk.PhotoImage(pil)
            self._view_photo_cache_put(cache_key, self.tk_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, out_w, out_h))

        # Draw frames
        if self.split_enabled_var.get():
            self._ensure_pages_ready()
            qL = self.pages.get("L", {}).get("quad")
            qR = self.pages.get("R", {}).get("quad")
            if qL is not None:
                self._draw_one(np.asarray(qL, dtype=np.float32), "Left", "#ff4040")
            if qR is not None:
                self._draw_one(np.asarray(qR, dtype=np.float32), "Right", "#4aa3ff")
        else:
            if self.points_full is not None:
                if self.frame_mode_var.get() == "rect":
                    self._ensure_rect_points_from_any_full()
                    self._apply_square_lock_full()
                self._draw_one(np.asarray(self.points_full, dtype=np.float32), "Frame", "#ff0000")

        self._update_status()

    def _update_status(self):
        if not self.files or self.meta is None:
            self.status.config(text="Open input folder to start.")
            return
        p = Path(self._current_path())
        icc_len = len(self.meta.get("icc_profile") or b"")
        xmp_len = len(self.meta.get("xmp") or b"")
        split = self.split_enabled_var.get()
        self.status.config(text=(
            f"File: {p.name} ({self.idx + 1}/{len(self.files)})\n"
            f"Size: {self.img_w}x{self.img_h}  ICC(in): {icc_len}B  XMP(in): {xmp_len}B\n"
            f"Start: {self.start_side_var.get()}  Side: {self.page_side_var.get()}  Toggle: {self.auto_toggle_side_var.get()}\n"
            f"Suggest: {self.auto_suggest_frame_var.get()}  Zoom: {self.zoom:.2f}x (fit={self.fit_scale:.3f})\n"
            f"Split: {split}  Active: {self.split_active_page_var.get()}\n"
            f"Frame: {self.frame_mode_var.get()}  Square: {self.lock_square_var.get()}\n"
            f"Margins(non/spine): {self.margin_nonspine_var.get()} / {self.margin_spine_var.get()}\n"
            f"Manual deskew: {self.manual_deskew_var.get():.1f}°  Auto: {self.auto_deskew_var.get()} (max {self.auto_deskew_max_var.get()}°)\n"
            f"CM: {self.cm_enabled_var.get()} to: {self.cm_to_var.get()} intent: {self.cm_intent_var.get()} bpc: {self.cm_bpc_var.get()}\n"
        ))

    def show_help(self):
        msg = (
            f"Script: {SCRIPT_VERSION}\n"
            "Author: Jan Houserek\n"
            "License: GPLv3\n"
            "\n"
            "Mouse / editing\n"
            "  • Mouse wheel: zoom (over image)\n"
            "  • Drag corner handles: reshape frame\n"
            "  • Drag inside frame: move frame\n"
            "\n"
            "Navigation\n"
            "  • N / Right: next scan\n"
            "  • P / Left: previous scan\n"
            "\n"
            "Cropping\n"
            "  • A: crop current scan\n"
            "  • B: batch crop scans (only scans with saved frames)\n"
            "\n"
            "Copy → next shortcuts\n"
            "  Frames: C (copy), M (copy+mirror)\n"
            "  Page split: Ctrl+C (copy), Ctrl+M (copy+mirror)\n"
            "  Deskew: Shift+C (copy), Shift+M (copy+mirror)\n"
            "  Margins: Alt+C (copy), Alt+M (copy+mirror)\n"
            "\n"
            "Suggest\n"
            "  • S: suggest frame\n"
            "\n"
            "Exit\n"
            "  • Esc: quit"
        )
        messagebox.showinfo("Help", msg)


    # ---------- zoom ----------
    def on_mouse_wheel(self, event):
        if self.img_rgb is None:
            return

        ix0, iy0 = self._canvas_to_img(event.x, event.y)

        step = 1.1 if event.delta > 0 else (1.0 / 1.1)
        self.zoom = float(np.clip(self.zoom * step, 0.1, 8.0))

        self._render()

        bbox = self.canvas.bbox("all")
        if not bbox:
            return
        full_w = max(1.0, float(bbox[2]))
        full_h = max(1.0, float(bbox[3]))

        cx, cy = self._img_to_canvas_xy(ix0, iy0)
        self.canvas.xview_moveto(max(0.0, (cx - event.x) / full_w))
        self.canvas.yview_moveto(max(0.0, (cy - event.y) / full_h))

    # ---------- mouse (frame edit) ----------
    def _active_pts_full(self) -> Optional[np.ndarray]:
        if self.split_enabled_var.get():
            self._ensure_pages_ready()
            key = self._active_page_key()
            q = self.pages.get(key, {}).get("quad")
            if q is None:
                return None
            return np.asarray(q, dtype=np.float32).copy()
        if self.points_full is None:
            return None
        return np.asarray(self.points_full, dtype=np.float32).copy()

    def on_mouse_down(self, event):
        if self.img_rgb is None:
            return

        # ensure active vars loaded (split)
        if self.split_enabled_var.get():
            self._pull_active_page_to_vars()

        if self.points_full is None:
            return

        pts_full = np.asarray(self.points_full, dtype=np.float32)
        pts_canvas = self._img_to_canvas_pts(pts_full)

        cx = float(self.canvas.canvasx(event.x))
        cy = float(self.canvas.canvasy(event.y))

        d = np.sqrt((pts_canvas[:, 0] - cx) ** 2 + (pts_canvas[:, 1] - cy) ** 2)
        i = int(np.argmin(d))
        if d[i] < 20:
            self.drag_i = i
            self.drag_mode = "corner"
            self.drag_last_xy_img = self._canvas_to_img(event.x, event.y)
            return

        x_img, y_img = self._canvas_to_img(event.x, event.y)
        if point_in_polygon(x_img, y_img, pts_full):
            self.drag_i = None
            self.drag_mode = "move"
            self.drag_last_xy_img = (x_img, y_img)
            return

        self.drag_mode = None
        self.drag_last_xy_img = None
        self.drag_i = None

    def on_mouse_drag(self, event):
        if self.drag_mode is None or self.img_rgb is None or self.points_full is None:
            return

        x_img, y_img = self._canvas_to_img(event.x, event.y)
        x_img = float(np.clip(x_img, 0, self.img_w - 1))
        y_img = float(np.clip(y_img, 0, self.img_h - 1))

        if self.drag_mode == "move":
            if self.drag_last_xy_img is None:
                self.drag_last_xy_img = (x_img, y_img)
                return
            lx, ly = self.drag_last_xy_img
            dx, dy = (x_img - lx), (y_img - ly)
            self._translate_points_full(dx, dy)
            if self.split_enabled_var.get():
                self._push_vars_to_active_page()
            self.drag_last_xy_img = (x_img, y_img)
            self._render()
            self._schedule_autosave()
            return

        if self.drag_mode == "corner" and self.drag_i is not None:
            pts = np.asarray(self.points_full, dtype=np.float32)

            if self.frame_mode_var.get() == "rect":
                tl = pts[0].copy()
                tr = pts[1].copy()
                br = pts[2].copy()
                bl = pts[3].copy()

                x0 = float(min(tl[0], bl[0]))
                x1 = float(max(tr[0], br[0]))
                y0 = float(min(tl[1], tr[1]))
                y1 = float(max(bl[1], br[1]))

                if self.drag_i == 0:      # TL
                    x0, y0 = x_img, y_img
                elif self.drag_i == 1:    # TR
                    x1, y0 = x_img, y_img
                elif self.drag_i == 2:    # BR
                    x1, y1 = x_img, y_img
                elif self.drag_i == 3:    # BL
                    x0, y1 = x_img, y_img

                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                self.points_full = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                self._apply_square_lock_full()
                if self.split_enabled_var.get():
                    self._push_vars_to_active_page()
            else:
                pts[self.drag_i] = [x_img, y_img]
                self.points_full = pts
                if self.split_enabled_var.get():
                    self._push_vars_to_active_page()

            self._render()
            self._schedule_autosave()

    def on_mouse_up(self, event):
        self.drag_i = None
        self.drag_mode = None
        self.drag_last_xy_img = None

    # ---------- navigation ----------
    def _remember_last(self):
        if self.img_rgb is None:
            return
        # Use current active vars as "last"
        if self.split_enabled_var.get():
            self._pull_active_page_to_vars()

        if self.points_full is None:
            return

        self._last_rgb8_for_suggest = self._rgb8_for_suggest(self.img_rgb, maxdim=1200)
        self._last_points_full = np.asarray(self.points_full, dtype=np.float32).copy()
        self._last_settings = self._current_frame_data(include_image=False)

    def next(self):
        if not self.files:
            return
        self._remember_last()
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self._toggle_side_if_enabled()
            self.load_current()

    def prev(self):
        if not self.files:
            return
        self._remember_last()
        if self.idx > 0:
            self.idx -= 1
            self._toggle_side_if_enabled()
            self.load_current()

    # ---------- copy/suggest ----------
    
    def _copy_to_next(self, scope: str, mirror: bool):
        """Copy selected scope from current scan to next scan.

        scope:
          - "frame": copy frame geometry (and mirror if requested)
          - "split": copy page split settings only
          - "deskew": copy deskew settings only (mirror negates manual angle)
          - "margins": copy margins only
        """
        if not self.files or self.idx >= len(self.files) - 1:
            return

        # Save current state (including split active page quad) and remember values
        if self.split_enabled_var.get():
            self._push_vars_to_active_page()
        self.save_frame(silent=True)
        self._remember_last()

        # Advance
        self.idx += 1
        self._toggle_side_if_enabled()
        self.load_current()

        # Apply remembered settings for the requested scope
        if self._last_settings is None:
            self._last_settings = {}

        if scope == "frame":
            if self._last_points_full is not None:
                pts = np.asarray(self._last_points_full, dtype=np.float32).copy()
                if mirror:
                    pts = mirror_points_horiz(pts, width_px=self.img_w)
                self.points_full = pts.astype(np.float32, copy=False)
                if self.split_enabled_var.get():
                    self._ensure_pages_ready()
                    k = self._active_page_key()
                    self.pages[k]["quad"] = self.points_full.copy()

            if self.frame_mode_var.get() == "rect":
                self._ensure_rect_points_from_any_full()
                self._apply_square_lock_full()
                if self.split_enabled_var.get():
                    self._push_vars_to_active_page()

        elif scope == "split":
            # Copy split enable/mode/active page only
            self.split_enabled_var.set(bool(self._last_settings.get("split_enabled", self.split_enabled_var.get())))
            self.split_mode_var.set(str(self._last_settings.get("split_mode", self.split_mode_var.get())))
            ap = str(self._last_settings.get("split_active_page", self.split_active_page_var.get()))
            if mirror:
                ap = "R" if ap.upper().startswith("L") else "L"
            self.split_active_page_var.set(ap)

        elif scope == "deskew":
            md = float(self._last_settings.get("manual_deskew_deg", self.manual_deskew_var.get()))
            if mirror:
                md = -md
            self.manual_deskew_var.set(md)
            self.auto_deskew_var.set(bool(self._last_settings.get("auto_deskew", self.auto_deskew_var.get())))
            self.auto_deskew_max_var.set(float(self._last_settings.get("auto_deskew_max_deg", self.auto_deskew_max_var.get())))

        elif scope == "margins":
            self.margin_nonspine_var.set(int(self._last_settings.get("margin_nonspine", self.margin_nonspine_var.get())))
            self.margin_spine_var.set(int(self._last_settings.get("margin_spine", self.margin_spine_var.get())))

        else:
            return

        # Ensure preview updates + persist
        self._render()
        self.save_frame(silent=True)

    def copy_to_next(self):
        self._copy_to_next(scope="frame", mirror=False)

    def copy_mirror_to_next(self):
        self._copy_to_next(scope="frame", mirror=True)

    def copy_split_to_next(self):
        self._copy_to_next(scope="split", mirror=False)

    def copy_split_mirror_to_next(self):
        self._copy_to_next(scope="split", mirror=True)

    def copy_deskew_to_next(self):
        self._copy_to_next(scope="deskew", mirror=False)

    def copy_deskew_mirror_to_next(self):
        self._copy_to_next(scope="deskew", mirror=True)

    def copy_margins_to_next(self):
        self._copy_to_next(scope="margins", mirror=False)

    def copy_margins_mirror_to_next(self):
        self._copy_to_next(scope="margins", mirror=True)

    # ---------- batch copy (to end) ----------
    def _copy_scope_to_end(self, scope: str, step: int = 1):
        """Copy selected scope from current scan to scans until the end.

        step=1 -> every scan from next to end
        step=2 -> every second scan (same parity as current) from current+2 to end
        """
        if not self.files:
            return
        step = 1 if int(step) != 2 else 2

        # Ensure current is saved first
        if self.split_enabled_var.get():
            self._push_vars_to_active_page()
        self.save_frame(silent=True)

        # Source snapshot (current scan)
        src = self._current_frame_data(include_image=False)

        start = self.idx + step
        if start >= len(self.files):
            return

        applied = 0
        failed = 0
        for j in range(start, len(self.files), step):
            img_path = self.files[j]
            fpath = self._frame_path(img_path)

            try:
                if fpath.exists():
                    dst = json.loads(fpath.read_text(encoding="utf-8"))
                else:
                    dst = {
                        "app": APP_NAME,
                        "script_version": SCRIPT_VERSION,
                        "image": str(img_path),
                    }

                if scope == "frame":
                    dst["frame_mode"] = src.get("frame_mode", dst.get("frame_mode", "rect"))
                    dst["lock_square"] = bool(src.get("lock_square", dst.get("lock_square", False)))

                    if bool(src.get("split_enabled", False)):
                        dst["split_enabled"] = True
                        dst["split_mode"] = src.get("split_mode", dst.get("split_mode", "manual"))
                        dst["split_active_page"] = src.get("split_active_page", dst.get("split_active_page", "L"))
                        dst["pages"] = src.get("pages", {})
                        dst.pop("quad_xy", None)
                    else:
                        dst["split_enabled"] = False
                        dst.pop("pages", None)
                        dst["quad_xy"] = src.get("quad_xy", dst.get("quad_xy", []))

                elif scope == "split":
                    dst["split_enabled"] = bool(src.get("split_enabled", dst.get("split_enabled", False)))
                    dst["split_mode"] = src.get("split_mode", dst.get("split_mode", "manual"))
                    dst["split_active_page"] = src.get("split_active_page", dst.get("split_active_page", "L"))

                elif scope == "deskew":
                    dst["manual_deskew_deg"] = float(src.get("manual_deskew_deg", dst.get("manual_deskew_deg", 0.0)))
                    dst["auto_deskew"] = bool(src.get("auto_deskew", dst.get("auto_deskew", False)))
                    dst["auto_deskew_max_deg"] = float(src.get("auto_deskew_max_deg", dst.get("auto_deskew_max_deg", 3.0)))

                elif scope == "margins":
                    dst["margin_nonspine"] = int(src.get("margin_nonspine", dst.get("margin_nonspine", 0)))
                    dst["margin_spine"] = int(src.get("margin_spine", dst.get("margin_spine", 0)))

                else:
                    continue

                fpath.write_text(json.dumps(dst, ensure_ascii=False, indent=2), encoding="utf-8")
                applied += 1
            except Exception as e:
                print(f"[WARN] Copy-to-end failed for {img_path}: {e}")
                failed += 1

        msg = f"Scope: {scope}\nStep: {step}\nApplied: {applied}\nFailed: {failed}"
        messagebox.showinfo("Copy to end", msg)

    def copy_frames_to_end(self):
        self._copy_scope_to_end(scope="frame", step=1)

    def copy_frames_every2_to_end(self):
        self._copy_scope_to_end(scope="frame", step=2)

    def copy_split_to_end(self):
        self._copy_scope_to_end(scope="split", step=1)

    def copy_split_every2_to_end(self):
        self._copy_scope_to_end(scope="split", step=2)

    def copy_deskew_to_end(self):
        self._copy_scope_to_end(scope="deskew", step=1)

    def copy_deskew_every2_to_end(self):
        self._copy_scope_to_end(scope="deskew", step=2)

    def copy_margins_to_end(self):
        self._copy_scope_to_end(scope="margins", step=1)

    def copy_margins_every2_to_end(self):
        self._copy_scope_to_end(scope="margins", step=2)


    def suggest_frame_now(self):
        if not self.files:
            return
        self._remember_last()
        self._auto_suggest_from_last(mirror=False)
        if self.frame_mode_var.get() == "rect":
            self._ensure_rect_points_from_any_full()
            self._apply_square_lock_full()
        if self.split_enabled_var.get():
            self._push_vars_to_active_page()
        self._render()
        self.save_frame(silent=True)

    # ---------- frame save ----------
    def _current_frame_data(self, include_image: bool = True) -> Dict[str, Any]:
        if self.split_enabled_var.get():
            # ensure active vars are stored
            self._push_vars_to_active_page()
            self._ensure_pages_ready()

        if self.points_full is None and not self.split_enabled_var.get():
            raise ValueError("No frame points.")

        data: Dict[str, Any] = {
            "app": APP_NAME,
            "script_version": SCRIPT_VERSION,
            "page_side": self.page_side_var.get(),
            "start_side": self.start_side_var.get(),
            "auto_toggle_side": bool(self.auto_toggle_side_var.get()),
            "auto_suggest_frame": bool(self.auto_suggest_frame_var.get()),
            "frame_mode": self.frame_mode_var.get(),
            "lock_square": bool(self.lock_square_var.get()),
            "margin_nonspine": int(self.margin_nonspine_var.get()),
            "margin_spine": int(self.margin_spine_var.get()),
            "manual_deskew_deg": float(self.manual_deskew_var.get()),
            "auto_deskew": bool(self.auto_deskew_var.get()),
            "auto_deskew_max_deg": float(self.auto_deskew_max_var.get()),
            "cm_enabled": bool(self.cm_enabled_var.get()),
            "cm_to_icc": str(self.cm_to_var.get()).strip(),
            "cm_intent": self.cm_intent_var.get(),
            "cm_bpc": bool(self.cm_bpc_var.get()),
            "split_enabled": bool(self.split_enabled_var.get()),
            "split_mode": self.split_mode_var.get(),
            "split_active_page": self.split_active_page_var.get(),
        }

        if self.split_enabled_var.get():
            pages_out: Dict[str, Any] = {}
            for k in ("L", "R"):
                q = self.pages.get(k, {}).get("quad")
                pages_out[k] = {"quad_xy": (np.asarray(q, dtype=np.float32).tolist() if q is not None else [])}
            data["pages"] = pages_out
        else:
            quad_full = np.asarray(self.points_full, dtype=np.float32).astype(float)
            data["quad_xy"] = quad_full.tolist()

        if include_image and self.files:
            data["image"] = str(self._current_path())
        return data

    def save_frame(self, silent: bool = False):
        if not self.files:
            return
        self._ensure_output_dirs()
        img_path = self._current_path()
        fpath = self._frame_path(img_path)
        data = self._current_frame_data()
        fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if not silent:
            messagebox.showinfo("Saved", f"Frame saved:\n{fpath}")

    # ---------- crop ----------
    def _apply_one(self, img_path: str, data: Dict[str, Any]) -> List[Path]:
        """
        Returns a list of written output files.
        If split_enabled: outputs two files: *_L.tif and *_R.tif
        Else: outputs one file: *.tif
        """
        self._ensure_output_dirs()
        rgb, meta = read_tiff_rgb(img_path)

        if bool(data.get("cm_enabled", False)):
            try:
                rgb2, dst_bytes = apply_icc_transform(
                    rgb,
                    src_icc_bytes=meta.get("icc_profile"),
                    dst_icc_path=str(data.get("cm_to_icc", "")).strip(),
                    intent=str(data.get("cm_intent", "relative")),
                    bpc=bool(data.get("cm_bpc", True)),
                )
                rgb = rgb2
                meta["icc_profile"] = dst_bytes
            except Exception as e:
                print(f"[WARN] CM failed for {img_path}: {e}")

        bgr = rgb[:, :, ::-1].copy()

        side = str(data.get("page_side", "unknown"))
        mn = int(data.get("margin_nonspine", data.get("pad_nonspine", 0)))
        ms = int(data.get("margin_spine", data.get("pad_spine", 0)))
        top, right, bottom, left = derive_side_margins(side, mn, ms)

        md = float(data.get("manual_deskew_deg", 0.0))
        auto_on = bool(data.get("auto_deskew", False))
        maxdeg = float(data.get("auto_deskew_max_deg", 3.0))

        out_files: List[Path] = []
        stem = Path(img_path).stem

        def process_one(quad: np.ndarray, suffix: str) -> Path:
            if quad.shape != (4, 2):
                raise ValueError("Invalid frame quad_xy.")

            if data.get("frame_mode", "rect") == "rect":
                x0 = int(round(float(np.min(quad[:, 0]))))
                y0 = int(round(float(np.min(quad[:, 1]))))
                x1 = int(round(float(np.max(quad[:, 0]))))
                y1 = int(round(float(np.max(quad[:, 1]))))
                out_bgr = crop_axis_aligned_bgr(bgr, (x0, y0, x1, y1))
            else:
                out_bgr = warp_perspective_bgr(bgr, quad)

            out_bgr = add_padding(out_bgr, top, right, bottom, left)

            if abs(md) > 1e-6:
                out_bgr = rotate_image(out_bgr, -md)

            if auto_on:
                ang = estimate_text_angle_deg(out_bgr)
                if ang is not None and abs(ang) <= maxdeg:
                    out_bgr = rotate_image(out_bgr, -ang)

            out_rgb = out_bgr[:, :, ::-1].copy()
            out_file = self._out_dir() / f"{stem}{suffix}.tif"
            write_tiff_rgb_uncompressed(str(out_file), out_rgb, meta)
            return out_file

        if bool(data.get("split_enabled", False)):
            pages = data.get("pages", {})
            qL = np.array(pages.get("L", {}).get("quad_xy", []), dtype=np.float32)
            qR = np.array(pages.get("R", {}).get("quad_xy", []), dtype=np.float32)
            if qL.shape == (4, 2):
                out_files.append(process_one(qL, "_L"))
            if qR.shape == (4, 2):
                out_files.append(process_one(qR, "_R"))
        else:
            quad = np.array(data.get("quad_xy"), dtype=np.float32)
            out_files.append(process_one(quad, ""))

        return out_files

    def crop_current(self):
        if not self.files:
            return
        self.save_frame(silent=True)
        data = self._current_frame_data()
        try:
            out_files = self._apply_one(self._current_path(), data)
            messagebox.showinfo("Cropped", "Wrote:\n" + "\n".join(str(p) for p in out_files))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def batch_crop(self):
        if not self.files:
            return
        self._ensure_output_dirs()

        count = 0
        fail = 0
        for img_path in self.files:
            fpath = self._frame_path(img_path)
            if not fpath.exists():
                continue
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                outs = self._apply_one(img_path, data)
                count += len(outs)
            except Exception as e:
                print(f"[ERROR] Batch crop failed for {img_path}: {e}")
                fail += 1

        messagebox.showinfo(
            "Batch crop",
            f"Done.\nOutputs: {count}\nFailed: {fail}\nOutput dir: {self._out_dir()}"
        )


def main():
    root = tk.Tk()
    _ = ScanMowerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
