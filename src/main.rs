#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use eframe::egui::{self, ColorImage};
use egui::{Align, CursorIcon, Layout, RichText};
use image::{imageops, DynamicImage, ImageBuffer, Luma, Rgba, RgbaImage};

use imageproc::contrast::otsu_level;
use imageproc::filter::gaussian_blur_f32;
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use imageproc::map::map_colors;
use walkdir::WalkDir;

use serde::{Deserialize, Serialize};

use eframe::egui::IconData;

fn main() -> eframe::Result<()> {
    // Načtení PNG ikony přibalené v binárce
    let rgba = {
        let bytes = include_bytes!("../assets/scanmower_256.png");
        let img = image::load_from_memory(bytes).expect("icon png").to_rgba8();
        img.into_raw()
    };
    let icon = IconData { rgba, width: 256, height: 256 };

    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_icon(icon)
            .with_title("ScanMower"),
        ..Default::default()
    };

    // Pokud tvoje verze eframe vrací z closure Result, nech `Ok(...)`:
    eframe::run_native("ScanMower", native, Box::new(|_cc| {
        let _ = std::fs::create_dir_all(cache_root_dir());
        Ok(Box::new(App::default()))
    }))
    
    // Pokud máš starší API, kde closure vrací rovnou Box<dyn App>, použij toto:
    // eframe::run_native("ScanMower", native, Box::new(|_cc| Box::<App>::default()))
}

// Aplikovat na
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
enum ApplyScope {
    OnlyThis,
    All,
    FromHere,
    EveryOther,
    EveryOtherFromHere,
}
impl ApplyScope {
    fn label(self) -> &'static str {
        match self {
            ApplyScope::OnlyThis => "Aplikovat pouze na tento sken",
            ApplyScope::All => "Aplikovat na vše",
            ApplyScope::FromHere => "Aplikovat na vše od tohoto skenu",
            ApplyScope::EveryOther => "Aplikovat na každou druhou stranu",
            ApplyScope::EveryOtherFromHere => "Aplikovat na každou druhou stranu od tohoto skenu",
        }
    }
}

// ===== Výstupní formát =====
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
enum OutputFormat {
    Png,
    Jpeg,
    Tiff,
}
impl OutputFormat {
    fn as_ext(&self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
            OutputFormat::Tiff => "tif",
        }
    }
}

// ===== Rotace =====
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
enum Rotation {
    None,
    Left90,   // 90° CCW
    Right90,  // 90° CW
    Rot180,
}

// ===== Okraje =====
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
enum MarginSource {
    Artificial,   // barevný rámeček
    FromOriginal, // rozšíření výřezu z originálu
}
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
enum MarginUnits {
    Px,
    Mm,
}

// ===== Parametry (per-soubor) =====
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
struct Params {
    // Rotace / narovnání
    rotation: Rotation,
    deskew_max_deg: f32,
    manual_threshold: Option<u8>,
    auto_deskew: bool,
    manual_deskew_deg: f32,

    // Okraje
    margin_source: MarginSource,
    margin_units: MarginUnits,
    margin_amount: f32,    // px nebo mm
    margin_color: [u8; 4], // pro Artificial

    // Výstup
    out_format: OutputFormat,
    jpeg_quality: u8,
    dpi: Option<u32>,
    strip_metadata: bool,

    // Přesné zadání: přidáno přejmenování
    rename_enabled: bool,
    rename_mask: String, // např. "scan_###"
    rename_start: usize, // počáteční číslo

    // Page splitting (two-page scan)
    // If enabled, export will produce two pages (left + right).
    split_enabled: bool,
    // Vertical split position in normalized coordinates of the rotated preview/image: 0.0..1.0
    // Split X position at TOP in normalized coordinates of the rotated preview/image: 0.0..1.0
    #[serde(default)]
    split_x_top_norm: f32,
    // Split X position at BOTTOM in normalized coordinates of the rotated preview/image: 0.0..1.0
    #[serde(default)]
    split_x_bottom_norm: f32,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            rotation: Rotation::None,
            deskew_max_deg: 5.0,
            manual_threshold: None,
            auto_deskew: true,
            manual_deskew_deg: 0.0,
            margin_source: MarginSource::Artificial,
            margin_units: MarginUnits::Px,
            margin_amount: 0.0,
            margin_color: [255, 255, 255, 255],
            out_format: OutputFormat::Tiff,
            jpeg_quality: 92,
            dpi: Some(300),
            strip_metadata: false,
            rename_enabled: false,
            rename_mask: "scan_###".to_string(),
            rename_start: 1,

            split_enabled: false,
            split_x_top_norm: 0.5,
            split_x_bottom_norm: 0.5,
        }
    }
}

// ===== Stav per soubor =====
#[derive(Clone, Debug)]
struct FileState {
    params: Params,
    last_preview_img_size: Option<[u32; 2]>,
    crop_enabled: bool,
    crop_rect_img: Option<egui::Rect>, // v pixelech rotovaného náhledu
}
impl Default for FileState {
    fn default() -> Self {
        Self {
            params: Params::default(),
            last_preview_img_size: None,
            crop_enabled: false,
            crop_rect_img: None,
        }
    }
}

// ===== Cache klíč náhledu =====
#[derive(Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
struct CacheKey {
    path: PathBuf,
    file_len: u64,
    mtime_unix: i64,
    thr: Option<u8>,
    deskew_tenths: i16,
    auto: bool,
    manual_tenths: i16,
    rotation: Rotation,
}

fn file_stamp(path: &Path) -> (u64, i64) {
    let meta = std::fs::metadata(path).ok();
    let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (len, mtime)
}

fn cache_root_dir() -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|x| x.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    base.join("cache")
}

fn cache_key_id(key: &CacheKey) -> String {
    let bytes = serde_json::to_vec(key).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn cache_preview_dir() -> PathBuf {
    cache_root_dir().join("preview")
}

fn cache_thumb_dir() -> PathBuf {
    cache_root_dir().join("thumb")
}

fn cache_paths(key: &CacheKey) -> (PathBuf, PathBuf) {
    let root = cache_preview_dir();
    let id = cache_key_id(key);
    (root.join(format!("{id}.png")), root.join(format!("{id}.json")))
}

fn load_disk_cached_color_image(key: &CacheKey) -> Option<ColorImage> {
    let (png_path, json_path) = cache_paths(key);
    let j = std::fs::read(&json_path).ok()?;
    let stored: CacheKey = serde_json::from_slice(&j).ok()?;
    if &stored != key {
        return None;
    }
    let bytes = std::fs::read(png_path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let pixels = img.into_raw();
    Some(ColorImage::from_rgba_unmultiplied(size, &pixels))
}

fn save_disk_cached_rgba(key: &CacheKey, rgba: &image::RgbaImage) {
    let root = cache_preview_dir();
    let _ = std::fs::create_dir_all(&root);
    let (png_path, json_path) = cache_paths(key);

    // zmenši pro cache (rychlé načítání); zachovej poměr stran
    let max_dim: u32 = 1600;
    let (w, h) = (rgba.width(), rgba.height());
    let thumb = if w.max(h) > max_dim {
        let (nw, nh) = if w >= h {
            (max_dim, ((h as f32) * (max_dim as f32) / (w as f32)).round().max(1.0) as u32)
        } else {
            (((w as f32) * (max_dim as f32) / (h as f32)).round().max(1.0) as u32, max_dim)
        };
        image::imageops::resize(rgba, nw, nh, image::imageops::FilterType::Triangle)
    } else {
        rgba.clone()
    };

    let dynimg = image::DynamicImage::ImageRgba8(thumb);
    if dynimg
        .save_with_format(&png_path, image::ImageFormat::Png)
        .is_ok()
    {
        if let Ok(j) = serde_json::to_vec(key) {
            let _ = std::fs::write(json_path, j);
        }
    }
}


// ===== Disk cache: thumbnails (rychlé miniatury do seznamu) =====
#[derive(Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
struct ThumbKey {
    path: PathBuf,
    file_len: u64,
    mtime_unix: i64,
}

impl ThumbKey {
    fn from_path(path: &Path) -> Self {
        let (file_len, mtime_unix) = file_stamp(path);
        Self { path: path.to_path_buf(), file_len, mtime_unix }
    }
}

fn thumb_key_id(key: &ThumbKey) -> String {
    let bytes = serde_json::to_vec(key).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn thumb_paths(key: &ThumbKey) -> (PathBuf, PathBuf) {
    let root = cache_thumb_dir();
    let id = thumb_key_id(key);
    (root.join(format!("{id}.png")), root.join(format!("{id}.json")))
}

fn load_disk_cached_thumb(key: &ThumbKey) -> Option<ColorImage> {
    let (png_path, json_path) = thumb_paths(key);
    let j = std::fs::read(&json_path).ok()?;
    let stored: ThumbKey = serde_json::from_slice(&j).ok()?;
    if &stored != key {
        return None;
    }
    let bytes = std::fs::read(png_path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let pixels = img.into_raw();
    Some(ColorImage::from_rgba_unmultiplied(size, &pixels))
}

fn save_disk_cached_thumb(key: &ThumbKey, rgba: &image::RgbaImage) {
    let root = cache_thumb_dir();
    let _ = std::fs::create_dir_all(&root);
    let (png_path, json_path) = thumb_paths(key);

    // agresivně malé – jde jen o miniaturu
    let max_dim: u32 = 512;
    let (w, h) = (rgba.width(), rgba.height());
    let thumb = if w.max(h) > max_dim {
        let (nw, nh) = if w >= h {
            (max_dim, ((h as f32) * (max_dim as f32) / (w as f32)).round().max(1.0) as u32)
        } else {
            (((w as f32) * (max_dim as f32) / (h as f32)).round().max(1.0) as u32, max_dim)
        };
        image::imageops::resize(rgba, nw, nh, image::imageops::FilterType::Triangle)
    } else {
        rgba.clone()
    };

    let dynimg = image::DynamicImage::ImageRgba8(thumb);
    if dynimg.save_with_format(&png_path, image::ImageFormat::Png).is_ok() {
        if let Ok(j) = serde_json::to_vec(key) {
            let _ = std::fs::write(json_path, j);
        }
    }
}

impl CacheKey {
    fn from_params(path: &Path, p: &Params) -> Self {
        let (file_len, mtime_unix) = file_stamp(path);
        Self {
            path: path.to_path_buf(),
            file_len,
            mtime_unix,
            thr: p.manual_threshold,
            deskew_tenths: (p.deskew_max_deg * 10.0).round() as i16,
            auto: p.auto_deskew,
            manual_tenths: (p.manual_deskew_deg * 10.0).round() as i16,
            rotation: p.rotation,
        }
    }
}

// ===== Crop interakce =====
#[derive(Clone, Copy, Debug)]
enum CropHandle {
    Move,
    N,
    S,
    W,
    E,
    NW,
    NE,
    SW,
    SE,
}

// šířky zón v *screen* px (snazší trefení)
const EDGE_BAND: f32 = 14.0;
const CORNER_BAND: f32 = 16.0;


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SplitDragKind {
    Both,
    Top,
    Bottom,
}

struct App {
    files: Vec<PathBuf>,
    selected: Option<usize>,
    outdir: Option<PathBuf>,

    per_file: HashMap<PathBuf, FileState>,

    job: Option<JobHandle>,
    progress_text: String,

    cache: HashMap<CacheKey, egui::TextureHandle>,
    cache_order: VecDeque<CacheKey>,
    cache_capacity: usize,

    thumb_cache: HashMap<ThumbKey, egui::TextureHandle>,
    thumb_order: VecDeque<ThumbKey>,
    thumb_capacity: usize,

    // Disk cache prewarm (thumbnail cache generation)
    prewarm_total: usize,
    prewarm_done: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    prewarm_running: bool,

    zoom: f32,

    last_project_path: Option<PathBuf>,

    // „Aplikovat na…“ volby pro sekce
    scope_rotace: ApplyScope,
    scope_narovnani: ApplyScope,
    scope_okraje: ApplyScope,
    scope_orez: ApplyScope,
    scope_split: ApplyScope,
    scope_vystup: ApplyScope,
    scope_rename: ApplyScope,


    // globální dočasné stavy dragování
    drag_handle: Option<CropHandle>,
    drag_origin_rect: Option<egui::Rect>,
    drag_start_mouse_img: Option<egui::Pos2>,
    drag_keep_ratio: bool,
    drag_from_center: bool,
    drag_aspect: Option<f32>,

    // page splitting: dragging the divider
    split_dragging: bool,
    split_drag_kind: SplitDragKind,

    // do App
    right_panel_open: bool,
    left_panel_open: bool,
    show_thumbs: bool,
    thumb_size: f32,
    file_filter: String,

}
impl Default for App {
    fn default() -> Self {
        Self {
            files: vec![],
            selected: None,
            outdir: None,
            per_file: HashMap::new(),
            job: None,
            progress_text: String::new(),
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_capacity: 64,
            thumb_cache: HashMap::new(),
            thumb_order: VecDeque::new(),
            thumb_capacity: 256,
            prewarm_total: 0,
            prewarm_done: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            prewarm_running: false,
            zoom: 1.0,
            drag_handle: None,
            drag_origin_rect: None,
            drag_start_mouse_img: None,
            drag_keep_ratio: false,
            drag_from_center: false,
            drag_aspect: None,
            split_dragging: false,
            split_drag_kind: SplitDragKind::Both,
            scope_rotace: ApplyScope::All,
            scope_narovnani: ApplyScope::All,
            scope_okraje: ApplyScope::All,
            scope_orez: ApplyScope::All,
            scope_split: ApplyScope::All,
            scope_vystup: ApplyScope::All,
            scope_rename: ApplyScope::All,
            right_panel_open: true,
            show_thumbs: true,
            thumb_size: 72.0,
            file_filter: String::new(),
            left_panel_open: true,
            last_project_path: None,

        }
    }
}

struct JobHandle {
    _cancel: Arc<Mutex<bool>>,
    rx: mpsc::Receiver<JobEvent>,
    _join: thread::JoinHandle<()>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
enum JobEvent {
    Started(usize),
    Progress(usize, usize, PathBuf),
    Done(usize),
    Error(String),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct SavedRect {
    min: [f32; 2],
    max: [f32; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedFileState {
    params: Params,
    crop_enabled: bool,
    crop_rect_img: Option<SavedRect>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Project {
    files: Vec<PathBuf>,
    selected: Option<usize>,
    outdir: Option<PathBuf>,
    per_file: std::collections::HashMap<PathBuf, SavedFileState>,
}

impl App {
    fn state_mut_for(&mut self, path: &Path) -> &mut FileState {
        self.per_file.entry(path.to_path_buf()).or_default()
    }
    fn state_for(&self, path: &Path) -> Option<&FileState> {
        self.per_file.get(path)
    }

    fn add_files(&mut self, paths: Vec<PathBuf>) {
        let mut added: Vec<PathBuf> = Vec::new();
        for p in paths {
            if is_image(&p) {
                if !self.per_file.contains_key(&p) {
                    self.per_file.insert(p.clone(), FileState::default());
                    added.push(p.clone());
                }
                if !self.files.contains(&p) {
                    self.files.push(p);
                }
            }
        }
        self.files.sort();
        if self.selected.is_none() && !self.files.is_empty() {
            self.selected = Some(0);
        }

        // background prewarm disk cache thumbnails (jen pro nově přidané)
        if !added.is_empty() {
            self.start_prewarm(added);
        }
    }
    fn add_folder_recursively(&mut self, folder: &Path) {
        let mut list = vec![];
        for e in WalkDir::new(folder)
            .max_depth(1)                 // <- jen přímo v dané složce
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if e.file_type().is_file() && is_image(e.path()) {
                list.push(e.path().to_path_buf());
            }
        }
        self.add_files(list);
    }

fn start_prewarm(&mut self, paths: Vec<PathBuf>) {
    use rayon::prelude::*;

    // pokud už běží, necháme doběhnout (jednoduchost)
    if self.prewarm_running {
        return;
    }

    self.prewarm_total = paths.len();
    self.prewarm_done.store(0, Ordering::Relaxed);
    self.prewarm_running = true;

    let done = self.prewarm_done.clone();

    // IO-heavy → netlačit na disk: omezíme paralelismus (typicky 2–4 je optimum)
    let workers = 4usize.min(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)).max(1);

    std::thread::spawn(move || {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build();
        let run = || {
            paths.par_iter().for_each(|p| {
                let key = ThumbKey::from_path(p);
                if load_disk_cached_thumb(&key).is_none() {
                    if let Ok(img) = image::open(p) {
                        let rgba = img.to_rgba8();
                        save_disk_cached_thumb(&key, &rgba);
                    }
                }
                done.fetch_add(1, Ordering::Relaxed);
            });
        };
        if let Ok(pool) = pool {
            pool.install(run);
        } else {
            run();
        }
    });
}

    fn insert_files_at(&mut self, index: usize, paths: Vec<PathBuf>) {
        // Vyfiltruj jen obrázky, zamez duplicitám a udrž stabilní pořadí bloku
        let mut new_paths: Vec<PathBuf> = paths
            .into_iter()
            .filter(|p| is_image(p))
            .collect();
        new_paths.sort(); // volitelně řadit v rámci vkládaného bloku

        // Bez duplicit ve `self.files`
        new_paths.retain(|p| !self.files.contains(p));

        // Připrav per-file stav
        for p in &new_paths {
            self.per_file.entry(p.clone()).or_insert_with(FileState::default);
        }

        // Vlož jako souvislý blok
        let mut pos = index.min(self.files.len());
        for p in new_paths {
            self.files.insert(pos, p);
            pos += 1;
        }

        if self.selected.is_none() && !self.files.is_empty() {
            self.selected = Some(0);
        }
    }
    fn update_job_status(&mut self) {
        // prewarm dokončen? (thread nemá přístup k &mut self, takže to hlídáme tady)
        if self.prewarm_running && self.prewarm_total > 0 {
            let d = self.prewarm_done.load(Ordering::Relaxed);
            if d >= self.prewarm_total {
                self.prewarm_running = false;
            }
        }

        if let Some(job) = self.job.take() {
            let job = job;
            let mut keep = true;
            while let Ok(ev) = job.rx.try_recv() {
                match ev {
                    JobEvent::Started(total) => {
                        self.progress_text = format!("Zpracování spuštěno ({} souborů)…", total)
                    }
                    JobEvent::Progress(i, total, ref p) => {
                        self.progress_text = format!(
                            "{} / {}: {}",
                            i,
                            total,
                            p.file_name().and_then(|s| s.to_str()).unwrap_or("")
                        )
                    }
                    JobEvent::Done(done) => {
                        self.progress_text = format!("Hotovo: {} souborů.", done);
                        keep = false;
                    }
                    JobEvent::Error(e) => {
                        self.progress_text = format!("Chyba: {}", e);
                        keep = false;
                    }
                }
            }
            if keep {
                self.job = Some(job);
            }
        }
    }

    fn targets_from(&self, start_idx: usize, scope: ApplyScope) -> Vec<usize> {
        let n = self.files.len();
        match scope {
            ApplyScope::OnlyThis => vec![start_idx],
            ApplyScope::All => (0..n).collect(),
            ApplyScope::FromHere => (start_idx..n).collect(),
            ApplyScope::EveryOther => (0..n).filter(|i| i % 2 == 0).collect(),
            ApplyScope::EveryOtherFromHere => {
                let par = start_idx % 2;
                (start_idx..n).filter(|i| i % 2 == par).collect()
            }
        }
    }

    // --- převod mezi FileState <-> SavedFileState ---
    fn to_saved_file_state(st: &FileState) -> SavedFileState {
        SavedFileState {
            params: st.params.clone(),
            crop_enabled: st.crop_enabled,
            crop_rect_img: st.crop_rect_img.map(|r| SavedRect {
                min: [r.min.x, r.min.y],
                max: [r.max.x, r.max.y],
            }),
        }
    }
    fn from_saved_file_state(s: &SavedFileState) -> FileState {
        FileState {
            params: s.params.clone(),
            last_preview_img_size: None, // dopočítá se z náhledu
            crop_enabled: s.crop_enabled,
            crop_rect_img: s.crop_rect_img.map(|rr| {
                egui::Rect::from_min_max(
                    egui::pos2(rr.min[0], rr.min[1]),
                    egui::pos2(rr.max[0], rr.max[1]),
                )
            }),
        }
    }

    // --- složení a aplikace projektu ---
    fn build_project(&self) -> Project {
        let mut per: HashMap<PathBuf, SavedFileState> = HashMap::new();
        for (p, st) in &self.per_file {
            per.insert(p.clone(), Self::to_saved_file_state(st));
        }
        Project {
            files: self.files.clone(),
            selected: self.selected,
            outdir: self.outdir.clone(),
            per_file: per,
        }
    }
    fn apply_project(&mut self, prj: Project) {
        self.files = prj.files;
        self.selected = prj.selected;
        self.outdir = prj.outdir;

        self.per_file.clear();
        for (p, ss) in prj.per_file {
            self.per_file.insert(p, Self::from_saved_file_state(&ss));
        }

        // po načtení vyprázdni cache náhledů, ať se vygenerují s novými parametry
        self.cache.clear();
        self.cache_order.clear();

        // pokud není vybráno nic, ale máme soubory, vyber první
        if self.selected.is_none() && !self.files.is_empty() {
            self.selected = Some(0);
        }
    }

    // --- uložení / načtení na konkrétní cestu ---
    fn save_project_to_path(&mut self, path: &Path) -> Result<()> {
        let prj = self.build_project();
        let f = std::fs::File::create(path)
            .with_context(|| format!("Nelze vytvořit projekt: {}", path.display()))?;
        serde_json::to_writer_pretty(f, &prj)?;
        self.last_project_path = Some(path.to_path_buf());
        Ok(())
    }
    fn open_project_from_path(&mut self, path: &Path) -> Result<()> {
        let f = std::fs::File::open(path)
            .with_context(|| format!("Nelze otevřít projekt: {}", path.display()))?;
        let prj: Project = serde_json::from_reader(f)
            .with_context(|| "Poškozený nebo nekompatibilní soubor projektu")?;
        self.apply_project(prj);
        self.last_project_path = Some(path.to_path_buf());
        Ok(())
    }

    // --- dialogové obálky + rychlé uložení ---
    fn save_project_to_last_or_as(&mut self) {
        if let Some(p) = self.last_project_path.clone() {
            match self.save_project_to_path(&p) {
                Ok(()) => self.progress_text = format!("Projekt uložen: {}", p.display()),
                Err(e) =>  self.progress_text = format!("Chyba ukládání: {e}"),
            }
        } else {
            self.save_project_as_dialog();
        }
    }
    fn save_project_as_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("ScanMower projekt", &["smprj", "json"])
            .set_file_name("projekt.smprj")
            .save_file()
        {
            match self.save_project_to_path(&path) {
                Ok(()) => self.progress_text = format!("Projekt uložen: {}", path.display()),
                Err(e)  => self.progress_text = format!("Chyba ukládání: {e}"),
            }
        }
    }
    fn open_project_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("ScanMower projekt", &["smprj", "json"])
            .pick_file()
        {
            match self.open_project_from_path(&path) {
                Ok(()) => self.progress_text = format!("Projekt otevřen: {}", path.display()),
                Err(e)  => self.progress_text = format!("Chyba načítání: {e}"),
            }
        }
    }

    fn apply_rotation(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src_rot = self
            .state_for(&src_p)
            .map(|s| s.params.rotation)
            .unwrap_or(Rotation::None);

        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.rotation = src_rot;
        }
    }

    fn apply_narovnani(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src = self.state_for(&src_p).cloned().unwrap_or_default().params;
        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.rotation = src.rotation;
            st.params.auto_deskew = src.auto_deskew;
            st.params.deskew_max_deg = src.deskew_max_deg;
            st.params.manual_deskew_deg = src.manual_deskew_deg;
            st.params.manual_threshold = src.manual_threshold;
        }
    }

    fn apply_okraje(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src = self.state_for(&src_p).cloned().unwrap_or_default().params;
        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.margin_source = src.margin_source;
            st.params.margin_units = src.margin_units;
            st.params.margin_amount = src.margin_amount;
            st.params.margin_color = src.margin_color;
        }
    }

    fn apply_split(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src = self.state_for(&src_p).cloned().unwrap_or_default().params;

        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.split_enabled = src.split_enabled;
            st.params.split_x_top_norm = src.split_x_top_norm;
            st.params.split_x_bottom_norm = src.split_x_bottom_norm;
        }
    }

    fn scale_rect_between(
        src_rect: egui::Rect,
        src_wh: [u32; 2],
        dst_wh: [u32; 2],
    ) -> egui::Rect {
        let (sw, sh) = (src_wh[0] as f32, src_wh[1] as f32);
        let (dw, dh) = (dst_wh[0] as f32, dst_wh[1] as f32);
        if sw <= 0.0 || sh <= 0.0 || dw <= 0.0 || dh <= 0.0 {
            return egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(dw.max(1.0), dh.max(1.0)));
        }
        let nx = src_rect.min.x / sw;
        let ny = src_rect.min.y / sh;
        let nw = src_rect.width() / sw;
        let nh = src_rect.height() / sh;
        let min = egui::pos2(nx * dw, ny * dh);
        let size = egui::vec2((nw * dw).max(1.0), (nh * dh).max(1.0));
        egui::Rect::from_min_size(min, size)
    }

    fn apply_orez(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src_st = self.state_for(&src_p).cloned().unwrap_or_default();
        let src_rect = src_st.crop_rect_img;
        let src_wh = src_st.last_preview_img_size;
        let src_enabled = src_st.crop_enabled;

        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.crop_enabled = src_enabled;
            if let (Some(r), Some(sw), Some(dw)) = (src_rect, src_wh, st.last_preview_img_size) {
                let rr = Self::scale_rect_between(r, sw, dw);
                // clamp do obrázku
                let w = dw[0] as f32;
                let h = dw[1] as f32;
                let min = egui::pos2(rr.min.x.clamp(0.0, w), rr.min.y.clamp(0.0, h));
                let max = egui::pos2(rr.max.x.clamp(0.0, w), rr.max.y.clamp(0.0, h));
                st.crop_rect_img = Some(egui::Rect::from_min_max(min, max));
            } else {
                // velikost náhledu neznáme → rám nastavíme až při zobrazení (auto)
                st.crop_rect_img = None;
            }
        }
    }

    fn apply_vystup(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src = self.state_for(&src_p).cloned().unwrap_or_default().params;
        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.out_format = src.out_format;
            st.params.jpeg_quality = src.jpeg_quality;
            st.params.dpi = src.dpi;
            st.params.strip_metadata = src.strip_metadata;
        }
    }

    fn apply_rename(&mut self, from_idx: usize, scope: ApplyScope) {
        let src_p = self.files[from_idx].clone();
        let src = self.state_for(&src_p).cloned().unwrap_or_default().params;
        let targets = self.targets_from(from_idx, scope);
        for i in targets {
            let p = self.files[i].clone();
            let st = self.state_mut_for(&p);
            st.params.rename_enabled = src.rename_enabled;
            st.params.rename_mask = src.rename_mask.clone();
            st.params.rename_start = src.rename_start;
        }
    }


/// Vrátí miniaturu pro seznam souborů (rychlá, z disk cache `cache/thumb`).
fn thumb_texture(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
    let key = ThumbKey::from_path(path);

    if let Some(tex) = self.thumb_cache.get(&key).cloned() {
        return Some(tex);
    }

    // 1) disk cache
    let color = if let Some(c) = load_disk_cached_thumb(&key) {
        c
    } else {
        // 2) rychlý decode + resize a uložit do disk cache
        let img = image::open(path).ok()?;
        let rgba = img.to_rgba8();
        save_disk_cached_thumb(&key, &rgba);

        // znovu načti z RAM (nepovinné), tady rovnou vytvoříme ColorImage z "rgba" (už je malé / nebo se zmenší uvnitř save)
        // pro UI použijeme rovnou zmenšený buffer stejně jako v save
        let max_dim: u32 = 512;
        let (w, h) = (rgba.width(), rgba.height());
        let thumb = if w.max(h) > max_dim {
            let (nw, nh) = if w >= h {
                (max_dim, ((h as f32) * (max_dim as f32) / (w as f32)).round().max(1.0) as u32)
            } else {
                (((w as f32) * (max_dim as f32) / (h as f32)).round().max(1.0) as u32, max_dim)
            };
            image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Triangle)
        } else {
            rgba
        };

        ColorImage::from_rgba_unmultiplied(
            [thumb.width() as usize, thumb.height() as usize],
            thumb.as_raw(),
        )
    };

    let tag = format!("thumb-{}", self.thumb_order.len());
    let tex = ctx.load_texture(tag, color, Default::default());

    self.thumb_cache.insert(key.clone(), tex.clone());
    self.thumb_order.push_back(key);
    if self.thumb_order.len() > self.thumb_capacity {
        if let Some(old) = self.thumb_order.pop_front() {
            self.thumb_cache.remove(&old);
        }
    }
    Some(tex)
}

    /// Vytvoří / vrátí náhled (rotovaný, neořezaný). Uloží rozměr do per-file stavu.
    fn preview_texture(&mut self, ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
        let params_snapshot = self
            .state_for(path)
            .map(|s| s.params.clone())
            .unwrap_or_default();
        let key = CacheKey::from_params(path, &params_snapshot);

        if let Some(tex) = self.cache.get(&key).cloned() {
            let sz = tex.size_vec2();
            self.state_mut_for(path).last_preview_img_size = Some([sz.x as u32, sz.y as u32]);
            return Some(tex);
        }
        // Disk cache (rychlý start po opětovném spuštění)
        if let Some(color) = load_disk_cached_color_image(&key) {
            let tag = format!("preview-disk-{}", self.cache_order.len());
            let tex = ctx.load_texture(tag, color, Default::default());
            let sz = tex.size_vec2();
            self.state_mut_for(path).last_preview_img_size = Some([sz.x as u32, sz.y as u32]);
            self.cache.insert(key.clone(), tex.clone());
            self.cache_order.push_back(key);
            if self.cache_order.len() > self.cache_capacity {
                if let Some(old) = self.cache_order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            return Some(tex);
        }



        let img = image::open(path).ok()?;
        let (rot_rgba, _thr) = make_rotated_rgba(&img, &params_snapshot);
        // uložit do disk cache (PNG thumbnail)
        save_disk_cached_rgba(&key, &rot_rgba);
        let color = ColorImage::from_rgba_unmultiplied(
            [rot_rgba.width() as usize, rot_rgba.height() as usize],
            rot_rgba.as_raw(),
        );

        let tag = format!("preview-{}", self.cache_order.len());
        let tex = ctx.load_texture(tag, color, Default::default());

        self.state_mut_for(path).last_preview_img_size = Some([rot_rgba.width(), rot_rgba.height()]);
        self.cache.insert(key.clone(), tex.clone());
        self.cache_order.push_back(key);
        if self.cache_order.len() > self.cache_capacity {
            if let Some(old) = self.cache_order.pop_front() {
                self.cache.remove(&old);
            }
        }
        Some(tex)
    }

    /// Pokud je ruční ořez zapnut a zatím není rám, pokus se jej dopočítat.
    fn ensure_auto_crop_rect_if_needed(&mut self, path: &Path) {
        let (need, last_size, params) = {
            let st = self.state_mut_for(path);
            (st.crop_enabled && st.crop_rect_img.is_none(), st.last_preview_img_size, st.params.clone())
        };
        if !need {
            return;
        }
        let auto_rect = auto_crop_rect_for_with_params(path, &params).or_else(|| {
            last_size.map(|[w, h]| egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(w as f32, h as f32)))
        });
        if let Some(r) = auto_rect {
            self.state_mut_for(path).crop_rect_img = Some(r);
        }
    }

    fn start_job(&mut self, all: bool) {
        if self.job.is_some() {
            return;
        }
        let outdir = match &self.outdir {
            Some(p) => p.clone(),
            None => {
                self.progress_text = "Nejprve zvol výstupní složku.".into();
                return;
            }
        };

        let files: Vec<PathBuf> = if all {
            self.files.clone()
        } else {
            self.selected
                .and_then(|i| self.files.get(i).cloned())
                .into_iter()
                .collect()
        };
        if files.is_empty() {
            self.progress_text = "Žádné soubory ke zpracování.".into();
            return;
        }
        if let Err(e) = fs::create_dir_all(&outdir) {
            self.progress_text = format!("Nelze vytvořit výstupní složku: {}", e);
            return;
        }

        #[derive(Clone)]
        struct Task {
            path: PathBuf,
            params: Params,
            crop_norm: Option<(f32, f32, f32, f32)>,
            out_base: Option<String>,

            // Page splitting
            split_side: Option<SplitSide>,
            split_x_top_norm: f32,
            split_x_bottom_norm: f32,
        }

        #[derive(Clone, Copy, Debug)]
        enum SplitSide {
            Left,
            Right,
        }
        let mut tasks: Vec<Task> = Vec::with_capacity(files.len());
        for (idx, p) in files.iter().enumerate() {
            let st = self.state_for(p).cloned().unwrap_or_default();
            let crop_norm = if st.crop_enabled {
                if let (Some(r), Some([w, h])) = (st.crop_rect_img, st.last_preview_img_size) {
                    if w > 0 && h > 0 {
                        Some((
                            r.min.x / w as f32,
                            r.min.y / h as f32,
                            r.width() / w as f32,
                            r.height() / h as f32,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let out_base = if st.params.rename_enabled {
                let index = st.params.rename_start.saturating_add(idx);
                Some(apply_rename_mask(&st.params.rename_mask, index))
            } else {
                None
            };

            // If splitting is enabled, create two output tasks (left/right).
            if st.params.split_enabled {
                let split_top = st.params.split_x_top_norm.clamp(0.05, 0.95);
                let split_bottom = st.params.split_x_bottom_norm.clamp(0.05, 0.95);
                let split = ((split_top + split_bottom) * 0.5).clamp(0.05, 0.95);

                // Remap optional crop rect from full image to each half.
                let remap_crop = |c: Option<(f32, f32, f32, f32)>, x0: f32, x1: f32| -> Option<(f32, f32, f32, f32)> {
                    let (nx, ny, nw, nh) = c?;
                    let a0 = nx;
                    let a1 = (nx + nw).min(1.0);
                    let ix0 = a0.max(x0);
                    let ix1 = a1.min(x1);
                    if ix1 <= ix0 {
                        return None;
                    }
                    let new_x = (ix0 - x0) / (x1 - x0);
                    let new_w = (ix1 - ix0) / (x1 - x0);
                    Some((new_x, ny, new_w, nh))
                };

                // base names: either rename mask or original stem.
                let base_l = out_base.as_ref().map(|b| format!("{b}_L"));
                let base_r = out_base.as_ref().map(|b| format!("{b}_R"));

                tasks.push(Task {
                    path: p.clone(),
                    params: st.params.clone(),
                    crop_norm: remap_crop(crop_norm, 0.0, split),
                    out_base: base_l,
                    split_side: Some(SplitSide::Left),
                    split_x_top_norm: split_top,
                    split_x_bottom_norm: split_bottom,
                });
                tasks.push(Task {
                    path: p.clone(),
                    params: st.params,
                    crop_norm: remap_crop(crop_norm, split, 1.0),
                    out_base: base_r,
                    split_side: Some(SplitSide::Right),
                    split_x_top_norm: split_top,
                    split_x_bottom_norm: split_bottom,
                });
            } else {
                tasks.push(Task {
                    path: p.clone(),
                    params: st.params,
                    crop_norm,
                    out_base,
                    split_side: None,
                    split_x_top_norm: 0.5,
            split_x_bottom_norm: 0.5,
                });
            }
        }

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(Mutex::new(false));
        let cancel_clone = cancel.clone();

        let join = thread::spawn(move || {
            let total = tasks.len();
            let _ = tx.send(JobEvent::Started(total));
            let mut done = 0usize;


use std::collections::BTreeMap;

// Seskup úlohy podle zdrojového souboru, abychom při splitu nedekódovali 2×
let mut groups: BTreeMap<PathBuf, Vec<Task>> = BTreeMap::new();
for t in tasks {
    groups.entry(t.path.clone()).or_default().push(t);
}

let mut i = 0usize;
for (_path, group) in groups {
    if *cancel_clone.lock().unwrap() {
        let _ = tx.send(JobEvent::Error("Zrušeno".into()));
        return;
    }

    // načti zdroj jednou
    let src_path = group[0].path.clone();
    let src_bytes = std::fs::read(&src_path).ok();
    let img = match image::open(&src_path) {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(JobEvent::Error(format!("nelze otevřít {}: {e}", src_path.display())));
            return;
        }
    };

    // Pokud se parametry v rámci stejného souboru liší, fallback na původní (bez sdílení decode)
    let same_params = group.iter().all(|t| t.params == group[0].params);

    // Sdílený rotovaný obraz (největší úspora pro split L/R)
    let shared_rot = if same_params {
        let (rot, _thr) = make_rotated_rgba(&img, &group[0].params);
        Some(rot)
    } else {
        None
    };

    for t in group.into_iter() {
        i += 1;
        if *cancel_clone.lock().unwrap() {
            let _ = tx.send(JobEvent::Error("Zrušeno".into()));
            return;
        }

        let _ = tx.send(JobEvent::Progress(i, total, t.path.clone()));

        let res = (|| -> Result<()> {
            // Rotovaný RGBA (neořezaný)
            let mut rot_rgba = if let Some(ref rot) = shared_rot {
                rot.clone()
            } else {
                let (rot, _thr) = make_rotated_rgba(&img, &t.params);
                rot
            };

            // Optional page split (two-page scan): crop to left/right half first.
            if let Some(side) = t.split_side {
                let rw = rot_rgba.width();
                let rh = rot_rgba.height();

                let x_top = (t.split_x_top_norm.clamp(0.05, 0.95) * (rw as f32)).round();
                let x_bottom = (t.split_x_bottom_norm.clamp(0.05, 0.95) * (rw as f32)).round();
                let denom = (rh.saturating_sub(1)).max(1) as f32;

                // Determine maximal needed output width for each side (slanted axis).
                let mut max_left: u32 = 1;
                let mut max_right: u32 = 1;
                for y in 0..rh {
                    let ty = (y as f32) / denom;
                    let mut x = (x_top + (x_bottom - x_top) * ty).round() as i32;
                    x = x.clamp(1, (rw as i32) - 1);
                    let xu = x as u32;
                    max_left = max_left.max(xu);
                    max_right = max_right.max(rw - xu);
                }

                rot_rgba = match side {
                    SplitSide::Left => {
                        let mut out = RgbaImage::from_pixel(max_left, rh, Rgba([255, 255, 255, 255]));
                        for y in 0..rh {
                            let ty = (y as f32) / denom;
                            let mut x = (x_top + (x_bottom - x_top) * ty).round() as i32;
                            x = x.clamp(1, (rw as i32) - 1);
                            let xu = x as u32;

                            for xx in 0..xu {
                                let px = *rot_rgba.get_pixel(xx, y);
                                out.put_pixel(xx, y, px);
                            }
                        }
                        out
                    }
                    SplitSide::Right => {
                        let mut out = RgbaImage::from_pixel(max_right, rh, Rgba([255, 255, 255, 255]));
                        for y in 0..rh {
                            let ty = (y as f32) / denom;
                            let mut x = (x_top + (x_bottom - x_top) * ty).round() as i32;
                            x = x.clamp(1, (rw as i32) - 1);
                            let xu = x as u32;

                            let w = rw - xu;
                            for xx in 0..w {
                                let px = *rot_rgba.get_pixel(xu + xx, y);
                                out.put_pixel(xx, y, px);
                            }
                        }
                        out
                    }
                };
            }

            // Ruční/auto ořez
            let crop_xywh = if let Some((nx, ny, nw, nh)) = t.crop_norm {
                let rw = rot_rgba.width();
                let rh = rot_rgba.height();
                let mut x = (nx * rw as f32).round().clamp(0.0, rw.saturating_sub(1) as f32) as u32;
                let mut y = (ny * rh as f32).round().clamp(0.0, rh.saturating_sub(1) as f32) as u32;
                let mut w = (nw * rw as f32).round() as u32;
                let mut h = (nh * rh as f32).round() as u32;
                if w == 0 { w = 1; }
                if h == 0 { h = 1; }
                x = x.min(rw.saturating_sub(1));
                y = y.min(rh.saturating_sub(1));
                w = w.min(rw.saturating_sub(x));
                h = h.min(rh.saturating_sub(y));
                (x, y, w, h)
            } else {
                // Auto detekce hran stránky: preferuj „světlý papír na tmavém pozadí“
                let rotated_gray = DynamicImage::ImageRgba8(rot_rgba.clone()).to_luma8();
                if let Some((x, y, w, h)) = detect_page_bbox_smart(&rotated_gray) {
                    (x, y, w, h)
                } else {
                    // fallback na obsah (tmavé pixely)
                    let thr = otsu_level(&rotated_gray);
                    let mask = threshold_to_mask(&rotated_gray, thr);
                    if let Some((x, y, w, h)) = content_bbox(&mask) {
                        (x, y, w, h)
                    } else {
                        (0, 0, rot_rgba.width(), rot_rgba.height())
                    }
                }
            };

            let (x, y, w, h) = crop_xywh;
            let rot_rgba_uncropped = rot_rgba.clone(); // jen pro margin from original, je to už po splitu (pokud je)
            rot_rgba = image::imageops::crop_imm(&rot_rgba, x, y, w, h).to_image();

            // Okraj v px
            let px_from_mm = |mm: f32, dpi: u32| -> u32 {
                let px = (mm / 25.4) * dpi as f32;
                px.round().max(0.0) as u32
            };
            let margin_px: u32 = match t.params.margin_units {
                MarginUnits::Px => t.params.margin_amount.max(0.0).round() as u32,
                MarginUnits::Mm => {
                    let dpi = t.params.dpi.unwrap_or(300);
                    px_from_mm(t.params.margin_amount.max(0.0), dpi)
                }
            };

            // Aplikace okraje
            let out = if margin_px > 0 {
                match t.params.margin_source {
                    MarginSource::Artificial => {
                        add_margin_rgba(&rot_rgba, margin_px, Rgba(t.params.margin_color))
                    }
                    MarginSource::FromOriginal => {
                        add_margin_rgba_from_original(
                            &rot_rgba_uncropped,
                            (x, y, w, h),
                            margin_px,
                        )
                    }
                }
            } else {
                rot_rgba
            };

            // Ulož
            let mut base = if let Some(ref b) = t.out_base {
                b.clone()
            } else {
                t.path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("out")
                    .to_string()
            };
            // If this is a split subpage and caller didn't provide a custom base, add a suffix.
            if t.out_base.is_none() {
                if let Some(side) = t.split_side {
                    base = match side {
                        SplitSide::Left => format!("{base}_L"),
                        SplitSide::Right => format!("{base}_R"),
                    };
                }
            }
            let out_path = unique_out_path(&outdir, &base, t.params.out_format.as_ext());
            save_image_with_metadata(
                &DynamicImage::ImageRgba8(out),
                &out_path,
                &t.params,
				&t.path,
                src_bytes.as_deref(),
            )?;

            Ok(())
        })();

        match res {
            Ok(()) => done += 1,
            Err(e) => {
                let _ = tx.send(JobEvent::Error(format!(
                    "{}: {}",
                    t.path.display(),
                    e
                )));
            }
        }
    }
}
            let _ = tx.send(JobEvent::Done(done));
        });

        self.job = Some(JobHandle { _cancel: cancel, rx, _join: join });
        self.progress_text.clear();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_job_status();

		// Klávesové zkratky pro práci s projektem
        ctx.input(|i| {
            let ctrl_like = i.modifiers.command || i.modifiers.ctrl;
            if ctrl_like && i.key_pressed(egui::Key::S) {
                if i.modifiers.shift {
                    self.save_project_as_dialog();
                } else {
                    self.save_project_to_last_or_as();
                }
            }
            if ctrl_like && i.key_pressed(egui::Key::O) {
                self.open_project_dialog();
            }
        });

        // Horní lišta
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Přidat soubory…").clicked() {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
                        .pick_files()
                    {
                        self.add_files(files);
                    }
                }
                if ui.button("Přidat složku…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.add_folder_recursively(&dir);
                    }
                }
                if ui.button("Vyčistit seznam").clicked() {
                    self.files.clear();
                    self.selected = None;
                }

                // stav disk cache prewarm
                if self.prewarm_running {
                    use std::sync::atomic::Ordering;
                    let done = self.prewarm_done.load(Ordering::Relaxed);
                    ui.label(format!("Cache: {}/{}", done, self.prewarm_total));
                    if done >= self.prewarm_total {
                        self.prewarm_running = false;
                    }
                }

                ui.separator();
                // světle modré tlačítko „Výstupní složka…“
                let light_blue = egui::Color32::from_rgb(180, 210, 255);
                let blue_stroke = egui::Color32::from_rgb(90, 130, 200);
                if ui
                    .add(
                        egui::Button::new(
                            egui::RichText::new("Výstupní složka…").color(egui::Color32::BLACK),
                        )
                        .fill(light_blue)
                        .stroke(egui::Stroke::new(1.0, blue_stroke)),
                    )
                    .clicked()
                {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.outdir = Some(dir);
                    }
                }
                if let Some(out) = &self.outdir {
                    ui.label(format!("Výstup: {}", out.display()));
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Klikací otazník s větším textem
                    ui.menu_button("❓", |ui| {
                        ui.set_min_width(280.0);
                        let big = 18.0;
                        let mid = 15.0;

                        ui.label(
                            egui::RichText::new(format!("ScanMower v{}", env!("CARGO_PKG_VERSION")))
                                .strong()
                                .size(big),
                        );

                        let desc = env!("CARGO_PKG_DESCRIPTION");
                        if !desc.is_empty() {
                            ui.label(egui::RichText::new(desc).size(mid));
                        }

                        ui.separator();

                        let repo = env!("CARGO_PKG_REPOSITORY");
                        if !repo.is_empty() {
                            ui.hyperlink_to(
                                egui::RichText::new(repo).size(mid),
                                repo,
                            );
                        }

                        let home = env!("CARGO_PKG_HOMEPAGE");
                        if !home.is_empty() {
                            ui.hyperlink_to(
                                egui::RichText::new(home).size(mid),
                                home,
                            );
                        }

                        ui.separator();

                        let authors = env!("CARGO_PKG_AUTHORS");
                        if !authors.is_empty() {
                            ui.label(egui::RichText::new(format!("👤 Autor: {authors}")).size(mid));
                        }

                        let license = env!("CARGO_PKG_LICENSE");
                        if !license.is_empty() {
                            ui.label(egui::RichText::new(format!("⚖ Licence: {license}")).size(mid));
                        }
                    });

                    ui.label(RichText::new("ScanMower").strong());
                });

            });
        });


        // ===== Levý panel – lze skrýt =====
        let mut post_reset_drag = false;
        if self.left_panel_open {
            egui::SidePanel::left("left")
                .min_width(300.0)
                .show(ctx, |ui| {
                    // horní lišta panelu + tlačítko schovat
                    ui.horizontal(|ui| {
                        ui.heading("Ovládání");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Skrýt ◀").clicked() {
                                self.left_panel_open = false;
                            }
                        });
                    });

                    ui.separator();

                    // Zoom
                    ui.heading("Zoom");
                    ui.add(egui::Slider::new(&mut self.zoom, 0.1..=4.0).text("Měřítko"));
                    ui.horizontal(|ui| {
                        if ui.button("−").clicked() { self.zoom = (self.zoom * 0.9).max(0.1); }
                        if ui.button("100%").clicked() { self.zoom = 1.0; }
                        if ui.button("+").clicked() { self.zoom = (self.zoom * 1.1).min(4.0); }
                    });

                    ui.separator();

                    let selected_path: Option<PathBuf> =
                        self.selected.and_then(|i| self.files.get(i).cloned());
                    let controls_enabled = selected_path.is_some();

                    // 1) LOKÁLY MIMO CLOSURE
                    let selected_idx_copy = self.selected;

                    let mut scope_rotace_local    = self.scope_rotace;
                    let mut scope_narovnani_local = self.scope_narovnani;
                    let mut scope_okraje_local    = self.scope_okraje;
                    let mut scope_orez_local      = self.scope_orez;
                    let mut scope_split_local     = self.scope_split;
                    let mut scope_vystup_local    = self.scope_vystup;
                    let mut scope_rename_local    = self.scope_rename;

                    let mut do_apply_rotace    = false;
                    let mut do_apply_narovnani = false;
                    let mut do_apply_okraje    = false;
                    let mut do_apply_orez      = false;
                    let mut do_apply_split     = false;
                    let mut do_apply_vystup    = false;
                    let mut do_apply_rename    = false;

                    // auto-crop po přepnutí checkboxu uděláme až po closure
                    let mut need_auto_crop_rect = false;

                    ui.add_enabled_ui(controls_enabled, |ui| {
                        if let Some(p) = selected_path.as_ref() {

                            // ========== Rozdělení skenu (dvoustrana) ==========
                            // Umístěno mezi Zoom a Rotace (podle UX požadavku).
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Rozdělení (L/P)");

                                let resp = ui.checkbox(&mut st.params.split_enabled, "Rozdělit sken na levou/pravou stránku");
                                if resp.changed() && st.params.split_enabled {
                                    // Auto-detect divider when enabling.
                                    if let Some(x) = detect_split_x_norm(p, &st.params) {
                                        st.params.split_x_top_norm = x;
                                        st.params.split_x_bottom_norm = x;
                                    }
                                }

                                if st.params.split_enabled {
                                    ui.add(
                                        egui::Slider::new(&mut st.params.split_x_top_norm, 0.05..=0.95)
                                            .text("Osa nahoře (X)")
                                            .clamping(egui::SliderClamping::Always),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut st.params.split_x_bottom_norm, 0.05..=0.95)
                                            .text("Osa dole (X)")
                                            .clamping(egui::SliderClamping::Always),
                                    );
                                    ui.horizontal(|ui| {
                                        if ui.button("Detekovat znovu").clicked() {
                                            if let Some(x) = detect_split_x_norm(p, &st.params) {
                                                st.params.split_x_top_norm = x;
                                        st.params.split_x_bottom_norm = x;
                                            }
                                        }
                                        ui.label("Tip: Střed můžeš také přetáhnout v náhledu.");
                                    });
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_split")
                                    .selected_text(scope_split_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_split_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_split_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_split_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_split_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_split_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_split = true; }
                            });

                            ui.separator();

                            // ========== Rotace ==========
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Rotace");
                                let active_fill = egui::Color32::from_gray(60);
                                let active_stroke = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(60));
                                ui.horizontal_wrapped(|ui| {
                                    let mut b0 = egui::Button::new(if st.params.rotation == Rotation::None {
                                        egui::RichText::new("0°").color(egui::Color32::WHITE)
                                    } else { egui::RichText::new("0°") }).min_size(egui::vec2(68.0, 28.0));
                                    if st.params.rotation == Rotation::None { b0 = b0.fill(active_fill).stroke(active_stroke); }
                                    if ui.add(b0).clicked() { st.params.rotation = Rotation::None; }

                                    let mut bl = egui::Button::new(if st.params.rotation == Rotation::Left90 {
                                        egui::RichText::new("90° vlevo").color(egui::Color32::WHITE)
                                    } else { egui::RichText::new("90° vlevo") }).min_size(egui::vec2(96.0, 28.0));
                                    if st.params.rotation == Rotation::Left90 { bl = bl.fill(active_fill).stroke(active_stroke); }
                                    if ui.add(bl).clicked() { st.params.rotation = Rotation::Left90; }

                                    let mut br = egui::Button::new(if st.params.rotation == Rotation::Right90 {
                                        egui::RichText::new("90° vpravo").color(egui::Color32::WHITE)
                                    } else { egui::RichText::new("90° vpravo") }).min_size(egui::vec2(102.0, 28.0));
                                    if st.params.rotation == Rotation::Right90 { br = br.fill(active_fill).stroke(active_stroke); }
                                    if ui.add(br).clicked() { st.params.rotation = Rotation::Right90; }

                                    let mut b180 = egui::Button::new(if st.params.rotation == Rotation::Rot180 {
                                        egui::RichText::new("180°").color(egui::Color32::WHITE)
                                    } else { egui::RichText::new("180°") }).min_size(egui::vec2(68.0, 28.0));
                                    if st.params.rotation == Rotation::Rot180 { b180 = b180.fill(active_fill).stroke(active_stroke); }
                                    if ui.add(b180).clicked() { st.params.rotation = Rotation::Rot180; }
                                });
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_rotace")
                                    .selected_text(scope_rotace_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_rotace_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_rotace_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_rotace_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_rotace_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_rotace_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() {
                                    do_apply_rotace = true; // ⬅️ jen rotace
                                }
                            });

                            ui.separator();

                            // ========== Narovnání ==========
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Narovnání");
                                ui.checkbox(&mut st.params.auto_deskew, "Automatické narovnání");

                                if st.params.auto_deskew {
                                    ui.horizontal(|ui| {
                                        ui.label("Auto deskew rozsah (±°)");
                                        ui.add(
                                            egui::DragValue::new(&mut st.params.deskew_max_deg)
                                                .range(0.0..=10.0)
                                                .clamp_existing_to_range(true)
                                                .speed(0.1)
                                        );
                                        if ui.button("−1°").clicked()  { st.params.deskew_max_deg = (st.params.deskew_max_deg - 1.0).max(0.0); }
                                        if ui.button("−0.1°").clicked(){ st.params.deskew_max_deg = (st.params.deskew_max_deg - 0.1).max(0.0); }
                                        if ui.button("+0.1°").clicked(){ st.params.deskew_max_deg = (st.params.deskew_max_deg + 0.1).min(10.0); }
                                        if ui.button("+1°").clicked()  { st.params.deskew_max_deg = (st.params.deskew_max_deg + 1.0).min(10.0); }
                                    });
                                } else {
                                    ui.horizontal(|ui| {
                                        ui.label("Ruční natočení (°)");
                                        ui.add(
                                            egui::DragValue::new(&mut st.params.manual_deskew_deg)
                                                .range(-10.0..=10.0)
                                                .clamp_existing_to_range(true)
                                                .speed(0.1)
                                        );
                                        if ui.button("−1°").clicked()  { st.params.manual_deskew_deg = (st.params.manual_deskew_deg - 1.0).max(-10.0); }
                                        if ui.button("−0.1°").clicked(){ st.params.manual_deskew_deg = (st.params.manual_deskew_deg - 0.1).max(-10.0); }
                                        if ui.button("+0.1°").clicked(){ st.params.manual_deskew_deg = (st.params.manual_deskew_deg + 0.1).min( 10.0); }
                                        if ui.button("+1°").clicked()  { st.params.manual_deskew_deg = (st.params.manual_deskew_deg + 1.0).min( 10.0); }
                                    });
                                }

                                let mut manual = st.params.manual_threshold.is_some();
                                ui.checkbox(&mut manual, "Vlastní threshold");
                                if manual {
                                    let mut v = st.params.manual_threshold.unwrap_or(128) as i32;
                                    ui.horizontal(|ui| {
                                        ui.label("Threshold");
                                        ui.add(
                                            egui::DragValue::new(&mut v)
                                                .range(0..=255)
                                                .clamp_existing_to_range(true)
                                                .speed(1)
                                        );
                                        if ui.button("−5").clicked() { v -= 5; }
                                        if ui.button("−1").clicked() { v -= 1; }
                                        if ui.button("+1").clicked() { v += 1; }
                                        if ui.button("+5").clicked() { v += 5; }
                                    });
                                    v = v.clamp(0, 255);
                                    st.params.manual_threshold = Some(v as u8);
                                } else {
                                    st.params.manual_threshold = None;
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_narovnani")
                                    .selected_text(scope_narovnani_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_narovnani_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_narovnani_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_narovnani_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_narovnani_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_narovnani_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_narovnani = true; }
                            });
                            ui.separator();

                            // ========== Okraje ==========
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Okraje");
                                egui::ComboBox::from_id_salt("margin_source")
                                    .selected_text(match st.params.margin_source {
                                        MarginSource::FromOriginal => "Z originálního skenu",
                                        MarginSource::Artificial => "Umělý okraj",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut st.params.margin_source, MarginSource::FromOriginal, "Z originálního skenu");
                                        ui.selectable_value(&mut st.params.margin_source, MarginSource::Artificial, "Umělý okraj");
                                    });

                                ui.horizontal(|ui| {
                                    ui.label("Okraj:");
                                    let suffix = match st.params.margin_units { MarginUnits::Px => " px", MarginUnits::Mm => " mm" };
                                    ui.add(
                                        egui::DragValue::new(&mut st.params.margin_amount)
                                            .range(0.0..=500.0)
                                            .clamp_existing_to_range(true)
                                            .speed(match st.params.margin_units { MarginUnits::Px => 1.0, MarginUnits::Mm => 0.5 })
                                            .suffix(suffix)
                                    );

                                    let (small, big) = match st.params.margin_units {
                                        MarginUnits::Px => (1.0, 10.0),
                                        MarginUnits::Mm => (0.5, 2.0),
                                    };
                                    if ui.button(format!("−{}", big)).clicked()  { st.params.margin_amount = (st.params.margin_amount - big).max(0.0); }
                                    if ui.button(format!("−{}", small)).clicked(){ st.params.margin_amount = (st.params.margin_amount - small).max(0.0); }
                                    if ui.button(format!("+{}", small)).clicked(){ st.params.margin_amount = (st.params.margin_amount + small).min(500.0); }
                                    if ui.button(format!("+{}", big)).clicked()  { st.params.margin_amount = (st.params.margin_amount + big).min(500.0); }

                                    egui::ComboBox::from_id_salt("margin_units")
                                        .selected_text(match st.params.margin_units { MarginUnits::Px => "px", MarginUnits::Mm => "mm" })
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(&mut st.params.margin_units, MarginUnits::Px, "px");
                                            ui.selectable_value(&mut st.params.margin_units, MarginUnits::Mm, "mm");
                                        });
                                });
                                if st.params.margin_source == MarginSource::Artificial {
                                    ui.label("Barva okraje");
                                    let mut col = egui::Color32::from_rgba_unmultiplied(
                                        st.params.margin_color[0], st.params.margin_color[1],
                                        st.params.margin_color[2], st.params.margin_color[3],
                                    );
                                    if egui::color_picker::color_edit_button_srgba(
                                        ui, &mut col, egui::color_picker::Alpha::Opaque,
                                    ).changed() {
                                        let [r, g, b, a] = col.to_array();
                                        st.params.margin_color = [r, g, b, a];
                                    }
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_okraje")
                                    .selected_text(scope_okraje_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_okraje_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_okraje_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_okraje_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_okraje_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_okraje_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_okraje = true; }
                            });
                            ui.separator();

                            // ========== Ořez ==========
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Ořez");
                                let was_enabled = st.crop_enabled;
                                ui.checkbox(&mut st.crop_enabled, "Ruční ořez (táhni v náhledu)");
                                if st.crop_enabled && !was_enabled {
                                    // nevolej self tady; jen si poznamenej
                                    need_auto_crop_rect = true;
                                }
                            } // drop(st)

                            {
                                let st = self.state_mut_for(p);
                                if ui.button("Reset ořezového rámu").clicked() {
                                    st.crop_rect_img = None;
                                    post_reset_drag = true;
                                }
                                if ui.button("Nastavit rám automaticky").clicked() {
                                    let params = st.params.clone();
                                    if let Some(r) = auto_crop_rect_for_with_params(p, &params) {
                                        st.crop_rect_img = Some(r);
                                    }
                                }
                            }

                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_orez")
                                    .selected_text(scope_orez_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_orez_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_orez_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_orez_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_orez_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_orez_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_orez = true; }
                            });
                            ui.separator();

                            // ========== Výstup ==========
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Výstup");
                                ui.label("Formát výstupu:");
                                egui::ComboBox::from_id_salt("fmt")
                                    .selected_text(match st.params.out_format {
                                        OutputFormat::Png => "PNG",
                                        OutputFormat::Jpeg => "JPEG",
                                        OutputFormat::Tiff => "TIFF",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut st.params.out_format, OutputFormat::Png, "PNG");
                                        ui.selectable_value(&mut st.params.out_format, OutputFormat::Jpeg, "JPEG");
                                        ui.selectable_value(&mut st.params.out_format, OutputFormat::Tiff, "TIFF");
                                    });
                                if st.params.out_format == OutputFormat::Jpeg {
                                    ui.add(egui::Slider::new(&mut st.params.jpeg_quality, 50..=100).text("JPEG kvalita"));
                                }
                                let mut dpi_on = st.params.dpi.is_some();
                                ui.checkbox(&mut dpi_on, "Nastavit DPI");
                                if dpi_on {
                                    let mut dpi_val = st.params.dpi.unwrap_or(300) as i32;
                                    ui.horizontal(|ui| {
                                        ui.label("DPI");
                                        ui.add(
                                            egui::DragValue::new(&mut dpi_val)
                                                .range(72..=1200)
                                                .clamp_existing_to_range(true)
                                                .speed(10)
                                        );
                                        if ui.button("−50").clicked() { dpi_val -= 50; }
                                        if ui.button("−10").clicked() { dpi_val -= 10; }
                                        if ui.button("+10").clicked() { dpi_val += 10; }
                                        if ui.button("+50").clicked() { dpi_val += 50; }
                                    });
                                    dpi_val = dpi_val.clamp(72, 1200);
                                    st.params.dpi = Some(dpi_val as u32);
                                } else {
                                    st.params.dpi = None;
                                }
                                ui.checkbox(&mut st.params.strip_metadata, "Smazat metadata");
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_vystup")
                                    .selected_text(scope_vystup_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_vystup_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_vystup_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_vystup_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_vystup_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_vystup_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_vystup = true; }
                            });

                            // ========== Přejmenování ==========
                            ui.separator();
                            {
                                let st = self.state_mut_for(p);
                                ui.heading("Přejmenování skenů");
                                ui.checkbox(&mut st.params.rename_enabled, "Přejmenovat skeny");
                                if st.params.rename_enabled {
                                    ui.horizontal(|ui| {
                                        ui.label("Maska:");
                                        ui.text_edit_singleline(&mut st.params.rename_mask);
                                        ui.label("(# = číslice)");
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label("Začít od:");
                                        ui.add(
                                            egui::DragValue::new(&mut st.params.rename_start)
                                                .range(1..=999_999)
                                                .clamp_existing_to_range(true)
                                                .speed(1)
                                        );
                                        if ui.button("−10").clicked() { st.params.rename_start = st.params.rename_start.saturating_sub(10).max(1); }
                                        if ui.button("−1").clicked()  { st.params.rename_start = st.params.rename_start.saturating_sub(1).max(1); }
                                        if ui.button("+1").clicked()  { st.params.rename_start = (st.params.rename_start + 1).min(999_999); }
                                        if ui.button("+10").clicked() { st.params.rename_start = (st.params.rename_start + 10).min(999_999); }
                                    });
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Aplikovat na…");
                                egui::ComboBox::from_id_salt("scope_rename")
                                    .selected_text(scope_rename_local.label())
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut scope_rename_local, ApplyScope::OnlyThis,           ApplyScope::OnlyThis.label());
                                        ui.selectable_value(&mut scope_rename_local, ApplyScope::All,                ApplyScope::All.label());
                                        ui.selectable_value(&mut scope_rename_local, ApplyScope::FromHere,           ApplyScope::FromHere.label());
                                        ui.selectable_value(&mut scope_rename_local, ApplyScope::EveryOther,         ApplyScope::EveryOther.label());
                                        ui.selectable_value(&mut scope_rename_local, ApplyScope::EveryOtherFromHere, ApplyScope::EveryOtherFromHere.label());
                                    });
                                if ui.button("Aplikovat").clicked() { do_apply_rename = true; }
                            });

                            ui.separator();

                            // Zpracování
                            let btn_size = egui::vec2(ui.available_width(), 36.0);
                            let green = egui::Color32::from_rgb(70, 150, 90);
                            if ui.add_sized(
                                btn_size,
                                egui::Button::new("Zpracovat vybraný")
                                    .fill(green)
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_black_alpha(40))),
                            ).clicked() {
                                self.start_job(false);
                            }
                            if ui.add_sized(
                                btn_size,
                                egui::Button::new("Zpracovat vše")
                                    .fill(green)
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_black_alpha(40))),
                            ).clicked() {
                                self.start_job(true);
                            }
                        } else {
                            ui.label("Načti soubor, aby šlo ovládací prvky použít.");
                        }
                    }); // === konec closure ===
					// --- PROJEKT (uvnitř SidePanel::show, mimo add_enabled_ui, aby byl vždy aktivní) ---
                    ui.separator();
                    ui.heading("Projekt");
                    ui.horizontal_wrapped(|ui| {
                        let can_quick_save = self.last_project_path.is_some();
                        if ui.add_enabled(can_quick_save, egui::Button::new("Uložit")).clicked() {
                            self.save_project_to_last_or_as();
                        }
                        if ui.button("Uložit jako…").clicked() {
                            self.save_project_as_dialog();
                        }
                        if ui.button("Otevřít").clicked() {
                            self.open_project_dialog();
                        }
                    });
                    if let Some(p) = &self.last_project_path {
                        ui.small(format!("Akt. projekt: {}", p.display()));
                    }

                    ui.separator();
                    ui.label(&self.progress_text);

                    // 2) ZÁPIS SCOPE ZPĚT A SPUŠTĚNÍ AKCÍ
                    self.scope_rotace    = scope_rotace_local;
                    self.scope_narovnani = scope_narovnani_local;
                    self.scope_okraje    = scope_okraje_local;
                    self.scope_orez      = scope_orez_local;
                    self.scope_split     = scope_split_local;
                    self.scope_vystup    = scope_vystup_local;
                    self.scope_rename    = scope_rename_local;

                    if let Some(sel) = selected_idx_copy {
                        if do_apply_rotace    { self.apply_rotation(sel, self.scope_rotace); }
                        if do_apply_narovnani { self.apply_narovnani(sel, self.scope_narovnani); }
                        if do_apply_okraje    { self.apply_okraje(sel,    self.scope_okraje); }
                        if do_apply_orez      { self.apply_orez(sel,      self.scope_orez); }
                        if do_apply_split     { self.apply_split(sel,     self.scope_split); }
                        if do_apply_vystup    { self.apply_vystup(sel,    self.scope_vystup); }
                        if do_apply_rename    { self.apply_rename(sel,    self.scope_rename); }
                    }

                    if need_auto_crop_rect {
                        if let Some(p) = selected_path.as_ref() {
                            self.ensure_auto_crop_rect_if_needed(p);
                        }
                    }
                    });

        } else {
            // Panel je skrytý – "úchyt" pro znovuotevření
            egui::Area::new(egui::Id::new("left_reveal_handle"))
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(6.0, 60.0))
                .show(ctx, |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        if ui.button("▶ Ovládání").clicked() {
                            self.left_panel_open = true;
                        }
                    });
                });
        }

        if post_reset_drag {
            self.drag_handle = None;
            self.drag_origin_rect = None;
            self.drag_start_mouse_img = None;
            self.drag_aspect = None;
        }

        // Pravý panel – seznam (lze skrýt), filtr a miniatury
        if self.right_panel_open {
            egui::SidePanel::right("right")
                .resizable(true)            // lze měnit šířku tahem
                .default_width(220.0)       // výchozí šířka (<= 256 px)
                .min_width(100.0)           // lze zmenšit cca na 100 px
                .max_width(640.0)
                .show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.heading("Soubory");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Skrýt ▶").clicked() {
                                self.right_panel_open = false;
                            }
                        });
                    });

                    ui.separator();

                    // Styl aktivního tlačítka (stejný jako u Rotace)
                    let active_fill   = egui::Color32::from_gray(60);
                    let active_stroke = egui::Stroke::new(1.0, egui::Color32::from_black_alpha(60));

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Filtrovat:");
                        ui.add_sized(
                            egui::vec2(140.0, 0.0),
                            egui::TextEdit::singleline(&mut self.file_filter)
                        );

                        // Přepínač „Miniatury“ s aktivním vzhledem
                        let btn = if self.show_thumbs {
                            egui::Button::new(egui::RichText::new("Miniatury").color(egui::Color32::WHITE))
                                .fill(active_fill)
                                .stroke(active_stroke)
                                .min_size(egui::vec2(90.0, 24.0))
                        } else {
                            egui::Button::new("Miniatury")
                                .min_size(egui::vec2(90.0, 24.0))
                        };
                        if ui.add(btn).clicked() {
                            self.show_thumbs = !self.show_thumbs;
                        }
                    });

                    if self.show_thumbs {
                        ui.add(
                            egui::Slider::new(&mut self.thumb_size, 32.0..=256.0)
                                .text("Velikost náhledu")
                        );
                    }

                    ui.separator();

                    // --- Rozdělení (dvoustrana): náhled levé/pravé stránky ---
                    if let Some(sel_idx) = self.selected {
                        if let Some(p) = self.files.get(sel_idx).cloned() {
                            let st = self.state_for(&p).cloned().unwrap_or_default();
                            if st.params.split_enabled {
                                if let Some(tex) = self.preview_texture(ui.ctx(), &p) {
                                    let split_top = st.params.split_x_top_norm.clamp(0.05, 0.95);
                let split_bottom = st.params.split_x_bottom_norm.clamp(0.05, 0.95);
                let split = ((split_top + split_bottom) * 0.5).clamp(0.05, 0.95);
                                    let tex_sz = tex.size_vec2();
                                    let w = tex_sz.x.max(1.0);
                                    let h = tex_sz.y.max(1.0);

                                    ui.collapsing("Rozdělení (náhled)", |ui| {
                                        let avail_w = ui.available_width().max(40.0);

                                        // Levá
                                        ui.label("Levá stránka");
                                        let max_h = 360.0;
                                        let left_w_px = (w * split).max(1.0);
                                        let left_h_px = h.max(1.0);
                                        let scale_l = (avail_w / left_w_px).min(max_h / left_h_px).max(0.01);
                                        let draw_l = egui::vec2(left_w_px * scale_l, left_h_px * scale_l);
                                        let uv_l = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(split, 1.0));
                                        ui.add(egui::widgets::Image::new((tex.id(), draw_l)).uv(uv_l));

                                        ui.separator();

                                        // Pravá
                                        ui.label("Pravá stránka");
                                        let right_w_px = (w * (1.0 - split)).max(1.0);
                                        let right_h_px = h.max(1.0);
                                        let scale_r = (avail_w / right_w_px).min(max_h / right_h_px).max(0.01);
                                        let draw_r = egui::vec2(right_w_px * scale_r, right_h_px * scale_r);
                                        let uv_r = egui::Rect::from_min_max(egui::pos2(split, 0.0), egui::pos2(1.0, 1.0));
                                        ui.add(egui::widgets::Image::new((tex.id(), draw_r)).uv(uv_r));
                                    });

                                    ui.separator();
                                }
                            }
                        }
                    }

                    // Připrav seznam předem, ať během kreslení nedržíme borrow na self.files
                    let filter_lc = self.file_filter.to_ascii_lowercase();
                    let files_to_show: Vec<(usize, PathBuf)> = self.files
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| {
                            let name = p.file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_ascii_lowercase();
                            if filter_lc.is_empty() || name.contains(&filter_lc) {
                                Some((i, p.clone()))
                            } else {
                                None
                            }
                        })
                        .collect();

                    let mut remove_idx: Option<usize> = None;

                    egui::ScrollArea::vertical()
                        .id_salt("right_files_scroll")
                        .show(ui, |ui| {
                            for (i, p) in files_to_show {
                                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                                let selected = Some(i) == self.selected;

                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        // Miniatura
                                        if self.show_thumbs {
                                            if let Some(tex) = self.thumb_texture(ui.ctx(), &p) {
                                                let tex_sz = tex.size_vec2();
                                                let scale = (self.thumb_size / tex_sz.x.max(tex_sz.y)).max(0.01);
                                                let draw = tex_sz * scale;
                                                let img_resp = ui.add(
                                                    egui::widgets::Image::new((tex.id(), draw))
                                                        .sense(egui::Sense::click())
                                                );
                                                if img_resp.clicked() {
                                                    self.selected = Some(i);
                                                }
                                            } else {
                                                let (rect, _resp) = ui.allocate_exact_size(
                                                    egui::vec2(self.thumb_size, self.thumb_size),
                                                    egui::Sense::hover()
                                                );
                                                ui.painter().rect_filled(rect, 4.0, egui::Color32::from_gray(40));
                                            }
                                        }

                                        // Název + výběr + kontextová nabídka
                                        let resp = ui.selectable_label(selected, name);
                                        if resp.clicked() {
                                            self.selected = Some(i);
                                        }
                                        resp.context_menu(|ui| {
                                            if ui.button("Přidat soubor(y) před…").clicked() {
                                                if let Some(paths) = rfd::FileDialog::new()
                                                    .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
                                                    .pick_files()
                                                {
                                                    self.insert_files_at(i, paths);
                                                }
                                                ui.close_menu();
                                            }
                                            if ui.button("Přidat soubor(y) za…").clicked() {
                                                if let Some(paths) = rfd::FileDialog::new()
                                                    .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
                                                    .pick_files()
                                                {
                                                    self.insert_files_at(i.saturating_add(1), paths);
                                                }
                                                ui.close_menu();
                                            }
                                            ui.separator();
                                            if ui.button("Odebrat ze seznamu").clicked() {
                                                remove_idx = Some(i);
                                                ui.close_menu();
                                            }
                                        });
                                    });
                                });
                            }
                        });

                    // Odebrání položky až po vykreslení seznamu
                    if let Some(rem) = remove_idx {
                        if rem < self.files.len() {
                            self.files.remove(rem);
                            if let Some(sel) = self.selected {
                                if sel == rem {
                                    self.selected = None;
                                } else if sel > rem {
                                    self.selected = Some(sel - 1);
                                }
                            }
                        }
                    }
                });
        } else {
            // Panel je skrytý – "úchyt" pro znovuotevření
            egui::Area::new(egui::Id::new("right_reveal_handle"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-6.0, 60.0))
                .show(ctx, |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        if ui.button("◀ Soubory").clicked() {
                            self.right_panel_open = true;
                        }
                    });
                });
        }

        // Střed – náhled a editace rámu
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Náhled");
            ui.separator();

            let selected_path: Option<PathBuf> =
                self.selected.and_then(|idx| self.files.get(idx).cloned());

            if let Some(p) = selected_path {
                // auto-rám, když je zapnut ruční a není rám
                self.ensure_auto_crop_rect_if_needed(&p);

                if let Some(tex) = self.preview_texture(ui.ctx(), &p) {
                    let (crop_enabled, mut crop_rect_img, split_enabled, split_x_top_norm, split_x_bottom_norm) = {
                        let st_snapshot = self.state_for(&p).cloned().unwrap_or_default();
                        (
                            st_snapshot.crop_enabled,
                            st_snapshot.crop_rect_img,
                            st_snapshot.params.split_enabled,
                            st_snapshot.params.split_x_top_norm,
                            st_snapshot.params.split_x_bottom_norm,
                        )
                    };

                    egui::ScrollArea::both()
                        .drag_to_scroll(false)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let tex_size = tex.size_vec2();
                            let draw_size = tex_size * self.zoom;

                            let img = egui::widgets::Image::new((tex.id(), draw_size))
                                .sense(egui::Sense::click_and_drag());
                            let resp = ui.add(img);
                            let painter = ui.painter_at(resp.rect);

                            // převody: screen <-> image
                            // POZOR: closure nesmí sahat na `self`, jinak nejde později mutovat stav (borrow checker).
                            let zoom = self.zoom;
                            let rect_min = resp.rect.min;
                            let tex_w = tex_size.x;
                            let tex_h = tex_size.y;

                            let to_img = move |screen: egui::Pos2| {
                                let mut p = screen - rect_min;
                                p.x /= zoom;
                                p.y /= zoom;
                                p.x = p.x.clamp(0.0, tex_w);
                                p.y = p.y.clamp(0.0, tex_h);
                                egui::pos2(p.x, p.y)
                            };

                            let to_screen =
                                move |p_img: egui::Pos2| rect_min + p_img.to_vec2() * zoom;

                            // ===== Page splitting divider (drag to adjust) =====
                            if split_enabled {
                                let line_color = egui::Color32::from_rgb(220, 200, 80);

                                let x_top_img = (split_x_top_norm.clamp(0.05, 0.95) * tex_size.x).clamp(0.0, tex_size.x);
                                let x_bottom_img = (split_x_bottom_norm.clamp(0.05, 0.95) * tex_size.x).clamp(0.0, tex_size.x);

                                let top_pt = to_screen(egui::pos2(x_top_img, 0.0));
                                let bottom_pt = to_screen(egui::pos2(x_bottom_img, tex_size.y));

                                painter.line_segment(
                                    [top_pt, bottom_pt],
                                    egui::Stroke::new(2.0, line_color),
                                );

                                // small handles (top/bottom) to hint independent movement
                                painter.circle_filled(top_pt, 4.0, line_color);
                                painter.circle_filled(bottom_pt, 4.0, line_color);

                                // Handle hover + drag (keep it simple, independent of crop handles)
                                let split_band_px = 10.0_f32;
                                let mut hover = false;
                                if let Some(pointer) = resp.hover_pos() {
                                    let denom = (bottom_pt.y - top_pt.y).max(1.0);
                                    let ty = ((pointer.y - top_pt.y) / denom).clamp(0.0, 1.0);
                                    let x_line = top_pt.x + (bottom_pt.x - top_pt.x) * ty;

                                    if (pointer.x - x_line).abs() <= split_band_px {
                                        hover = true;
                                        ui.output_mut(|o| o.cursor_icon = CursorIcon::ResizeColumn);
                                    }
                                }
                                if resp.drag_started() && hover {
                                    self.split_dragging = true;

                                    // decide which part of the axis we're dragging
                                    if let Some(pointer) = resp.interact_pointer_pos() {
                                        let denom = (bottom_pt.y - top_pt.y).max(1.0);
                                        let ty = ((pointer.y - top_pt.y) / denom).clamp(0.0, 1.0);
                                        self.split_drag_kind = if ty <= 0.25 {
                                            SplitDragKind::Top
                                        } else if ty >= 0.75 {
                                            SplitDragKind::Bottom
                                        } else {
                                            SplitDragKind::Both
                                        };
                                    } else {
                                        self.split_drag_kind = SplitDragKind::Both;
                                    }

                                    // prevent crop drag from latching in the same gesture
                                    self.drag_handle = None;
                                    self.drag_origin_rect = None;
                                    self.drag_start_mouse_img = None;
                                    self.drag_aspect = None;
                                }
                                if self.split_dragging && resp.dragged() {
                                    if let Some(pointer) = resp.interact_pointer_pos() {
                                        let p_img = to_img(pointer);
                                        let y_norm = (p_img.y / tex_size.y).clamp(0.0, 1.0);
                                        let new_norm = (p_img.x / tex_size.x).clamp(0.05, 0.95);
                                        let drag_kind = self.split_drag_kind;

                                        let st = self.state_mut_for(&p);
                                        let cur_top = st.params.split_x_top_norm;
                                        let cur_bottom = st.params.split_x_bottom_norm;
                                        let cur_at_y = cur_top + (cur_bottom - cur_top) * y_norm;

                                        match drag_kind {
                                            SplitDragKind::Top => {
                                                st.params.split_x_top_norm = new_norm;
                                            }
                                            SplitDragKind::Bottom => {
                                                st.params.split_x_bottom_norm = new_norm;
                                            }
                                            SplitDragKind::Both => {
                                                let delta = new_norm - cur_at_y;
                                                st.params.split_x_top_norm = (cur_top + delta).clamp(0.05, 0.95);
                                                st.params.split_x_bottom_norm = (cur_bottom + delta).clamp(0.05, 0.95);
                                            }
                                        }
                                    }
                                }
                                if self.split_dragging && resp.drag_stopped() {
                                    self.split_dragging = false;
                                    self.split_drag_kind = SplitDragKind::Both;
                                }
                            }

if crop_enabled && crop_rect_img.is_none() {
                                crop_rect_img = Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), tex_size));
                            }

                            // pomocná: clamp + min size
                            let clamp_rect = |mut min: egui::Pos2,
                                              mut max: egui::Pos2,
                                              min_w: f32,
                                              min_h: f32| {
                                if min.x > max.x { std::mem::swap(&mut min.x, &mut max.x); }
                                if min.y > max.y { std::mem::swap(&mut min.y, &mut max.y); }
                                if max.x - min.x < min_w { max.x = (min.x + min_w).min(tex_size.x); }
                                if max.y - min.y < min_h { max.y = (min.y + min_h).min(tex_size.y); }
                                min.x = min.x.clamp(0.0, tex_size.x);
                                min.y = min.y.clamp(0.0, tex_size.y);
                                max.x = max.x.clamp(0.0, tex_size.x);
                                max.y = max.y.clamp(0.0, tex_size.y);
                                egui::Rect::from_min_max(min, max)
                            };

                            // ===== Interakce s rámem =====
                            if crop_enabled {
                                // 1) Určení hovered handle s prioritou rohy → hrany → move
                                let mut hovered_handle: Option<CropHandle> = None;
                                if let Some(r) = crop_rect_img {
                                    let rect_screen = egui::Rect::from_min_max(to_screen(r.min), to_screen(r.max));
                                    let corner_sq = |pt: egui::Pos2| -> egui::Rect {
                                        egui::Rect::from_center_size(to_screen(pt), egui::vec2(CORNER_BAND, CORNER_BAND))
                                    };
                                    // hranné pásy
                                    let left_band = egui::Rect::from_min_max(
                                        egui::pos2(rect_screen.min.x - EDGE_BAND, rect_screen.min.y),
                                        egui::pos2(rect_screen.min.x + EDGE_BAND, rect_screen.max.y),
                                    );
                                    let right_band = egui::Rect::from_min_max(
                                        egui::pos2(rect_screen.max.x - EDGE_BAND, rect_screen.min.y),
                                        egui::pos2(rect_screen.max.x + EDGE_BAND, rect_screen.max.y),
                                    );
                                    let top_band = egui::Rect::from_min_max(
                                        egui::pos2(rect_screen.min.x, rect_screen.min.y - EDGE_BAND),
                                        egui::pos2(rect_screen.max.x, rect_screen.min.y + EDGE_BAND),
                                    );
                                    let bottom_band = egui::Rect::from_min_max(
                                        egui::pos2(rect_screen.min.x, rect_screen.max.y - EDGE_BAND),
                                        egui::pos2(rect_screen.max.x, rect_screen.max.y + EDGE_BAND),
                                    );

                                    let move_zone = rect_screen.shrink(EDGE_BAND);

                                    if let Some(pointer) = resp.hover_pos() {
                                        // rohy (priorita)
                                        let nw = r.min;
                                        let ne = egui::pos2(r.max.x, r.min.y);
                                        let sw = egui::pos2(r.min.x, r.max.y);
                                        let se = r.max;

                                        let p2 = pointer;
                                        if corner_sq(nw).contains(p2)      { hovered_handle = Some(CropHandle::NW); }
                                        else if corner_sq(ne).contains(p2) { hovered_handle = Some(CropHandle::NE); }
                                        else if corner_sq(sw).contains(p2) { hovered_handle = Some(CropHandle::SW); }
                                        else if corner_sq(se).contains(p2) { hovered_handle = Some(CropHandle::SE); }
                                        // hrany
                                        else if left_band.contains(p2)  { hovered_handle = Some(CropHandle::W); }
                                        else if right_band.contains(p2) { hovered_handle = Some(CropHandle::E); }
                                        else if top_band.contains(p2)   { hovered_handle = Some(CropHandle::N); }
                                        else if bottom_band.contains(p2){ hovered_handle = Some(CropHandle::S); }
                                        // přesun (jen pokud dostatečně uvnitř)
                                        else if move_zone.contains(p2) {
                                            hovered_handle = Some(CropHandle::Move);
                                        }
                                    }

                                    // kurzor
                                    let cursor = match hovered_handle {
                                        Some(CropHandle::NW) | Some(CropHandle::SE) => Some(CursorIcon::ResizeNwSe),
                                        Some(CropHandle::NE) | Some(CropHandle::SW) => Some(CursorIcon::ResizeNeSw),
                                        Some(CropHandle::N)  | Some(CropHandle::S)  => Some(CursorIcon::ResizeRow),
                                        Some(CropHandle::W)  | Some(CropHandle::E)  => Some(CursorIcon::ResizeColumn),
                                        Some(CropHandle::Move) => Some(CursorIcon::Move),
                                        _ => None,
                                    };
                                    if let Some(c) = cursor { ui.output_mut(|o| o.cursor_icon = c); }
                                }

                                // 2) Drag start
                                if resp.drag_started() {
                                    let mods = ui.input(|i| i.modifiers);
                                    self.drag_keep_ratio = mods.shift;
                                    self.drag_from_center = mods.alt;

                                    let p_img = resp.interact_pointer_pos().map(to_img).unwrap_or(egui::pos2(0.0, 0.0));
                                    self.drag_start_mouse_img = Some(p_img);

                                    if let Some(r) = crop_rect_img {
                                        if let Some(h) = hovered_handle {
                                            self.drag_handle = Some(h);
                                            self.drag_origin_rect = Some(r);
                                            let aspect = (r.height() / r.width().max(1e-6)).max(1e-6);
                                            self.drag_aspect = Some(aspect);
                                        } else {
                                            if mods.ctrl || crop_rect_img.is_none() {
                                                self.drag_handle = Some(CropHandle::SE);
                                                self.drag_origin_rect = Some(egui::Rect::from_min_max(p_img, p_img));
                                                self.drag_aspect = Some(1.0);
                                                crop_rect_img = self.drag_origin_rect;
                                            } else {
                                                self.drag_handle = None;
                                                self.drag_origin_rect = None;
                                                self.drag_aspect = None;
                                            }
                                        }
                                    } else {
                                        // žádný rám → začni nový
                                        self.drag_handle = Some(CropHandle::SE);
                                        self.drag_origin_rect = Some(egui::Rect::from_min_max(p_img, p_img));
                                        self.drag_aspect = Some(1.0);
                                        crop_rect_img = self.drag_origin_rect;
                                    }
                                }

                                // 3) Drag update
                                if resp.dragged() {
                                    if let (Some(h), Some(origin), Some(start), Some(aspect)) =
                                        (self.drag_handle, self.drag_origin_rect, self.drag_start_mouse_img, self.drag_aspect)
                                    {
                                        if let Some(pointer) = resp.interact_pointer_pos() {
                                            let p_img = to_img(pointer);
                                            let center = origin.center();

                                            let make_rect_corner = |pivot: egui::Pos2,
                                                                    p: egui::Pos2,
                                                                    keep_ratio: bool,
                                                                    aspect: f32| {
                                                let mut dx = p.x - pivot.x;
                                                let mut dy = p.y - pivot.y;
                                                if keep_ratio {
                                                    let want_dy = dx.abs() * aspect;
                                                    let want_dx = (dy.abs() / aspect).abs();
                                                    if want_dy > dy.abs() { dy = want_dy.copysign(dy); }
                                                    else { dx = want_dx.copysign(dx); }
                                                }
                                                let a = egui::pos2(pivot.x.min(pivot.x + dx), pivot.y.min(pivot.y + dy));
                                                let b = egui::pos2(pivot.x.max(pivot.x + dx), pivot.y.max(pivot.y + dy));
                                                egui::Rect::from_min_max(a, b)
                                            };
                                            let make_rect_centered = |c: egui::Pos2,
                                                                      p: egui::Pos2,
                                                                      keep_ratio: bool,
                                                                      aspect: f32| {
                                                let mut hx = (p.x - c.x).abs();
                                                let mut hy = (p.y - c.y).abs();
                                                if keep_ratio {
                                                    let hy_from_hx = aspect * hx;
                                                    let hx_from_hy = if aspect > 0.0 { hy / aspect } else { hx };
                                                    if hy_from_hx > hy { hy = hy_from_hx; } else { hx = hx_from_hy; }
                                                }
                                                egui::Rect::from_min_max(
                                                    egui::pos2(c.x - hx, c.y - hy),
                                                    egui::pos2(c.x + hx, c.y + hy),
                                                )
                                            };

                                            // nový návrh dle úchytu
                                            let proposed: egui::Rect = match h {
                                                CropHandle::Move => {
                                                    let delta = p_img - start;
                                                    egui::Rect::from_min_max(origin.min + delta, origin.max + delta)
                                                }
                                                // rohy
                                                CropHandle::NW => {
                                                    if self.drag_from_center { make_rect_centered(center, p_img, self.drag_keep_ratio, aspect) }
                                                    else { make_rect_corner(egui::pos2(origin.max.x, origin.max.y), p_img, self.drag_keep_ratio, aspect) }
                                                }
                                                CropHandle::NE => {
                                                    if self.drag_from_center { make_rect_centered(center, p_img, self.drag_keep_ratio, aspect) }
                                                    else { make_rect_corner(egui::pos2(origin.min.x, origin.max.y), p_img, self.drag_keep_ratio, aspect) }
                                                }
                                                CropHandle::SW => {
                                                    if self.drag_from_center { make_rect_centered(center, p_img, self.drag_keep_ratio, aspect) }
                                                    else { make_rect_corner(egui::pos2(origin.max.x, origin.min.y), p_img, self.drag_keep_ratio, aspect) }
                                                }
                                                CropHandle::SE => {
                                                    if self.drag_from_center { make_rect_centered(center, p_img, self.drag_keep_ratio, aspect) }
                                                    else { make_rect_corner(egui::pos2(origin.min.x, origin.min.y), p_img, self.drag_keep_ratio, aspect) }
                                                }
                                                // hrany
                                                CropHandle::W => {
                                                    if self.drag_from_center {
                                                        let new_min_x = center.x - (center.x - p_img.x).abs();
                                                        let new_max_x = center.x + (center.x - p_img.x).abs();
                                                        if self.drag_keep_ratio {
                                                            let width = (new_max_x - new_min_x).max(1.0);
                                                            let height = width * aspect;
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(new_min_x, center.y - height / 2.0),
                                                                egui::pos2(new_max_x, center.y + height / 2.0),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(new_min_x, origin.min.y),
                                                                egui::pos2(new_max_x, origin.max.y),
                                                            )
                                                        }
                                                    } else {
                                                        let new_min_x = p_img.x;
                                                        if self.drag_keep_ratio {
                                                            let width = (origin.max.x - new_min_x).abs().max(1.0);
                                                            let height = width * aspect;
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, center.y - height / 2.0),
                                                                egui::pos2(center.x + width / 2.0, center.y + height / 2.0),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(new_min_x.min(origin.max.x), origin.min.y),
                                                                egui::pos2(new_min_x.max(origin.max.x), origin.max.y),
                                                            )
                                                        }
                                                    }
                                                }
                                                CropHandle::E => {
                                                    if self.drag_from_center {
                                                        let new_max_x = center.x + (p_img.x - center.x).abs();
                                                        let new_min_x = center.x - (p_img.x - center.x).abs();
                                                        if self.drag_keep_ratio {
                                                            let width = (new_max_x - new_min_x).max(1.0);
                                                            let height = width * aspect;
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(new_min_x, center.y - height / 2.0),
                                                                egui::pos2(new_max_x, center.y + height / 2.0),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(new_min_x, origin.min.y),
                                                                egui::pos2(new_max_x, origin.max.y),
                                                            )
                                                        }
                                                    } else {
                                                        let new_max_x = p_img.x;
                                                        if self.drag_keep_ratio {
                                                            let width = (new_max_x - origin.min.x).abs().max(1.0);
                                                            let height = width * aspect;
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, center.y - height / 2.0),
                                                                egui::pos2(center.x + width / 2.0, center.y + height / 2.0),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(origin.min.x.min(new_max_x), origin.min.y),
                                                                egui::pos2(origin.min.x.max(new_max_x), origin.max.y),
                                                            )
                                                        }
                                                    }
                                                }
                                                CropHandle::N => {
                                                    if self.drag_from_center {
                                                        let new_min_y = center.y - (center.y - p_img.y).abs();
                                                        let new_max_y = center.y + (center.y - p_img.y).abs();
                                                        if self.drag_keep_ratio {
                                                            let height = (new_max_y - new_min_y).max(1.0);
                                                            let width = height / aspect.max(1e-6);
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, new_min_y),
                                                                egui::pos2(center.x + width / 2.0, new_max_y),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(origin.min.x, new_min_y),
                                                                egui::pos2(origin.max.x, new_max_y),
                                                            )
                                                        }
                                                    } else {
                                                        let new_min_y = p_img.y;
                                                        if self.drag_keep_ratio {
                                                            let height = (origin.max.y - new_min_y).abs().max(1.0);
                                                            let width = height / aspect.max(1e-6);
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, new_min_y.min(origin.max.y)),
                                                                egui::pos2(center.x + width / 2.0, new_min_y.max(origin.max.y)),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(origin.min.x, new_min_y.min(origin.max.y)),
                                                                egui::pos2(origin.max.x, new_min_y.max(origin.max.y)),
                                                            )
                                                        }
                                                    }
                                                }
                                                CropHandle::S => {
                                                    if self.drag_from_center {
                                                        let new_max_y = center.y + (p_img.y - center.y).abs();
                                                        let new_min_y = center.y - (p_img.y - center.y).abs();
                                                        if self.drag_keep_ratio {
                                                            let height = (new_max_y - new_min_y).max(1.0);
                                                            let width = height / aspect.max(1e-6);
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, new_min_y),
                                                                egui::pos2(center.x + width / 2.0, new_max_y),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(origin.min.x, new_min_y),
                                                                egui::pos2(origin.max.x, new_max_y),
                                                            )
                                                        }
                                                    } else {
                                                        let new_max_y = p_img.y;
                                                        if self.drag_keep_ratio {
                                                            let height = (new_max_y - origin.min.y).abs().max(1.0);
                                                            let width = height / aspect.max(1e-6);
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(center.x - width / 2.0, origin.min.y.min(new_max_y)),
                                                                egui::pos2(center.x + width / 2.0, origin.min.y.max(new_max_y)),
                                                            )
                                                        } else {
                                                            egui::Rect::from_min_max(
                                                                egui::pos2(origin.min.x, origin.min.y.min(new_max_y)),
                                                                egui::pos2(origin.max.x, origin.min.y.max(new_max_y)),
                                                            )
                                                        }
                                                    }
                                                }
                                            };

                                            let r2 = clamp_rect(proposed.min, proposed.max, 1.0, 1.0);
                                            crop_rect_img = Some(r2);
                                        }
                                    }
                                }

                                // 4) Kreslení
                                if let Some(r) = crop_rect_img {
                                    let rect_screen = egui::Rect::from_min_max(to_screen(r.min), to_screen(r.max));
                                    // stín kolem
                                    let outside = [
                                        egui::Rect::from_min_max(
                                            resp.rect.min, egui::pos2(rect_screen.min.x, resp.rect.max.y)),
                                        egui::Rect::from_min_max(
                                            egui::pos2(rect_screen.max.x, resp.rect.min.y), resp.rect.max),
                                        egui::Rect::from_min_max(
                                            egui::pos2(rect_screen.min.x, rect_screen.min.y),
                                            egui::pos2(rect_screen.max.x, rect_screen.min.y)),
                                        egui::Rect::from_min_max(
                                            egui::pos2(rect_screen.min.x, rect_screen.max.y),
                                            egui::pos2(rect_screen.max.x, resp.rect.max.y)),
                                    ];
                                    let shade = egui::Color32::from_black_alpha(120);
                                    for o in outside { painter.rect_filled(o, 0.0, shade); }
                                    painter.rect_stroke(rect_screen, 0.0, egui::Stroke::new(2.0, egui::Color32::YELLOW));

                                    // rohové „úchyty“
                                    for &pt in &[
                                        r.min,
                                        egui::pos2(r.max.x, r.min.y),
                                        egui::pos2(r.min.x, r.max.y),
                                        r.max,
                                    ] {
                                        painter.circle_filled(to_screen(pt), 4.0, egui::Color32::YELLOW);
                                    }
                                }

                                // 5) Drag stop
                                if resp.drag_stopped() {
                                    self.drag_handle = None;
                                    self.drag_origin_rect = None;
                                    self.drag_start_mouse_img = None;
                                    self.drag_aspect = None;
                                }
                            } // if crop_enabled
                        });

                    // zapsat zpět
                    let st = self.state_mut_for(&p);
                    st.crop_enabled = crop_enabled;
                    st.crop_rect_img = crop_rect_img;
                } else {
                    ui.label("Náhled se nepodařilo vytvořit.");
                }
            } else {
                ui.label("Přidej a vyber soubor pro náhled.");
            }
        });

        // Zoom kolečko (Ctrl)
        ctx.input(|i| {
            if i.modifiers.ctrl {
                let dy = i.smooth_scroll_delta.y + i.raw_scroll_delta.y;
                if dy != 0.0 {
                    let factor = if dy > 0.0 { 1.05 } else { 0.95 };
                    self.zoom = (self.zoom * factor).clamp(0.1, 4.0);
                }
            }
        });

        // Listování
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowDown) {
                if let Some(sel) = self.selected {
                    if sel + 1 < self.files.len() { self.selected = Some(sel + 1); }
                } else if !self.files.is_empty() {
                    self.selected = Some(0);
                }
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                if let Some(sel) = self.selected {
                    if sel > 0 { self.selected = Some(sel - 1); }
                } else if !self.files.is_empty() {
                    self.selected = Some(0);
                }
            }
        });
    }
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp")
    )
}

// ===== Rotace + jemné narovnání (text-first deskew) =====

fn resolve_deskew_angle_from_gray(gray: &ImageBuffer<Luma<u8>, Vec<u8>>, params: &Params) -> f32 {
    if params.auto_deskew {
        if params.deskew_max_deg.abs() > 0.0 {
            estimate_skew_deg_text_projection(gray, params.deskew_max_deg.abs())
        } else {
            0.0
        }
    } else {
        params.manual_deskew_deg
    }
}

fn make_rotated_rgba(img: &DynamicImage, params: &Params) -> (RgbaImage, u8) {
    // pevná rotace (bez interpolace)
    let mut rgba = img.to_rgba8();
    rgba = match params.rotation {
        Rotation::None    => rgba,
        Rotation::Right90 => image::imageops::rotate90(&rgba),
        Rotation::Left90  => image::imageops::rotate270(&rgba),
        Rotation::Rot180  => image::imageops::rotate180(&rgba),
    };

    // threshold + jemné narovnání – deskew podle vodorovných linií
    let gray = DynamicImage::ImageRgba8(rgba.clone()).to_luma8();
    let thr = params.manual_threshold.unwrap_or_else(|| otsu_level(&gray));
    let angle = resolve_deskew_angle_from_gray(&gray, params);

    let rotated = rotate_about_center(
        &rgba,
        angle.to_radians(),
        Interpolation::Bilinear,
        Rgba([255, 255, 255, 255]),
    );
    (rotated, thr)
}

// ===== Pomocné zpracování =====
fn threshold_to_mask(gray: &ImageBuffer<Luma<u8>, Vec<u8>>, thr: u8) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    map_colors(gray, |p| if p[0] < thr { Luma([255]) } else { Luma([0]) })
}

/// Downscale šedotónového obrázku tak, aby max(w,h)=max_dim (zachování poměru stran).
fn downscale_gray(
    gray: &ImageBuffer<Luma<u8>, Vec<u8>>,
    max_dim: u32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = gray.dimensions();
    let scale = (max_dim as f32 / w.max(h) as f32).min(1.0);
    if scale >= 0.999 {
        return gray.clone();
    }
    let nw = (w as f32 * scale).round().max(1.0) as u32;
    let nh = (h as f32 * scale).round().max(1.0) as u32;
    imageops::resize(gray, nw, nh, imageops::Lanczos3)
}

/// Deskew „podle textu“ – projekční profil.
/// 1) downscale, 2) jemný blur, 3) binarizace (ink=1), 4) sken úhlů a volba maxima variance.
fn estimate_skew_deg_text_projection(gray: &ImageBuffer<Luma<u8>, Vec<u8>>, max_deg: f32) -> f32 {
    let small = downscale_gray(gray, 1200);
    let blur = gaussian_blur_f32(&small, 0.8);
    let thr = otsu_level(&blur);
    // ink = černý text → 1, pozadí 0
    let mut bin = ImageBuffer::<Luma<u8>, Vec<u8>>::new(blur.width(), blur.height());
    for y in 0..blur.height() {
        for x in 0..blur.width() {
            let v = if blur.get_pixel(x, y)[0] < thr { 255 } else { 0 };
            bin.put_pixel(x, y, Luma([v]));
        }
    }

    // funkce: vodorovné projekce a její rozptyl
    let projection_score = |im: &ImageBuffer<Luma<u8>, Vec<u8>>| -> f64 {
        let h = im.height();
        if h == 0 { return 0.0; }
        let w = im.width();
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let n = h as f64;
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                if im.get_pixel(x, y)[0] > 0 { row += 1; }
            }
            let r = row as f64;
            sum += r;
            sum2 += r * r;
        }
        let mean = sum / n;
        let var = (sum2 / n) - mean * mean;
        var.max(0.0)
    };

    // sken úhlů
    let mut best_a = 0.0f32;
    let mut best_s = f64::MIN;

    let step_coarse = 0.5f32;
    let step_fine = 0.1f32;

    // hrubé hledání
    let mut a = -max_deg;
    while a <= max_deg {
        let rot = rotate_about_center(&bin, a.to_radians(), Interpolation::Nearest, Luma([0]));
        let s = projection_score(&rot);
        if s > best_s {
            best_s = s;
            best_a = a;
        }
        a += step_coarse;
    }

    // jemné hledání kolem maxima
    let a0 = (best_a - step_coarse).max(-max_deg);
    let a1 = (best_a + step_coarse).min(max_deg);
    let mut a = a0;
    best_s = f64::MIN;
    while a <= a1 {
        let rot = rotate_about_center(&bin, a.to_radians(), Interpolation::Nearest, Luma([0]));
        let s = projection_score(&rot);
        if s > best_s {
            best_s = s;
            best_a = a;
        }
        a += step_fine;
    }

    best_a
}

// ===== Morfologie =====
fn morphological_dilate3(src: &ImageBuffer<Luma<u8>, Vec<u8>>, iters: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let mut a = src.clone();
    for _ in 0..iters {
        let mut out = ImageBuffer::<Luma<u8>, Vec<u8>>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut on = false;
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        if a.get_pixel(xx, yy)[0] > 0 { on = true; break; }
                    }
                    if on { break; }
                }
                out.put_pixel(x, y, Luma([if on { 255 } else { 0 }]));
            }
        }
        a = out;
    }
    a
}
fn morphological_erode3(src: &ImageBuffer<Luma<u8>, Vec<u8>>, iters: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let mut a = src.clone();
    for _ in 0..iters {
        let mut out = ImageBuffer::<Luma<u8>, Vec<u8>>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut all_on = true;
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        if a.get_pixel(xx, yy)[0] == 0 { all_on = false; break; }
                    }
                    if !all_on { break; }
                }
                out.put_pixel(x, y, Luma([if all_on { 255 } else { 0 }]));
            }
        }
        a = out;
    }
    a
}
fn morphological_close3(src: &ImageBuffer<Luma<u8>, Vec<u8>>, iters: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let d = morphological_dilate3(src, iters);
    morphological_erode3(&d, iters)
}

// ===== Pomocné: Sobel magnituda (pro fallback) =====
fn sobel_magnitude(gray: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = gray.dimensions();
    let mut out = ImageBuffer::<Luma<u8>, Vec<u8>>::new(w, h);
    let clamp = |v: i32| -> u8 { v.unsigned_abs().min(255) as u8 };
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |xx: u32, yy: u32| gray.get_pixel(xx, yy)[0] as i32;
            let gx = -p(x - 1, y - 1) + p(x + 1, y - 1)
                - 2 * p(x - 1, y) + 2 * p(x + 1, y)
                - p(x - 1, y + 1) + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)
                + p(x - 1, y + 1) + 2 * p(x, y + 1) + p(x + 1, y + 1);
            let mag = ((gx * gx + gy * gy) as f64).sqrt() as i32;
            out.put_pixel(x, y, Luma([clamp(mag)]));
        }
    }
    out
}

// ===== Connected components: největší komponenta (4-sousedící) =====
fn largest_component_bbox(mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 { return None; }
    let mut visited = vec![false; (w as usize) * (h as usize)];
    let idx = |x: u32, y: u32| -> usize { (y as usize) * (w as usize) + (x as usize) };
    let mut best_area = 0u32;
    let mut best_bbox: Option<(u32, u32, u32, u32)> = None;
    let mut q = VecDeque::new();

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] == 0 { continue; }
            let id = idx(x, y);
            if visited[id] { continue; }
            visited[id] = true;
            q.clear();
            q.push_back((x, y));

            let (mut minx, mut miny, mut maxx, mut maxy) = (x, y, x, y);
            let mut area = 0u32;

            while let Some((cx, cy)) = q.pop_front() {
                area += 1;
                if cx < minx { minx = cx; }
                if cy < miny { miny = cy; }
                if cx > maxx { maxx = cx; }
                if cy > maxy { maxy = cy; }

                // 4-neighborhood
                let neigh = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];
                for (nx, ny) in neigh {
                    if nx < w && ny < h {
                        let nid = idx(nx, ny);
                        if !visited[nid] && mask.get_pixel(nx, ny)[0] > 0 {
                            visited[nid] = true;
                            q.push_back((nx, ny));
                        }
                    }
                }
            }

            if area > best_area {
                best_area = area;
                best_bbox = Some((minx, miny, maxx - minx + 1, maxy - miny + 1));
            }
        }
    }
    best_bbox
}

fn content_bbox(mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 { return None; }

    let mut minx = w;
    let mut miny = h;
    let mut maxx = 0;
    let mut maxy = 0;
    let mut found = false;

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] > 0 {
                found = true;
                if x < minx { minx = x; }
                if y < miny { miny = y; }
                if x > maxx { maxx = x; }
                if y > maxy { maxy = y; }
            }
        }
    }

    if found && maxx >= minx && maxy >= miny {
        Some((minx, miny, maxx - minx + 1, maxy - miny + 1))
    } else {
        None
    }
}

// ===== Detekce stránky: „světlý papír na tmavém pozadí“ + fallback =====

/// Hlavní smart detekce bboxu: napřed „světlý papír“, pak fallback „edge scan“.
fn detect_page_bbox_smart(gray: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<(u32, u32, u32, u32)> {
    detect_page_bbox_bright_on_dark(gray).or_else(|| detect_page_bbox_edges(gray))
}

/// Preferovaná cesta: papír je světlejší než pozadí (skener bez víka apod.)
fn detect_page_bbox_bright_on_dark(gray: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<(u32, u32, u32, u32)> {
    let (w0, h0) = gray.dimensions();
    if w0 < 8 || h0 < 8 { return None; }

    // downscale kvůli rychlosti/stabilitě
    let small = downscale_gray(gray, 1600);
    let (ws, hs) = small.dimensions();

    // jemné rozmazání → Otsu → „paper“ = světlé pixely
    let blur = gaussian_blur_f32(&small, 1.2);
    let t = otsu_level(&blur);
    let mut paper = ImageBuffer::<Luma<u8>, Vec<u8>>::new(ws, hs);
    for y in 0..hs {
        for x in 0..ws {
            let v = if blur.get_pixel(x, y)[0] >= t { 255 } else { 0 };
            paper.put_pixel(x, y, Luma([v]));
        }
    }
    // vyhladit → uzavřít díry v papíru (text)
    let paper = morphological_close3(&paper, 2);

    // největší komponenta = stránka
    if let Some((x, y, w, h)) = largest_component_bbox(&paper) {
        // přemapovat na původní rozlišení
        let sx = w0 as f32 / ws as f32;
        let sy = h0 as f32 / hs as f32;
        let x0 = ((x as f32) * sx).round() as u32;
        let y0 = ((y as f32) * sy).round() as u32;
        let x1 = (((x + w - 1) as f32) * sx).round() as u32;
        let y1 = (((y + h - 1) as f32) * sy).round() as u32;

        let x0 = x0.min(w0.saturating_sub(1));
        let y0 = y0.min(h0.saturating_sub(1));
        let x1 = x1.min(w0.saturating_sub(1));
        let y1 = y1.min(h0.saturating_sub(1));

        if x1 > x0 && y1 > y0 {
            return Some((x0, y0, x1 - x0 + 1, y1 - y0 + 1));
        }
    }
    None
}

/// Fallback: Sobel hrany + skenování od okrajů (robustní, když není jasná světlo/tma dominance).
fn detect_page_bbox_edges(gray: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = gray.dimensions();
    if w < 8 || h < 8 { return None; }

    // blur → sobel magnituda
    let g_blur = gaussian_blur_f32(gray, 1.0);
    let grad = sobel_magnitude(&g_blur);

    // thr + „dosesíťování“ hran
    let thr_g = otsu_level(&grad).saturating_add(5);
    let mut edges = ImageBuffer::<Luma<u8>, Vec<u8>>::new(w, h);
    for y in 0..h { for x in 0..w {
        edges.put_pixel(x, y, Luma([if grad.get_pixel(x, y)[0] >= thr_g { 255 } else { 0 }]));
    } }
    let mut edges = morphological_close3(&edges, 2);

    // posil okraje na tmavém pozadí
    let bg_thr = estimate_background(gray).saturating_sub(14);
    for y in 0..h {
        for x in 0..w {
            if gray.get_pixel(x, y)[0] < bg_thr {
                let px = edges.get_pixel_mut(x, y);
                px.0[0] = 255;
            }
        }
    }

    // sčítání hran v řádcích/sloupcích
    let mut row_counts = vec![0u32; h as usize];
    let mut col_counts = vec![0u32; w as usize];
    for y in 0..h {
        let mut s = 0u32;
        for x in 0..w { if edges.get_pixel(x, y)[0] > 0 { s += 1; } }
        row_counts[y as usize] = s;
    }
    for x in 0..w {
        let mut s = 0u32;
        for y in 0..h { if edges.get_pixel(x, y)[0] > 0 { s += 1; } }
        col_counts[x as usize] = s;
    }
    let row_max = *row_counts.iter().max().unwrap_or(&0);
    let col_max = *col_counts.iter().max().unwrap_or(&0);
    if row_max == 0 || col_max == 0 { return None; }

    // prahy relativní k maximu
    let row_min = (row_max as f32 * 0.25).max(8.0) as u32;
    let col_min = (col_max as f32 * 0.25).max(8.0) as u32;

    // top / bottom
    let mut top = None;
    for y in 0..h / 2 {
        if row_counts[y as usize] >= row_min { top = Some(y); break; }
    }
    let mut bottom = None;
    let mut yy = h as i64 - 1;
    while yy >= (h as i64) / 2 {
        let y = yy as u32;
        if row_counts[y as usize] >= row_min { bottom = Some(y); break; }
        yy -= 1;
    }
    // left / right
    let mut left = None;
    for x in 0..w / 2 {
        if col_counts[x as usize] >= col_min { left = Some(x); break; }
    }
    let mut right = None;
    let mut xx = w as i64 - 1;
    while xx >= (w as i64) / 2 {
        let x = xx as u32;
        if col_counts[x as usize] >= col_min { right = Some(x); break; }
        xx -= 1;
    }

    // fallbacky: zkuste největší komponentu po uzavření
    if top.is_none() || bottom.is_none() || left.is_none() || right.is_none() {
        let merged = morphological_close3(&edges, 2);
        return largest_component_bbox(&merged);
    }

    let (mut left, mut right, mut top, mut bottom) =
        (left.unwrap(), right.unwrap(), top.unwrap(), bottom.unwrap());

    if right <= left || bottom <= top {
        let merged = morphological_close3(&edges, 2);
        return largest_component_bbox(&merged);
    }

    // malé roztažení, aby se neukrojil papír
    left = left.saturating_sub(1);
    top = top.saturating_sub(1);
    right = (right + 1).min(w - 1);
    bottom = (bottom + 1).min(h - 1);

    Some((left, top, right - left + 1, bottom - top + 1))
}

fn estimate_background(gray: &ImageBuffer<Luma<u8>, Vec<u8>>) -> u8 {
    let (w, h) = gray.dimensions();
    let bw = (w as f32 * 0.02).max(1.0) as u32;
    let bh = (h as f32 * 0.02).max(1.0) as u32;
    let mut vals: Vec<u8> = Vec::with_capacity(((w * bh) + (h * bw) * 2) as usize);
    for y in 0..bh { for x in 0..w { vals.push(gray.get_pixel(x, y)[0]); } }
    for y in h - bh..h { for x in 0..w { vals.push(gray.get_pixel(x, y)[0]); } }
    for x in 0..bw { for y in 0..h { vals.push(gray.get_pixel(x, y)[0]); } }
    for x in w - bw..w { for y in 0..h { vals.push(gray.get_pixel(x, y)[0]); } }
    if vals.is_empty() { return 255; }
    vals.sort_unstable();
    vals[vals.len() / 2]
}

// Pomocník: auto crop bez borrowu na `self`
fn auto_crop_rect_for_with_params(path: &Path, params: &Params) -> Option<egui::Rect> {
    let img = image::open(path).ok()?;
    let (rot_rgba, _thr) = make_rotated_rgba(&img, params);
    let gray = DynamicImage::ImageRgba8(rot_rgba.clone()).to_luma8();
    detect_page_bbox_smart(&gray).map(|(x, y, w, h)| {
        egui::Rect::from_min_size(egui::pos2(x as f32, y as f32), egui::vec2(w as f32, h as f32))
    })
}

/// Heuristic splitter for two-page scans: finds a likely gutter position near the middle.
/// Returns X in normalized coordinates (0..1) of the rotated image.
fn detect_split_x_norm(path: &Path, params: &Params) -> Option<f32> {
    let img = image::open(path).ok()?;
    let (rot_rgba, _thr) = make_rotated_rgba(&img, params);
    let (w, h) = rot_rgba.dimensions();
    if w < 64 || h < 64 {
        return Some(0.5);
    }

    // Downsample by stepping pixels – fast and good enough.
    let step_x: u32 = (w / 800).max(1);
    let step_y: u32 = (h / 800).max(1);

    // Search only around the center.
    let x0 = (w as f32 * 0.30) as u32;
    let x1 = (w as f32 * 0.70) as u32;
    if x1 <= x0 + 4 {
        return Some(0.5);
    }

    // Score each candidate column by average brightness (gutter tends to be bright/low ink).
    // We take the MAX brightness.
    let mut best_x = (w / 2).max(1);
    let mut best_score: f32 = -1.0;

    for x in (x0..=x1).step_by(step_x as usize) {
        let mut sum: u64 = 0;
        let mut cnt: u64 = 0;
        // Ignore top/bottom margins a bit.
        let y_start = (h as f32 * 0.08) as u32;
        let y_end   = (h as f32 * 0.92) as u32;
        for y in (y_start..=y_end).step_by(step_y as usize) {
            let p = rot_rgba.get_pixel(x.min(w - 1), y.min(h - 1));
            // integer luma approx (BT.709-ish)
            let r = p[0] as u32;
            let g = p[1] as u32;
            let b = p[2] as u32;
            let l = (2126 * r + 7152 * g + 722 * b) / 10000;
            sum += l as u64;
            cnt += 1;
        }
        if cnt == 0 { continue; }
        let avg = (sum as f32) / (cnt as f32);
        if avg > best_score {
            best_score = avg;
            best_x = x;
        }
    }

    Some((best_x as f32 / (w as f32)).clamp(0.05, 0.95))
}

// ===== Uložení s metadaty, správná bitová hloubka =====
fn rgba8_is_opaque(img: &RgbaImage) -> bool {
    img.pixels().all(|p| p[3] == 255)
}

fn add_margin_rgba(img: &RgbaImage, margin: u32, color: Rgba<u8>) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut out = RgbaImage::from_pixel(w + 2 * margin, h + 2 * margin, color);
    image::imageops::replace(&mut out, img, margin as i64, margin as i64);
    out
}
fn add_margin_rgba_from_original(
    rotated_uncropped: &RgbaImage,
    crop_xywh: (u32, u32, u32, u32),
    margin_px: u32,
) -> RgbaImage {
    let (x, y, w, h) = crop_xywh;
    let (rw, rh) = rotated_uncropped.dimensions();
    let m = margin_px;
    let x0 = x.saturating_sub(m);
    let y0 = y.saturating_sub(m);
    let x1 = (x + w).saturating_add(m).min(rw);
    let y1 = (y + h).saturating_add(m).min(rh);
    let w2 = x1.saturating_sub(x0).max(1);
    let h2 = y1.saturating_sub(y0).max(1);
    image::imageops::crop_imm(rotated_uncropped, x0, y0, w2, h2).to_image()
}


#[derive(Default, Debug, Clone)]
struct TiffMeta {
    icc_profile: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
}

fn extract_tiff_meta(path: &Path) -> Option<TiffMeta> {
    use std::fs::File;
    use std::io::BufReader;
    use tiff::decoder::Decoder;
    use tiff::tags::Tag;

    let f = File::open(path).ok()?;
    let mut dec = Decoder::new(BufReader::new(f)).ok()?;

    let mut meta = TiffMeta::default();

    // ICC profile: TIFF tag 34675
    if let Ok(Some(v)) = dec.find_tag(Tag::IccProfile) {
        if let Ok(bytes) = v.into_u8_vec() {
            if !bytes.is_empty() {
                meta.icc_profile = Some(bytes);
            }
        }
    }

    // XMP: TIFF tag 700 (Adobe XMP packet)
    // Older tiff crate versions may not have a dedicated Tag variant, Unknown(700) works.
    if let Ok(Some(v)) = dec.find_tag(Tag::Unknown(700)) {
        if let Ok(bytes) = v.into_u8_vec() {
            if !bytes.is_empty() {
                meta.xmp = Some(bytes);
            }
        }
    }

    if meta.icc_profile.is_some() || meta.xmp.is_some() {
        Some(meta)
    } else {
        None
    }
}

fn write_tiff_meta_tags<'a, W, C, K>(
    imgw: &mut tiff::encoder::ImageEncoder<'a, W, C, K>,
    meta: Option<&TiffMeta>,
) where
    W: std::io::Write + std::io::Seek,
    // In tiff crate, ColorType is available via the colortype module.
    C: tiff::encoder::colortype::ColorType,
    K: tiff::encoder::TiffKind,
{
    use tiff::tags::Tag;

    let Some(meta) = meta else { return; };

    // In tiff >= 0.11, custom tags are written via the underlying DirectoryEncoder.
    if let Some(ref icc) = meta.icc_profile {
        let _ = imgw.encoder().write_tag(Tag::IccProfile, icc.as_slice());
    }
    if let Some(ref xmp) = meta.xmp {
        let _ = imgw.encoder().write_tag(Tag::Unknown(700), xmp.as_slice());
    }
}

fn save_image_with_metadata(
    img: &DynamicImage,
    path: &Path,
    params: &Params,
    src_path: &Path,
    src_bytes: Option<&[u8]>,
) -> Result<()> {
    match params.out_format {
        OutputFormat::Jpeg => {
            use image::codecs::jpeg::{JpegEncoder, PixelDensity};
            use image::{ColorType, ImageEncoder};

            let mut buf = Vec::<u8>::new();
            let mut enc = JpegEncoder::new_with_quality(&mut buf, params.jpeg_quality);
            if let Some(dpi) = params.dpi { enc.set_pixel_density(PixelDensity::dpi(dpi as u16)); }

            match img {
                DynamicImage::ImageLuma8(y) => enc.write_image(y.as_raw(), y.width(), y.height(), ColorType::L8)?,
                DynamicImage::ImageRgb8(rgb) => enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8)?,
                _ => {
                    let rgb = img.to_rgb8();
                    enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8)?;
                }
            }

            if !params.strip_metadata {
                use bytes::Bytes;
                use img_parts::{ImageEXIF, ImageICC};
                let mut out_jpeg = img_parts::jpeg::Jpeg::from_bytes(Bytes::copy_from_slice(&buf))?;
                if let Some(src) = src_bytes {
                    if let Ok(src_jpeg) = img_parts::jpeg::Jpeg::from_bytes(Bytes::copy_from_slice(src)) {
                        if let Some(exif) = src_jpeg.exif() { out_jpeg.set_exif(Some(exif)); }
                        if let Some(icc)  = src_jpeg.icc_profile() { out_jpeg.set_icc_profile(Some(icc)); }
                    }
                }
                let mut out = Vec::new();
                out_jpeg.encoder().write_to(&mut out)?;
                fs::write(path, out)?;
            } else {
                fs::write(path, &buf)?;
            }
        }
        OutputFormat::Png => {
            use image::codecs::png::PngEncoder;
            use image::{ColorType, ImageEncoder};

            let mut buf = Vec::<u8>::new();
            let enc = PngEncoder::new(&mut buf);

            match img {
                DynamicImage::ImageLuma8(y) => enc.write_image(y.as_raw(), y.width(), y.height(), ColorType::L8)?,
                DynamicImage::ImageLumaA8(ya) => {
                    let all_opaque = ya.pixels().all(|p| p.0[1] == 255);
                    if all_opaque {
                        let y = DynamicImage::ImageLumaA8(ya.clone()).to_luma8();
                        enc.write_image(y.as_raw(), y.width(), y.height(), ColorType::L8)?;
                    } else {
                        enc.write_image(ya.as_raw(), ya.width(), ya.height(), ColorType::La8)?;
                    }
                }
                DynamicImage::ImageRgb8(rgb) => enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8)?,
                DynamicImage::ImageRgba8(rgba) => {
                    if rgba8_is_opaque(rgba) {
                        let rgb = DynamicImage::ImageRgba8(rgba.clone()).to_rgb8();
                        enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8)?;
                    } else {
                        enc.write_image(rgba.as_raw(), rgba.width(), rgba.height(), ColorType::Rgba8)?;
                    }
                }
                DynamicImage::ImageLuma16(y16) => {
                    let v = y16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                    for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                    enc.write_image(&out, y16.width(), y16.height(), ColorType::L16)?;
                }
                DynamicImage::ImageLumaA16(ya16) => {
                    let all_opaque = ya16.pixels().all(|p| p.0[1] == u16::MAX);
                    if all_opaque {
                        let y16 = DynamicImage::ImageLumaA16(ya16.clone()).to_luma16();
                        let v = y16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                        for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                        enc.write_image(&out, y16.width(), y16.height(), ColorType::L16)?;
                    } else {
                        let v = ya16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                        for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                        enc.write_image(&out, ya16.width(), ya16.height(), ColorType::La16)?;
                    }
                }
                DynamicImage::ImageRgb16(rgb16) => {
                    let v = rgb16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                    for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                    enc.write_image(&out, rgb16.width(), rgb16.height(), ColorType::Rgb16)?;
                }
                DynamicImage::ImageRgba16(rgba16) => {
                    let all_opaque = rgba16.pixels().all(|p| p.0[3] == u16::MAX);
                    if all_opaque {
                        let rgb16 = DynamicImage::ImageRgba16(rgba16.clone()).to_rgb16();
                        let v = rgb16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                        for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                        enc.write_image(&out, rgb16.width(), rgb16.height(), ColorType::Rgb16)?;
                    } else {
                        let v = rgba16.as_raw(); let mut out = Vec::with_capacity(v.len() * 2);
                        for &x in v { out.extend_from_slice(&x.to_be_bytes()); }
                        enc.write_image(&out, rgba16.width(), rgba16.height(), ColorType::Rgba16)?;
                    }
                }
                _ => {
                    let rgba = img.to_rgba8();
                    if rgba8_is_opaque(&rgba) {
                        let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
                        enc.write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8)?;
                    } else {
                        let rgba = img.to_rgba8();
                        enc.write_image(rgba.as_raw(), rgba.width(), rgba.height(), ColorType::Rgba8)?;
                    }
                }
            }

            if !params.strip_metadata {
                use bytes::Bytes;
                use img_parts::{ImageEXIF, ImageICC};
                if let Ok(mut out_png) = img_parts::png::Png::from_bytes(Bytes::copy_from_slice(&buf)) {
                    if let Some(src) = src_bytes {
                        if let Ok(src_png) = img_parts::png::Png::from_bytes(Bytes::copy_from_slice(src)) {
                            if let Some(exif) = src_png.exif() { out_png.set_exif(Some(exif)); }
                            if let Some(icc)  = src_png.icc_profile() { out_png.set_icc_profile(Some(icc)); }
                        }
                    }
                    let mut out = Vec::new();
                    out_png.encoder().write_to(&mut out)?;
                    fs::write(path, out)?;
                } else {
                    fs::write(path, &buf)?;
                }
            } else {
                fs::write(path, &buf)?;
            }
        }
        OutputFormat::Tiff => {
            use std::fs::File;
            use std::io::BufWriter;
            use tiff::encoder::{colortype, Rational, TiffEncoder};
            use tiff::tags::ResolutionUnit;

            let tiff_meta = if !params.strip_metadata {
                extract_tiff_meta(src_path)
            } else {
                None
            };

            let file = File::create(path)?;
            let mut w = BufWriter::new(file);
            let mut enc = TiffEncoder::new(&mut w)?;

            match img {
                DynamicImage::ImageLuma8(y) => {
                    let mut imgw = enc.new_image::<colortype::Gray8>(y.width(), y.height())?;
                    if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                    write_tiff_meta_tags(&mut imgw, tiff_meta.as_ref());
                    imgw.write_data(y.as_raw())?;
                }
                DynamicImage::ImageLumaA8(ya) => {
                    let all_opaque = ya.pixels().all(|p| p.0[1] == 255);
                    if all_opaque {
                        let y = DynamicImage::ImageLumaA8(ya.clone()).to_luma8();
                        let mut imgw = enc.new_image::<colortype::Gray8>(y.width(), y.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(y.as_raw())?;
                    } else {
                        let rgba = DynamicImage::ImageLumaA8(ya.clone()).to_rgba8();
                        let mut imgw = enc.new_image::<colortype::RGBA8>(rgba.width(), rgba.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(rgba.as_raw())?;
                    }
                }
                DynamicImage::ImageRgb8(rgb) => {
                    let mut imgw = enc.new_image::<colortype::RGB8>(rgb.width(), rgb.height())?;
                    if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                    write_tiff_meta_tags(&mut imgw, tiff_meta.as_ref());
                    imgw.write_data(rgb.as_raw())?;
                }
                DynamicImage::ImageRgba8(rgba) => {
                    if rgba8_is_opaque(rgba) {
                        let rgb = DynamicImage::ImageRgba8(rgba.clone()).to_rgb8();
                        let mut imgw = enc.new_image::<colortype::RGB8>(rgb.width(), rgb.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(rgb.as_raw())?;
                    } else {
                        let mut imgw = enc.new_image::<colortype::RGBA8>(rgba.width(), rgba.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(rgba.as_raw())?;
                    }
                }
                DynamicImage::ImageLuma16(y16) => {
                    let mut imgw = enc.new_image::<colortype::Gray16>(y16.width(), y16.height())?;
                    if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                    write_tiff_meta_tags(&mut imgw, tiff_meta.as_ref());
                    imgw.write_data(y16.as_raw())?;
                }
                DynamicImage::ImageRgb16(rgb16) => {
                    let mut imgw = enc.new_image::<colortype::RGB16>(rgb16.width(), rgb16.height())?;
                    if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                    write_tiff_meta_tags(&mut imgw, tiff_meta.as_ref());
                    imgw.write_data(rgb16.as_raw())?;
                }
                DynamicImage::ImageRgba16(rgba16) => {
                    let all_opaque = rgba16.pixels().all(|p| p.0[3] == u16::MAX);
                    if all_opaque {
                        let rgb16 = DynamicImage::ImageRgba16(rgba16.clone()).to_rgb16();
                        let mut imgw = enc.new_image::<colortype::RGB16>(rgb16.width(), rgb16.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(rgb16.as_raw())?;
                    } else {
                        let mut imgw = enc.new_image::<colortype::RGBA16>(rgba16.width(), rgba16.height())?;
                        if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                        imgw.write_data(rgba16.as_raw())?;
                    }
                }
                _ => {
                    let rgb = img.to_rgb8();
                    let mut imgw = enc.new_image::<colortype::RGB8>(rgb.width(), rgb.height())?;
                    if let Some(dpi) = params.dpi { imgw.resolution(ResolutionUnit::Inch, Rational { n: dpi as u32, d: 1 }); }
                    write_tiff_meta_tags(&mut imgw, tiff_meta.as_ref());
                    imgw.write_data(rgb.as_raw())?;
                }
            }
        }
    }
    Ok(())
}

// ===== Pomocné: aplikace masky přejmenování =====
fn apply_rename_mask(mask: &str, index: usize) -> String {
    // povolené znaky: alnum, '-', '_', '#'
    let filtered: String = mask
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '#')
        .collect();

    let mut out = String::new();
    let chars: Vec<char> = filtered.chars().collect();
    let mut i = 0usize;
    let mut replaced_any = false;

    while i < chars.len() {
        if chars[i] == '#' {
            // spočítej délku běhu '#'
            let mut j = i;
            while j < chars.len() && chars[j] == '#' {
                j += 1;
            }
            let width = j - i;
            out.push_str(&format!("{:0width$}", index, width = width));
            replaced_any = true;
            i = j;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }

    if !replaced_any {
        out.push_str(&index.to_string());
    }
    out
}

fn unique_out_path(dir: &Path, base: &str, ext: &str) -> PathBuf {
    let candidate = dir.join(format!("{}.{}", base, ext));
    if !candidate.exists() {
        return candidate;
    }
    let mut k: usize = 1;
    loop {
        let c = dir.join(format!("{}_{}.{}", base, k, ext));
        if !c.exists() {
            return c;
        }
        k += 1;
    }
}