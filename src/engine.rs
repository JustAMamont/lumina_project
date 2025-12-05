use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyDict};
use std::thread::{self, JoinHandle};
use std::collections::HashMap;
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use crossbeam_channel::{bounded, Sender, Receiver, RecvTimeoutError, TrySendError};
use std::sync::Arc;
use ahash::AHasher; 
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::io::{stdout, Write, Seek, BufWriter, Read, SeekFrom, BufReader}; 
use std::env; 
use std::path::{Path, PathBuf}; 
use std::fs::{self, OpenOptions, File}; 
use colored::*;
use sysinfo::{Pid, System}; 
use indexmap::IndexMap;
use rustc_hash::FxHasher;
use bincode::Options;
use byteorder::{LittleEndian, ByteOrder, WriteBytesExt};
use crc32fast::Hasher as Crc32Hasher;
use fs2::FileExt;
use chrono::DateTime; 

#[cfg(unix)]
use libc::{getrusage, rusage, RUSAGE_SELF};

use crate::types::{LogEntry, TraceFrame, CallerCache, Theme, WriterMsg, BinaryLogRecord, RawTraceback};
use crate::utils::{get_time_parts, fast_colorize, get_level_meta, 
    set_theme, measure_text_height, set_colors_enabled, get_current_thread_hash, sanitize_input, remove_style_tags};

// === CONSTANTS ===
const MAGIC_HEADER: &[u8; 4] = b"LUM1";
const BINARY_BATCH_CAPACITY: usize = 1000;
const TEXT_FILE_BUFFER_SIZE: usize = 256 * 1024;
const BINARY_FILE_BUFFER_SIZE: usize = 256 * 1024;

// === HELPER FUNCTIONS ===

/// Navigates the Python stack to find the actual caller file and line number.
fn find_caller_in_rust(py: Python, skip_frames: usize) -> Option<(String, u32)> {
    let sys = PyModule::import_bound(py, "sys").ok()?;
    let getframe = sys.getattr("_getframe").ok()?;
    
    let mut frame = getframe.call1((skip_frames,)).ok();
    
    let skip_markers = [
        "lumina/__init__.py",
        "logging/__init__.py",
        "asyncio/",
        "concurrent/futures",
        "/usr/lib/python", 
        "\\Lib\\",
    ];

    let mut last_valid_frame_info: Option<(String, u32)> = None;

    while let Some(f) = frame {
        if let Ok(code) = f.getattr("f_code") {
            if let Ok(filename_py) = code.getattr("co_filename") {
                let filename: String = filename_py.extract().unwrap_or_default();
                let mut is_skipped = false;
                for marker in &skip_markers {
                    if filename.contains(marker) { is_skipped = true; break; }
                }

                let path = Path::new(&filename);
                let cwd = env::current_dir().unwrap_or_default();
                let final_path = path.strip_prefix(&cwd).unwrap_or(path).to_string_lossy().to_string();
                let lineno: u32 = f.getattr("f_lineno").ok().and_then(|l| l.extract().ok()).unwrap_or(0);

                if !is_skipped { return Some((final_path, lineno)); }
                last_valid_frame_info = Some((final_path, lineno));
            }
        }
        frame = f.getattr("f_back").ok();
        if frame.is_none() { break; }
        if frame.as_ref().unwrap().is_none() { break; } 
    }
    last_valid_frame_info
}

fn format_ram_delta(delta: i64) -> String {
    let sign = if delta >= 0 { "+" } else { "-" };
    let val = delta.abs() as f64;
    if val > 1024.0 * 1024.0 { format!("{}{:.2}MB", sign, val / 1024.0 / 1024.0) } 
    else if val > 1024.0 { format!("{}{:.2}KB", sign, val / 1024.0) } 
    else { format!("{}{:.0}B", sign, val) }
}

fn format_ram_abs(val: u64) -> String {
    format!("{:.1}MB", val as f64 / 1024.0 / 1024.0)
}

fn cleanup_old_logs(path_template: String, retention_days: u64) {
    if retention_days == 0 { return; }
    let path = Path::new(&path_template);
    let log_dir = match path.parent() { Some(p) if p.as_os_str().is_empty() => Path::new("."), Some(p) => p, None => return };
    if !log_dir.exists() { return; }

    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if let Ok(entries) = fs::read_dir(log_dir) {
        let now = SystemTime::now();
        let retention_duration = Duration::from_secs(retention_days * 24 * 60 * 60);
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == extension || (extension.is_empty()) {
                        if let Ok(metadata) = fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                if let Ok(age) = now.duration_since(modified) {
                                    if age > retention_duration { let _ = fs::remove_file(&path); }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


static CONSOLE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn update_console(
    entry: &LogEntry, repeat: usize, is_update: bool, first_ts: f64, 
    time_prefix: &str, time_nsecs: u32, last_height: &mut usize
) {
    if !entry.to_console { return; }
    
    let _guard = CONSOLE_LOCK.lock();

    let (_, lvl_color, icon) = get_level_meta(entry.level);
    let safe_msg_raw = sanitize_input(&entry.message);
    let colored_msg = fast_colorize(&safe_msg_raw);
    
    let is_light = { if let Ok(r) = crate::utils::CURRENT_THEME.read() { *r == Theme::Light } else { false } };

    let context_str = if let Some(ctx) = &entry.context {
        let mut s = String::new();
        for (k, v) in ctx {
            let safe_v = sanitize_input(v);
            if is_light { s.push_str(&format!(" {}={}", k.black().dimmed(), safe_v.blue())); } 
            else { s.push_str(&format!(" {}={}", k.dimmed(), safe_v.cyan())); }
        }
        s
    } else { String::new() };

    let suffix = if repeat > 0 {
        let diff = entry.timestamp - first_ts;
        let time_fmt = if diff < 1.0 { format!("{:.0}ms", diff * 1000.0) } else { format!("{:.1}s", diff) };
        if is_light { format!(" (x{} │ {})", repeat + 1, time_fmt).black().dimmed().to_string() } 
        else { format!(" (x{} │ {})", repeat + 1, time_fmt).yellow().dimmed().to_string() }
    } else { String::new() };

    let time_display = if is_light { format!("{}.{:03}", time_prefix, time_nsecs).black().dimmed() } 
    else { format!("{}.{:03}", time_prefix, time_nsecs).cyan().dimmed() };
    
    let header = format!("{} {} {: <9} │ {}{}{}", time_display, icon, lvl_color, colored_msg, context_str, suffix);
    let mut stdout = stdout();
    
    let mut current_height = measure_text_height(&header);

    if is_update {
        if *last_height > 0 {
            let move_up = format!("\x1b[{}A", *last_height);
            let _ = write!(stdout, "{}", move_up);
        } else { let _ = write!(stdout, "\x1b[1A"); }
        let _ = write!(stdout, "\x1b[J");
    }
    let _ = writeln!(stdout, "{}", header);

    if entry.exc_type.is_some() {
        let mut tb_str = String::new();
        if let Some(frames) = &entry.trace_frames {
            let header = if is_light { "Traceback (most recent call last):".magenta().to_string() } 
                         else { "Traceback (most recent call last):".yellow().dimmed().to_string() };
            tb_str.push_str(&format!("{}\n", header));
            for frame in frames.iter().take(5) {
                 let fname = if is_light { frame.filename.blue().to_string() } else { frame.filename.blue().underline().to_string() };
                 let line = if is_light { frame.lineno.to_string().purple().to_string() } else { frame.lineno.to_string().yellow().to_string() };
                 let func = if is_light { frame.name.black().bold().to_string() } else { frame.name.cyan().to_string() };
                 tb_str.push_str(&format!("  File \"{}\", line {}, in {}\n", fname, line, func));
            }
            if frames.len() > 5 { tb_str.push_str(&format!("  ... ({} more frames)\n", frames.len() - 5)); }
        }
        let _ = write!(stdout, "{}", tb_str);
        current_height += measure_text_height(&tb_str);
        
        if let (Some(t), Some(m)) = (&entry.exc_type, &entry.exc_message) {
            let err_line = format!("{}: {}", t.red().bold(), m);
            let _ = writeln!(stdout, "{}", err_line);
            let _ = writeln!(stdout, "");
            current_height += measure_text_height(&err_line) + 1;
        }
    }
    let _ = stdout.flush();
    *last_height = current_height;
}

// === ENGINE ===

#[pyclass]
pub struct LuminaEngine {
    // Sharded input channels. Python threads pick one based on their ThreadID.
    input_shards: Vec<Sender<LogEntry>>,
    
    // Global writer channel. All workers send formatted blobs here.
    writer_tx: Sender<WriterMsg>,
    
    // Handles to join threads on shutdown
    worker_threads: Arc<Mutex<Vec<JoinHandle<()>>>>,
    writer_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    
    last_push_exc_hash: Arc<AtomicU64>,
    dropped_logs_count: Arc<AtomicUsize>,
    caller_cache: Arc<Mutex<CallerCache>>,
    system_monitor: Arc<Mutex<System>>,
    
    app_name: Arc<str>,
    capture_caller: bool,
    path_template: String,
}

#[pymethods]
impl LuminaEngine {
    #[new]
    fn new(
        app_name: String,
        cleanup_path_template: String,
        channel_capacity: usize, 
        _file_buffer_size: usize, // No longer used directly, writer has its own const
        console_refresh_rate: u128, 
        flush_interval_ms: u64, 
        text_enabled: bool, 
        db_enabled: bool,
        capture_caller: bool,
        retention_days: u64,
        colors_enabled: bool,
        theme: Option<String>,
    ) -> Self {
        #[cfg(windows)] let _ = colored::control::set_virtual_terminal(true);
        let theme_enum = Theme::from_str(&theme.unwrap_or_else(|| "dark".to_string()));
        set_theme(theme_enum);
        set_colors_enabled(colors_enabled);

        if retention_days > 0 {
            let tmpl = cleanup_path_template.clone();
            thread::spawn(move || { cleanup_old_logs(tmpl, retention_days); });
        }
        
        // Use at least 2 workers, but not more than 8 or num_cpus
        let num_shards = num_cpus::get().min(8).max(2); 
        let mut input_txs = Vec::with_capacity(num_shards);
        let mut input_rxs = Vec::with_capacity(num_shards);
        
        for _ in 0..num_shards {
            let shard_capacity = (channel_capacity as f64 / num_shards as f64).ceil() as usize;
            let (tx, rx) = bounded::<LogEntry>(shard_capacity.max(1000));
            input_txs.push(tx);
            input_rxs.push(rx);
        }

        let (writer_tx, writer_rx) = bounded::<WriterMsg>(num_shards * 4);
        let dropped_logs_count = Arc::new(AtomicUsize::new(0));
        let system_monitor = Arc::new(Mutex::new(System::new()));
        
        let mut handles = Vec::new();
        
        for i in 0..num_shards {
            let rx = input_rxs.remove(0); // Take ownership
            let w_tx = writer_tx.clone();
            let dropped_c = dropped_logs_count.clone();
            
            let handle = thread::Builder::new()
                .name(format!("lumina-worker-{}", i))
                .spawn(move || {
                    run_worker(
                        rx, w_tx, 
                        dropped_c, 
                        text_enabled, db_enabled, 
                        flush_interval_ms, console_refresh_rate
                    );
                }).expect("Failed to spawn worker");
            handles.push(handle);
        }

        let writer_handle = thread::Builder::new()
            .name("lumina-writer".to_string())
            .spawn(move || {
                run_writer(writer_rx);
            }).expect("Failed to spawn writer");

        // Send start signal to the first shard
        let start_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        let app_name_arc: Arc<str> = app_name.clone().into();
        let start_msg = LogEntry {
            app_name: app_name_arc.clone(), timestamp: start_ts,
            level: 20, message: "🚀 Process started".to_string(), path_template: cleanup_path_template.clone(),
            to_console: false, to_file: true, rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, 
            trace_frames: None, context: None, context_hash: 0, signal_shutdown: false
        };
        let _ = input_txs[0].try_send(start_msg);

        LuminaEngine { 
            input_shards: input_txs,
            writer_tx,
            worker_threads: Arc::new(Mutex::new(handles)), 
            writer_thread: Arc::new(Mutex::new(Some(writer_handle))),
            last_push_exc_hash: Arc::new(AtomicU64::new(0)), 
            dropped_logs_count, 
            caller_cache: Arc::new(Mutex::new(HashMap::new())), 
            system_monitor, 
            app_name: app_name_arc, 
            capture_caller,
            path_template: cleanup_path_template 
        }
    }

    fn snapshot_resources(&self) -> (f64, u64, u64, u64, u64, u64) {
        #[cfg(unix)]
        {
            let mut usage = std::mem::MaybeUninit::<rusage>::uninit();
            if unsafe { getrusage(RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
                let usage = unsafe { usage.assume_init() };
                #[cfg(target_os = "linux")]
                let ram_peak = (usage.ru_maxrss as u64) * 1024;
                #[cfg(not(target_os = "linux"))]
                let ram_peak = usage.ru_maxrss as u64;
                let cpu_time_us = (usage.ru_utime.tv_sec as f64 * 1_000_000.0) + (usage.ru_utime.tv_usec as f64) +
                                  (usage.ru_stime.tv_sec as f64 * 1_000_000.0) + (usage.ru_stime.tv_usec as f64);
                let mut sys = self.system_monitor.lock();
                let pid = Pid::from_u32(std::process::id());
                sys.refresh_process(pid);
                let ram_current = if let Some(process) = sys.process(pid) { process.memory() } else { 0 };
                return (cpu_time_us, ram_current, ram_peak, usage.ru_nvcsw as u64, usage.ru_nivcsw as u64, (usage.ru_majflt + usage.ru_minflt) as u64);
            }
        }
        let mut sys = self.system_monitor.lock();
        let pid = Pid::from_u32(std::process::id());
        sys.refresh_process(pid);
        if let Some(process) = sys.process(pid) { return (process.cpu_usage() as f64, process.memory(), 0, 0, 0, 0); }
        (0.0, 0, 0, 0, 0, 0)
    }
    
    #[pyo3(signature = (name, start_data, end_data, t_start, t_end, min_duration_ms, slow_threshold_ms, console=true, tags=None))]
    fn push_profile(&self, py: Python<'_>, name: String, 
                    start_data: (f64, u64, u64, u64, u64, u64), 
                    end_data: (f64, u64, u64, u64, u64, u64), 
                    t_start: f64, t_end: f64, 
                    min_duration_ms: Option<f64>, 
                    slow_threshold_ms: Option<f64>,
                    console: bool,
                    tags: Option<&Bound<'_, PyDict>>) 
    {
        let duration_sec = t_end - t_start;
        let duration_ms = duration_sec * 1000.0;

        if let Some(min_ms) = min_duration_ms {
            if duration_ms < min_ms {
                return;
            }
        }

        let mut level = 20;
        let mut prefix = "";
        if let Some(slow_ms) = slow_threshold_ms {
            if duration_ms > slow_ms {
                level = 30;
                prefix = "🐌 ";
            }
        }

        let (cpu_start, ram_start, _, ctx_v_start, ctx_iv_start, pg_start) = start_data;
        let (cpu_end, ram_end, peak_end, ctx_v_end, ctx_iv_end, pg_end) = end_data;

        let is_unix_rusage = cpu_start > 1000.0 || cpu_end > 1000.0;
        let cpu_str = if is_unix_rusage {
            let cpu_delta_us = cpu_end - cpu_start;
            if duration_sec < 0.1 {
                format!("{:.0}µs", cpu_delta_us)
            } else {
                let usage = (cpu_delta_us / (duration_sec * 1_000_000.0)) * 100.0;
                format!("{:.1}%", usage)
            }
        } else {
            format!("{:.1}%", cpu_end)
        };

        let ram_delta = (ram_end as i64) - (ram_start as i64);
        let ram_str = format_ram_delta(ram_delta);
        let ram_tot_str = format_ram_abs(ram_end);

        let mut context = IndexMap::new();
        
        context.insert("time".to_string(), format!("{:.4}s", duration_sec));
        context.insert("ram".to_string(), ram_str);
        context.insert("ram_tot".to_string(), ram_tot_str);
        context.insert("cpu".to_string(), cpu_str);

        if peak_end > 0 && peak_end > ram_end {
            context.insert("ram_peak".to_string(), format_ram_abs(peak_end));
        }

        let pg_delta = pg_end.saturating_sub(pg_start);
        if pg_delta > 0 {
            context.insert("pg_faults".to_string(), pg_delta.to_string());
        }

        let ctx_v_delta = ctx_v_end.saturating_sub(ctx_v_start);
        let ctx_iv_delta = ctx_iv_end.saturating_sub(ctx_iv_start);

        if ctx_v_delta > 0 { context.insert("ctx_wait".to_string(), ctx_v_delta.to_string()); }
        if ctx_iv_delta > 0 { context.insert("ctx_forced".to_string(), ctx_iv_delta.to_string()); }

        if let Some(py_tags) = tags {
            for (k, v) in py_tags {
                context.insert(k.to_string(), v.to_string());
            }
        }

        let mut context_hash: u64 = 0;
        for (k, v) in &context {
            let mut entry_hasher = FxHasher::default();
            entry_hasher.write(k.as_bytes());
            entry_hasher.write(v.as_bytes());
            context_hash ^= entry_hasher.finish();
        }

        let location = if self.capture_caller {
             find_caller_in_rust(py, 2)
        } else { None };

        let mut msg_full = format!("{}Profile: {}", prefix, name);
        if let Some((fname, lno)) = location {
            let key = (fname.clone(), lno);
            let prefix_str = {
                let mut cache = self.caller_cache.lock();
                if let Some(p) = cache.get(&key) { p.clone() } 
                else { let p = format!("[{}:{}|dim] ", fname, lno); cache.insert(key, p.clone()); p }
            };
            msg_full = prefix_str + &msg_full;
        }

        let wall_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();

        let entry = LogEntry { 
            app_name: self.app_name.clone(), 
            timestamp: wall_ts,
            level, 
            message: msg_full, 
            path_template: self.path_template.clone(), 
            to_console: console, 
            to_file: true, 
            rate_limit: 0.0, 
            exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, 
            context: Some(context), context_hash, signal_shutdown: false 
        };
        
        // Push logic
        let thread_hash = get_current_thread_hash();
        let idx = (thread_hash as usize) % self.input_shards.len();
        if self.input_shards[idx].try_send(entry).is_err() {
            self.dropped_logs_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn push(&self, py: Python<'_>, timestamp: f64, level: u8, mut message: String, path_template: String, 
            to_console: bool, to_file: bool, rate_limit: f64, exc: Option<&Bound<'_, PyAny>>, 
            context_dict: Option<&Bound<'_, PyDict>>, 
            location_override: Option<(String, u32)>) {
        
        let found_location = if let Some(loc) = location_override {
            Some(loc)
        } else if self.capture_caller {
             find_caller_in_rust(py, 2)
        } else { None };

        if let Some((fname, lno)) = found_location {
            let key = (fname.clone(), lno);
            let prefix = {
                let mut cache = self.caller_cache.lock();
                if let Some(p) = cache.get(&key) { p.clone() } 
                else { let p = format!("[{}:{}|dim] ", fname, lno); cache.insert(key, p.clone()); p }
            };
            message = prefix + &message;
        }

        let mut exc_type = None; let mut exc_message = None; let mut trace_frames = None; let mut exc_hash: u64 = 0;
        if let Some(e) = exc {
            exc_type = e.get_type().name().ok().map(|n| n.to_string()); 
            exc_message = Some(e.to_string());
            let mut hasher = AHasher::default();
            if let Some(t) = &exc_type { t.hash(&mut hasher); } 
            if let Some(m) = &exc_message { m.hash(&mut hasher); }
            exc_hash = hasher.finish();
            
            let last_hash = self.last_push_exc_hash.load(Ordering::Relaxed);
            if last_hash != exc_hash {
                self.last_push_exc_hash.store(exc_hash, Ordering::Relaxed);
                if let Ok(mut tb) = e.getattr("__traceback__") {
                    let mut frames = Vec::with_capacity(8);
                    while !tb.is_none() {
                        if let (Ok(frame), Ok(lineno_any)) = (tb.getattr("tb_frame"), tb.getattr("tb_lineno")) {
                            if let Ok(code) = frame.getattr("f_code") {
                                let filename: String = code.getattr("co_filename").ok().and_then(|x| x.extract().ok()).unwrap_or_default();
                                let name: String = code.getattr("co_name").ok().and_then(|x| x.extract().ok()).unwrap_or_default();
                                let lineno: u32 = lineno_any.extract().unwrap_or(0);
                                frames.push(TraceFrame { filename, lineno, name });
                            }
                        }
                        if let Ok(next) = tb.getattr("tb_next") { tb = next; } else { break; }
                    }
                    trace_frames = Some(frames);
                }
            } 
        }

        let mut context: Option<IndexMap<String, String>> = None;
        let mut context_hash: u64 = 0;
        if let Some(dict) = context_dict {
            if !dict.is_empty() {
                let mut map = IndexMap::with_capacity(dict.len());
                for (k, v) in dict {
                    let key = k.to_string();
                    let val = v.to_string();
                    let mut entry_hasher = FxHasher::default();
                    entry_hasher.write(key.as_bytes());
                    entry_hasher.write(val.as_bytes());
                    context_hash ^= entry_hasher.finish();
                    map.insert(key, val);
                }
                context = Some(map);
            }
        }

        let entry = LogEntry { 
            app_name: self.app_name.clone(), timestamp, level, message, path_template, to_console, to_file, rate_limit, 
            exc_type, exc_message, exc_hash, trace_frames, context, context_hash, signal_shutdown: false 
        };
        
        let thread_hash = get_current_thread_hash();
        let idx = (thread_hash as usize) % self.input_shards.len();
        
        match self.input_shards[idx].try_send(entry) { 
            Ok(_) => {}, 
            Err(TrySendError::Full(_)) => { self.dropped_logs_count.fetch_add(1, Ordering::Relaxed); }, 
            Err(_) => {} 
        }
    }

    fn terminate(&self) {
        let shutdown_msg = LogEntry { 
            app_name: self.app_name.clone(), 
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
            level: 0, 
            message: String::new(), 
            path_template: self.path_template.clone(),
            to_console: false, 
            to_file: false, 
            rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, context: None, context_hash: 0, 
            signal_shutdown: true 
        };

        for tx in &self.input_shards {
             let _ = tx.send(shutdown_msg.clone());
        }

        let mut workers_guard = self.worker_threads.lock();
        for handle in workers_guard.drain(..) { 
            let _ = handle.join(); 
        }
        
        let _ = self.writer_tx.send(WriterMsg::Shutdown);
        let mut writer_guard = self.writer_thread.lock();
        if let Some(handle) = writer_guard.take() { 
            let _ = handle.join(); 
        }
    }
}

// === WORKER THREAD (Processor) ===
fn run_worker(
    rx: Receiver<LogEntry>,
    writer_tx: Sender<WriterMsg>,
    dropped_counter: Arc<AtomicUsize>,
    text_enabled: bool,
    db_enabled: bool,
    flush_interval_ms: u64,
    console_refresh_rate: u128,
) {
    let mut last_entry: Option<LogEntry> = None;
    let mut last_valid_traceframes: Option<Vec<TraceFrame>> = None;
    let mut last_valid_exc_hash: u64 = 0;
    
    let mut repeat_count: usize = 0;
    
    let mut first_ts: f64 = 0.0;
    let mut cached_ts_sec: i64 = 0;
    let mut cached_ts_prefix: String = String::new();
    let mut rate_limit_map: HashMap<u64, Instant, ahash::RandomState> = HashMap::default();
    
    let mut last_console_update = Instant::now();
    let mut last_msg_height: usize = 0;
    let mut last_flush_time = Instant::now();

    let mut binary_buffer: Vec<BinaryLogRecord> = Vec::with_capacity(BINARY_BATCH_CAPACITY);
    let pid = std::process::id();

    loop {
        let timeout = Duration::from_millis(flush_interval_ms);
        match rx.recv_timeout(timeout) {
            Err(RecvTimeoutError::Timeout) => {
                if let Some(last) = last_entry.take() {
                    // Finalize the last entry before timeout-based flush
                    if last.to_file {
                        let (prefix, nsecs) = get_time_parts(last.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                        if text_enabled {
                             let line = format_text_log(&last, repeat_count, first_ts, prefix, nsecs);
                             let target_path = get_target_path(&last.path_template, last.timestamp, ".log");
                             let _ = writer_tx.send(WriterMsg::TextLine { line, target_path });
                        }
                        if db_enabled {
                             add_to_binary_buffer(&mut binary_buffer, &last, pid, repeat_count);
                        }
                    }
                    if last.to_console {
                        let (prefix, nsecs) = get_time_parts(last.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                        update_console(&last, repeat_count, true, first_ts, prefix, nsecs, &mut last_msg_height);
                    }
                    // Flush whatever is in the binary buffer
                    if !binary_buffer.is_empty() {
                        flush_binary_buffer(&mut binary_buffer, &last, &writer_tx);
                    }
                    let _ = writer_tx.send(WriterMsg::Flush);
                    last_flush_time = Instant::now();
                    last_entry = None;
                    repeat_count = 0;
                }
                continue;
            },
            Err(RecvTimeoutError::Disconnected) => break,
            Ok(mut entry) => {
                if entry.signal_shutdown {
                    if let Some(last) = last_entry.take() {
                        finalize_and_flush(&last, repeat_count, &mut binary_buffer, &writer_tx, pid, text_enabled, db_enabled, &mut cached_ts_sec, &mut cached_ts_prefix, first_ts, &mut last_msg_height);
                    }
                    // Send a final "Process finished" message
                    if db_enabled {
                        let stop_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
                        let stop_msg = LogEntry {
                            app_name: entry.app_name.clone(), timestamp: stop_ts, level: 20, message: "🛑 Process finished".to_string(),
                            path_template: entry.path_template.clone(), to_console: false, to_file: true, 
                            rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, context: None, context_hash: 0, signal_shutdown: false
                        };
                        add_to_binary_buffer(&mut binary_buffer, &stop_msg, pid, 0);
                        flush_binary_buffer(&mut binary_buffer, &stop_msg, &writer_tx);
                    }
                    break; 
                }

                if dropped_counter.load(Ordering::Relaxed) > 0 {
                    let _ = dropped_counter.swap(0, Ordering::Relaxed);
                    // Force flush and reset state on drop
                    if let Some(last) = last_entry.take() {
                        finalize_and_flush(&last, repeat_count, &mut binary_buffer, &writer_tx, pid, text_enabled, db_enabled, &mut cached_ts_sec, &mut cached_ts_prefix, first_ts, &mut last_msg_height);
                    }
                    repeat_count = 0;
                }

                if entry.exc_hash != 0 {
                    if entry.trace_frames.is_none() { 
                        if entry.exc_hash == last_valid_exc_hash { entry.trace_frames = last_valid_traceframes.clone(); } 
                    } else { last_valid_exc_hash = entry.exc_hash; last_valid_traceframes = entry.trace_frames.clone(); }
                }

                if entry.rate_limit > 0.0 {
                    let mut hasher = AHasher::default(); entry.message.hash(&mut hasher); entry.level.hash(&mut hasher);
                    let msg_hash = hasher.finish();
                    let now = Instant::now();
                    if let Some(last_time) = rate_limit_map.get(&msg_hash) {
                        if now.duration_since(*last_time).as_secs_f64() < entry.rate_limit {
                            entry.to_console = false;
                        } else {
                            rate_limit_map.insert(msg_hash, now);
                        }
                    } else {
                        rate_limit_map.insert(msg_hash, now);
                    }
                }

                let is_dup = if let Some(ref last) = last_entry {
                    last.level == entry.level && last.message == entry.message && last.exc_hash == entry.exc_hash && 
                    last.app_name == entry.app_name && last.context_hash == entry.context_hash && last.to_console == entry.to_console 
                } else { false };

                if is_dup {
                    repeat_count += 1;
                    if let Some(ref mut last) = last_entry { last.timestamp = entry.timestamp; }
                } else {
                    if let Some(last) = last_entry.take() {
                        finalize_and_flush(&last, repeat_count, &mut binary_buffer, &writer_tx, pid, text_enabled, db_enabled, &mut cached_ts_sec, &mut cached_ts_prefix, first_ts, &mut last_msg_height);
                    }
                    repeat_count = 0;
                    
                    first_ts = entry.timestamp;
                    if entry.to_console {
                        let (prefix, nsecs) = get_time_parts(entry.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                        update_console(&entry, 0, false, first_ts, prefix, nsecs, &mut last_msg_height);
                        last_console_update = Instant::now(); 
                    }
                    last_entry = Some(entry);
                }

                if last_flush_time.elapsed().as_millis() >= flush_interval_ms as u128 || (db_enabled && binary_buffer.len() >= BINARY_BATCH_CAPACITY) {
                     if let Some(ref log) = last_entry {
                         flush_binary_buffer(&mut binary_buffer, log, &writer_tx);
                         let _ = writer_tx.send(WriterMsg::Flush);
                         last_flush_time = Instant::now();
                     }
                }
                if let Some(ref last) = last_entry {
                    if last.to_console && last_console_update.elapsed().as_millis() > console_refresh_rate {
                         let (prefix, nsecs) = get_time_parts(last.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                         update_console(last, repeat_count, true, first_ts, prefix, nsecs, &mut last_msg_height);
                         last_console_update = Instant::now();
                    }
                }
            }
        }
    }
}

// Helper to bundle finalization logic for a log entry
fn finalize_and_flush(
    last: &LogEntry, 
    repeats: usize, 
    bin_buf: &mut Vec<BinaryLogRecord>, 
    writer_tx: &Sender<WriterMsg>, 
    pid: u32, 
    text_enabled: bool, 
    db_enabled: bool,
    cached_ts_sec: &mut i64,
    cached_ts_prefix: &mut String,
    first_ts: f64,
    last_msg_height: &mut usize
) {
    if last.to_file {
        let (prefix, nsecs) = get_time_parts(last.timestamp, cached_ts_sec, cached_ts_prefix);
        if text_enabled {
            let line = format_text_log(last, repeats, first_ts, prefix, nsecs);
            let target_path = get_target_path(&last.path_template, last.timestamp, ".log");
            let _ = writer_tx.send(WriterMsg::TextLine { line, target_path });
        }
        if db_enabled {
            add_to_binary_buffer(bin_buf, last, pid, repeats);
        }
    }
    if last.to_console {
        let (prefix, nsecs) = get_time_parts(last.timestamp, cached_ts_sec, cached_ts_prefix);
        update_console(last, repeats, true, first_ts, prefix, nsecs, last_msg_height);
    }
    if db_enabled && !bin_buf.is_empty() {
        flush_binary_buffer(bin_buf, last, writer_tx);
    }
}


pub fn flush_binary_buffer(buf: &mut Vec<BinaryLogRecord>, last_log: &LogEntry, writer_tx: &Sender<WriterMsg>) {
    if buf.is_empty() {
        return;
    }
    let my_options = bincode::DefaultOptions::new().with_little_endian().with_fixint_encoding();
    if let Ok(raw_data) = my_options.serialize(&buf) {
        let compressed_data = lz4_flex::compress_prepend_size(&raw_data);
        
        let mut hasher = Crc32Hasher::new();
        hasher.update(&compressed_data);
        let checksum = hasher.finalize();
        let compressed_len = compressed_data.len() as u32;

        let mut blob = Vec::with_capacity(8 + compressed_data.len());
        // This is where WriteBytesExt trait is needed
        let _ = blob.write_u32::<LittleEndian>(checksum);
        let _ = blob.write_u32::<LittleEndian>(compressed_len);
        blob.extend_from_slice(&compressed_data);

        let target_path = get_target_path(&last_log.path_template, last_log.timestamp, ".ldb");
        let _ = writer_tx.send(WriterMsg::BinaryBlock { data: blob, target_path });
    }
    buf.clear();
}


fn format_text_log(entry: &LogEntry, repeat: usize, first_ts: f64, time_prefix: &str, time_nsecs: u32) -> String {
    let (lvl_str, _, _) = get_level_meta(entry.level);
    let clean_msg = remove_style_tags(&entry.message);
    let suffix = if repeat > 0 {
        let diff = entry.timestamp - first_ts;
        let time_fmt = if diff < 1.0 { format!("{:.0}ms", diff * 1000.0) } else { format!("{:.1}s", diff) };
        format!(" (x{} | {})", repeat + 1, time_fmt)
    } else { String::new() };

    let mut out = format!("{}.{:03} | {} | {}{}\n", time_prefix, time_nsecs, lvl_str, clean_msg, suffix);
    
    if entry.exc_type.is_some() {
        if let Some(frames) = &entry.trace_frames {
            out.push_str("Traceback (most recent call last):\n");
            for frame in frames {
                out.push_str(&format!("  File \"{}\", line {}, in {}\n", frame.filename, frame.lineno, frame.name));
            }
        }
        if let (Some(t), Some(m)) = (&entry.exc_type, &entry.exc_message) { 
            out.push_str(&format!("{}: {}\n", t, m)); 
        }
    }
    out
}

pub fn add_to_binary_buffer(buf: &mut Vec<BinaryLogRecord>, entry: &LogEntry, pid: u32, repeat: usize) {
    let traceback = if let (Some(exc_type), Some(exc_message), Some(frames)) = 
        (&entry.exc_type, &entry.exc_message, &entry.trace_frames) {
        Some(RawTraceback { exc_type: exc_type.clone(), exc_message: exc_message.clone(), frames: frames.clone() })
    } else { None };

    buf.push(BinaryLogRecord {
        ts: entry.timestamp, lvl: entry.level, app_name: entry.app_name.to_string(),
        pid, msg: entry.message.clone(), traceback, context: entry.context.clone(),
        count: (repeat + 1) as u32,
    });
}

fn get_target_path(template: &str, timestamp: f64, desired_ext: &str) -> PathBuf {
    let dt = DateTime::from_timestamp(timestamp as i64, 0).unwrap_or_default().with_timezone(&chrono::Local);
    let date_str = dt.format("%Y-%m-%d").to_string();
    let path_str = template.replace("{date}", &date_str);
    let mut path = PathBuf::from(path_str);
    path.set_extension(desired_ext.trim_start_matches('.'));
    path
}

// === WRITER THREAD (I/O Only) ===
fn run_writer(rx: Receiver<WriterMsg>) {
    let mut file_map: HashMap<PathBuf, BufWriter<File>> = HashMap::new();
    
    loop {
        match rx.recv() {
            Ok(msg) => match msg {
                WriterMsg::BinaryBlock { data, target_path } => {
                    // Get a mutable reference to the writer if it exists in the map
                    if !file_map.contains_key(&target_path) {
                        // If it doesn't exist, try to open and repair it
                        match open_and_repair_writer(&target_path, true) {
                            Ok(writer) => {
                                file_map.insert(target_path.clone(), writer);
                            },
                            Err(e) => {
                                eprintln!("🔥 Lumina IO Error: Failed to open/lock/repair file {:?}: {}", target_path, e);
                                // Skip this message, but keep the writer thread alive
                                continue;
                            }
                        }
                    }
                    
                    // Now we are sure the key exists if open_and_repair_writer succeeded
                    if let Some(writer) = file_map.get_mut(&target_path) {
                        if let Err(e) = writer.write_all(&data) {
                             eprintln!("🔥 Lumina IO Error: Failed to write to {:?}: {}", target_path, e);
                        }
                    }
                },
                WriterMsg::TextLine { line, target_path } => {
                    // Same logic as above but for text files
                     if !file_map.contains_key(&target_path) {
                        match open_and_repair_writer(&target_path, false) {
                            Ok(writer) => {
                                file_map.insert(target_path.clone(), writer);
                            },
                            Err(e) => {
                                eprintln!("🔥 Lumina IO Error: Failed to open/lock/repair file {:?}: {}", target_path, e);
                                continue;
                            }
                        }
                    }
                    if let Some(writer) = file_map.get_mut(&target_path) {
                        if let Err(e) = writer.write_all(line.as_bytes()) {
                            eprintln!("🔥 Lumina IO Error: Failed to write to {:?}: {}", target_path, e);
                        }
                    }
                },
                WriterMsg::Flush => {
                    for w in file_map.values_mut() { let _ = w.flush(); }
                },
                WriterMsg::Shutdown => {
                    for (_, mut writer) in file_map.drain() { 
                        let _ = writer.flush();
                        if let Ok(file) = writer.into_inner() {
                            let _ = file.unlock();
                        }
                    }
                    break;
                }
            },
            Err(_) => break, // Channel disconnected
        }
    }
}

/// Opens a writer for the given path.
/// If the file is binary and potentially corrupt, it will be validated and truncated.
fn open_and_repair_writer(path: &PathBuf, is_binary: bool) -> Result<BufWriter<File>, std::io::Error> {
    if let Some(p) = path.parent() { 
        if !p.exists() { 
            fs::create_dir_all(p)?;
        } 
    }
    
    // Open with read/write permissions for potential repair
    let mut file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
    
    // Try to get an exclusive lock to prevent race conditions during repair
    if file.try_lock_exclusive().is_err() {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Could not lock file {:?}", path)));
    }

    if is_binary {
        let file_len = file.metadata()?.len();

        // If file is new or just contains the header, no need to repair.
        if file_len <= 4 {
             if file_len == 0 {
                file.write_all(MAGIC_HEADER)?;
             }
        } else {
            // File has content, validate and find the last good position
            file.seek(SeekFrom::Start(0))?;
            let mut reader = BufReader::new(&file);
            let mut magic = [0u8; 4];
            
            // Check header, truncate if invalid
            if reader.read_exact(&mut magic).is_err() || magic != *MAGIC_HEADER {
                eprintln!("⚠️ Lumina: Invalid header in {:?}. Truncating file.", path);
                file.set_len(0)?;
                file.write_all(MAGIC_HEADER)?;
            } else {
                let mut last_valid_pos = 4; // Start after header
                loop {
                    let mut crc_bytes = [0u8; 4];
                    if reader.read_exact(&mut crc_bytes).is_err() { break; } // EOF, clean exit
                    
                    let mut len_bytes = [0u8; 4];
                    if reader.read_exact(&mut len_bytes).is_err() { break; } // Corrupted block
                    
                    let len = LittleEndian::read_u32(&len_bytes) as usize;
                    if len > 50 * 1024 * 1024 { break; } // Sanity check

                    let mut compressed = vec![0u8; len];
                    if reader.read_exact(&mut compressed).is_err() { break; } // Corrupted block
                    
                    let mut hasher = Crc32Hasher::new();
                    hasher.update(&compressed);
                    
                    if hasher.finalize() == LittleEndian::read_u32(&crc_bytes) {
                        // This block is valid, update position
                        last_valid_pos = reader.stream_position()?;
                    } else {
                        // CRC mismatch, this is where corruption starts
                        break; 
                    }
                }

                if last_valid_pos < file_len {
                    eprintln!("⚠️ Lumina: Detected corruption at the end of {:?}. Repairing by truncating.", path);
                    file.set_len(last_valid_pos)?;
                }
            }
        }
    }

    // Now that the file is repaired and locked, seek to the end for appending
    file.seek(SeekFrom::End(0))?;

    let buffer_size = if is_binary { BINARY_FILE_BUFFER_SIZE } else { TEXT_FILE_BUFFER_SIZE };
    Ok(BufWriter::with_capacity(buffer_size, file))
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;
    use crate::reader::{run_reader_impl, ReaderConfig};
    use std::io::Write;

    // Helper function is now updated to accept a path template.
    fn create_test_entry(level: u8, message: &str, path_template: String) -> LogEntry {
        LogEntry {
            app_name: "test_app".into(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
            level,
            message: message.to_string(),
            path_template, // Use the provided path template
            to_console: false,
            to_file: true,
            rate_limit: 0.0,
            exc_type: None,
            exc_message: None,
            exc_hash: 0,
            trace_frames: None,
            context: None,
            context_hash: 0,
            signal_shutdown: false,
        }
    }

    #[test]
    fn test_binary_format_integrity_and_repair() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("test.ldb");
        // This path is now a concrete path like "/tmp/xyz/test.ldb"
        let log_path_str = log_path.to_str().unwrap().to_string();

        let (writer_tx, writer_rx) = bounded::<WriterMsg>(10);
        
        let writer_handle = thread::spawn(move || run_writer(writer_rx));

        // --- Phase 1: Write a valid data block ---
        let mut bin_buf = Vec::new();
        // We pass the correct, concrete path to the log entry.
        let entry1 = create_test_entry(20, "message 1", log_path_str.clone());
        add_to_binary_buffer(&mut bin_buf, &entry1, 123, 0);
        
        // Send for writing. The writer will now use the correct path inside the temp directory.
        flush_binary_buffer(&mut bin_buf, &entry1, &writer_tx);
        writer_tx.send(WriterMsg::Flush).unwrap();
        
        // Sleep is still good practice to allow the OS scheduler to run the writer thread.
        thread::sleep(Duration::from_millis(100));

        // This check should now succeed because the writer wrote to the file we are checking.
        let valid_size = fs::metadata(&log_path).unwrap().len();
        assert!(valid_size > 4, "The file should contain more than just the header");

        // --- Phase 2: Simulate file corruption ---
        let mut file = OpenOptions::new().append(true).open(&log_path).unwrap();
        file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        drop(file);
        
        let corrupted_size = fs::metadata(&log_path).unwrap().len();
        assert_eq!(corrupted_size, valid_size + 4, "The file size should increase by 4 bytes of garbage");

        // --- Phase 3: Re-open the file implicitly on the next write ---
        let entry2 = create_test_entry(30, "message 2", log_path_str.clone());
        add_to_binary_buffer(&mut bin_buf, &entry2, 123, 0);
        
        // This operation should trigger file check and repair on the correct file.
        flush_binary_buffer(&mut bin_buf, &entry2, &writer_tx);
        
        // Shut down the writer to flush everything to disk.
        writer_tx.send(WriterMsg::Shutdown).unwrap();
        writer_handle.join().unwrap();

        // --- Phase 4: Check the result ---
        let repaired_size = fs::metadata(&log_path).unwrap().len();
        assert!(repaired_size > valid_size, "The second block should have been written");
    
        let export_path = dir.path().join("export.json");
        let config_export = ReaderConfig {
            file_pattern: Some(log_path_str),
            json_output: false,
            export_json: Some(export_path.to_str().unwrap().to_string()),
            ..Default::default()
        };

        run_reader_impl(config_export).unwrap();
        
        let result_json = fs::read_to_string(export_path).unwrap();
        let records: Vec<serde_json::Value> = serde_json::from_str(&result_json).unwrap();
        
        assert_eq!(records.len(), 2, "Exactly 2 records should be read");
        assert_eq!(records[0]["msg"], "message 1");
        assert_eq!(records[1]["msg"], "message 2");
    }

    #[test]
    fn test_worker_deduplication() {
        let (tx, rx) = bounded::<LogEntry>(100);
        let (writer_tx, writer_rx) = bounded::<WriterMsg>(10);
        let dropped = Arc::new(AtomicUsize::new(0));

        let worker_handle = thread::spawn(move || {
            run_worker(rx, writer_tx, dropped, false, true, 1000, 100);
        });
        
        // This test doesn't write to the filesystem, so a dummy path is fine.
        let template = "logs/dedup_test.ldb".to_string();

        // Send 3 identical messages.
        let entry1 = create_test_entry(20, "repeated message", template.clone());
        tx.send(entry1.clone()).unwrap();
        tx.send(entry1.clone()).unwrap();
        tx.send(entry1.clone()).unwrap();

        // Send a different message to "reset" the deduplication state.
        let entry2 = create_test_entry(30, "unique message", template.clone());
        tx.send(entry2.clone()).unwrap();
        
        // Send a termination signal.
        let mut shutdown_msg = create_test_entry(0, "", template.clone());
        shutdown_msg.signal_shutdown = true;
        tx.send(shutdown_msg).unwrap();

        worker_handle.join().unwrap();
        
        // Collect results from the writer's channel.
        let results: Vec<WriterMsg> = writer_rx.try_iter().collect();

        let mut total_records = Vec::new();
        for msg in results {
            if let WriterMsg::BinaryBlock { data, .. } = msg {
                 // Unpack the binary block to check its contents.
                 let checksum = LittleEndian::read_u32(&data[0..4]);
                 let len = LittleEndian::read_u32(&data[4..8]) as usize;
                 let compressed = &data[8..8+len];

                 let mut hasher = Crc32Hasher::new();
                 hasher.update(compressed);
                 assert_eq!(hasher.finalize(), checksum, "CRC32 checksum mismatch");
                 
                 let raw = lz4_flex::decompress_size_prepended(compressed).unwrap();
                 let my_options = bincode::DefaultOptions::new().with_little_endian().with_fixint_encoding();
                 let records: Vec<BinaryLogRecord> = my_options.deserialize(&raw).unwrap();
                 total_records.extend(records);
            }
        }
        
        // Expecting at least 2 records: one with count=3, another with count=1.
        assert!(total_records.len() >= 2, "At least 2 records are expected");
        
        let first_record = total_records.iter().find(|r| r.msg == "repeated message").unwrap();
        assert_eq!(first_record.count, 3, "The counter for the repeated message should be 3");
        
        let second_record = total_records.iter().find(|r| r.msg == "unique message").unwrap();
        assert_eq!(second_record.count, 1, "The counter for the unique message should be 1");
    }
}