use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyDict};
use std::thread::{self, JoinHandle};
use std::collections::HashMap;
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use crossbeam_channel::{bounded, Sender, TrySendError, RecvTimeoutError};
use std::sync::Arc;
use ahash::AHasher; 
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::io::{stdout, Write}; 
use std::env; 
use std::path::Path; 
use std::fs; 
use colored::*;
use sysinfo::{Pid, System}; 
use indexmap::IndexMap;
use rustc_hash::FxHasher;

#[cfg(unix)]
use libc::{getrusage, rusage, RUSAGE_SELF};

use crate::types::{LogEntry, TraceFrame, CallerCache, Theme};
use crate::utils::{get_time_parts, fast_colorize, get_level_meta, 
    set_theme, measure_text_height, set_colors_enabled};
use crate::drivers::{LogDriver, text::TextFileDriver, binary::LuminaDbDriver};

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
                    if filename.contains(marker) {
                        is_skipped = true;
                        break;
                    }
                }

                let path = Path::new(&filename);
                let cwd = env::current_dir().unwrap_or_default();
                let final_path = path.strip_prefix(&cwd).unwrap_or(path).to_string_lossy().to_string();
                let lineno: u32 = f.getattr("f_lineno").ok().and_then(|l| l.extract().ok()).unwrap_or(0);

                if !is_skipped {
                    return Some((final_path, lineno));
                }
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
    if val > 1024.0 * 1024.0 {
        format!("{}{:.2}MB", sign, val / 1024.0 / 1024.0)
    } else if val > 1024.0 {
        format!("{}{:.2}KB", sign, val / 1024.0)
    } else {
        format!("{}{:.0}B", sign, val)
    }
}

fn format_ram_abs(val: u64) -> String {
    format!("{:.1}MB", val as f64 / 1024.0 / 1024.0)
}

/// Updates the console output, handling rewriting previous lines if necessary.
fn update_console(
    entry: &LogEntry, repeat: usize, is_update: bool, first_ts: f64, 
    time_prefix: &str, time_nsecs: u32, last_height: usize
) -> usize {
    if !entry.to_console { return 0; }
    let (_, lvl_color, icon) = get_level_meta(entry.level);
    let safe_msg_raw = crate::utils::sanitize_input(&entry.message);
    let colored_msg = fast_colorize(&safe_msg_raw);
    
    let is_light = {
        if let Ok(r) = crate::utils::CURRENT_THEME.read() { *r == Theme::Light } else { false }
    };

    let context_str = if let Some(ctx) = &entry.context {
        let mut s = String::new();
        for (k, v) in ctx {
            let safe_v = crate::utils::sanitize_input(v);
            if is_light {
                s.push_str(&format!(" {}={}", k.black().dimmed(), safe_v.blue()));
            } else {
                s.push_str(&format!(" {}={}", k.dimmed(), safe_v.cyan()));
            }
        }
        s
    } else {
        String::new()
    };

    let suffix = if repeat > 0 {
        let diff = entry.timestamp - first_ts;
        let time_fmt = if diff < 1.0 { format!("{:.0}ms", diff * 1000.0) } else { format!("{:.1}s", diff) };
        if is_light {
            format!(" (x{} │ {})", repeat + 1, time_fmt).black().dimmed().to_string()
        } else {
            format!(" (x{} │ {})", repeat + 1, time_fmt).yellow().dimmed().to_string()
        }
    } else { String::new() };

    let time_display = if is_light {
        format!("{}.{:03}", time_prefix, time_nsecs).black().dimmed()
    } else {
        format!("{}.{:03}", time_prefix, time_nsecs).cyan().dimmed()
    };
    
    let header = format!("{} {} {: <9} │ {}{}{}", time_display, icon, lvl_color, colored_msg, context_str, suffix);
    let mut stdout = stdout();
    
    let mut current_height = measure_text_height(&header);

    if is_update {
        if last_height > 0 {
            let move_up = format!("\x1b[{}A", last_height);
            let _ = write!(stdout, "{}", move_up);
        } else { 
            let _ = write!(stdout, "\x1b[1A"); 
        }
        let _ = write!(stdout, "\x1b[J");
    }
    let _ = writeln!(stdout, "{}", header);

    if entry.exc_type.is_some() {
        let mut tb_str = String::new();
        if let Some(frames) = &entry.trace_frames {
            let header = if is_light { "Traceback (most recent call last):".magenta().to_string() } 
                         else { "Traceback (most recent call last):".yellow().dimmed().to_string() };
            tb_str.push_str(&format!("{}\n", header));
            // Limit frames in live console view to avoid spam and performance issues
            for frame in frames.iter().take(5) {
                 let fname = if is_light { frame.filename.blue().to_string() } else { frame.filename.blue().underline().to_string() };
                 let line = if is_light { frame.lineno.to_string().purple().to_string() } else { frame.lineno.to_string().yellow().to_string() };
                 let func = if is_light { frame.name.black().bold().to_string() } else { frame.name.cyan().to_string() };
                 tb_str.push_str(&format!("  File \"{}\", line {}, in {}\n", fname, line, func));
            }
            if frames.len() > 5 {
                tb_str.push_str(&format!("  ... ({} more frames)\n", frames.len() - 5));
            }
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
    current_height
}

/// Deletes log files older than `retention_days`.
fn cleanup_old_logs(path_template: String, retention_days: u64) {
    if retention_days == 0 { return; }
    let path = Path::new(&path_template);
    let log_dir = match path.parent() { Some(p) if p.as_os_str().is_empty() => Path::new("."), Some(p) => p, None => return };
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

/// The core engine of Lumina.
#[pyclass]
pub struct LuminaEngine {
    tx: Sender<LogEntry>,
    worker_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
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
        file_buffer_size: usize, 
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
            let template_for_cleanup = cleanup_path_template.clone();
            thread::spawn(move || { cleanup_old_logs(template_for_cleanup, retention_days); });
        }

        let (tx, rx) = bounded::<LogEntry>(channel_capacity);
        let dropped_logs_count = Arc::new(AtomicUsize::new(0));
        let worker_dropped_count = dropped_logs_count.clone();

        let system_monitor = Arc::new(Mutex::new(System::new()));

        let worker_capacity = channel_capacity;
        let worker_max_flush_ms = flush_interval_ms.max(1); 
        const MIN_FLUSH_MS: u64 = 1; 

        let start_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();

        let app_name_arc: Arc<str> = app_name.clone().into();
        let start_msg = LogEntry {
            app_name: app_name_arc.clone(),
            timestamp: start_ts,
            level: 20,
            message: "🚀 Process started".to_string(),
            path_template: cleanup_path_template.clone(),
            to_console: false,
            to_file: true,
            rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, 
            context: None, context_hash: 0,
            signal_shutdown: false
        };
        let _ = tx.send(start_msg);

        // Spawn the worker thread
        let handle = thread::Builder::new().name("lumina-worker".to_string()).spawn(move || {
            let mut drivers: Vec<Box<dyn LogDriver>> = Vec::new();
            if text_enabled { drivers.push(Box::new(TextFileDriver { writer: None, current_path: String::new(), buffer_size: file_buffer_size })); }
            if db_enabled { drivers.push(Box::new(LuminaDbDriver::new())); }

            let mut last_entry: Option<LogEntry> = None;
            let mut last_valid_traceframes: Option<Vec<TraceFrame>> = None;
            let mut last_valid_exc_hash: u64 = 0;
            
            let mut repeat_count: usize = 0;
            let mut unflushed_repeats: usize = 0;
            
            let mut first_ts: f64 = 0.0;
            let mut cached_ts_sec: i64 = 0;
            let mut cached_ts_prefix: String = String::new();
            let mut rate_limit_map: HashMap<u64, Instant, ahash::RandomState> = HashMap::default();
            let mut last_console_update = Instant::now();
            let mut skipped_console_update = false;
            let mut last_overload_check = Instant::now();
            let mut last_msg_height: usize = 0;
            
            let mut last_flush_time = Instant::now();

            loop {
                let current_occupancy = rx.len();
                let effective_flush_interval = if current_occupancy == 0 {
                    Duration::from_millis(worker_max_flush_ms)
                } else {
                    let ratio = current_occupancy as f64 / worker_capacity as f64;
                    let dynamic_ms = if ratio > 0.8 { MIN_FLUSH_MS } else if ratio < 0.1 { worker_max_flush_ms } else {
                        let effective_ratio = (ratio - 0.1) / 0.7;
                        let range = worker_max_flush_ms.saturating_sub(MIN_FLUSH_MS);
                        let reduction = (range as f64 * effective_ratio) as u64;
                        worker_max_flush_ms.saturating_sub(reduction).max(MIN_FLUSH_MS)
                    };
                    Duration::from_millis(dynamic_ms)
                };

                let entry_result = rx.recv_timeout(effective_flush_interval);

                match entry_result {
                    Err(RecvTimeoutError::Timeout) => {
                        if unflushed_repeats > 0 {
                            if let Some(ref prev) = last_entry {
                                let (prefix, nsecs) = get_time_parts(prev.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                let count_arg = unflushed_repeats - 1;
                                for d in &mut drivers { d.write(prev, count_arg, first_ts, prefix, nsecs); }
                            }
                            unflushed_repeats = 0; 
                        }
                        for d in &mut drivers { d.flush(); }
                        last_flush_time = Instant::now(); 
                        
                        if let Some(ref current_entry) = last_entry {
                            if current_entry.to_console && Instant::now().duration_since(last_console_update).as_millis() > console_refresh_rate {
                                let (prefix, nsecs) = get_time_parts(current_entry.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                last_msg_height = update_console(current_entry, repeat_count, true, first_ts, prefix, nsecs, last_msg_height);
                                last_console_update = Instant::now();
                            }
                        }
                        continue;
                    },
                    Err(RecvTimeoutError::Disconnected) => break,
                    Ok(mut entry) => {
                        if last_overload_check.elapsed().as_millis() > 1000 {
                             let dropped = worker_dropped_count.swap(0, Ordering::Relaxed);
                             if dropped > 0 {
                                 if unflushed_repeats > 0 {
                                     if let Some(prev) = last_entry.take() { 
                                         let (prefix, nsecs) = get_time_parts(prev.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                         let count_arg = unflushed_repeats - 1;
                                         for d in &mut drivers { d.write(&prev, count_arg, first_ts, prefix, nsecs); }
                                     }
                                 }
                                 unflushed_repeats = 0; repeat_count = 0; last_entry = None; 
                             }
                             last_overload_check = Instant::now();
                        }

                        if entry.signal_shutdown {
                             if let Some(prev) = last_entry.take() { 
                                 if unflushed_repeats > 0 {
                                    let (prev_prefix, prev_nsecs) = get_time_parts(prev.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                    if skipped_console_update && prev.to_console { 
                                        update_console(&prev, repeat_count, true, first_ts, prev_prefix, prev_nsecs, last_msg_height); 
                                    }
                                    let count_arg = unflushed_repeats - 1;
                                    for d in &mut drivers { d.write(&prev, count_arg, first_ts, prev_prefix, prev_nsecs); }
                                 }
                             }
                             let stop_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
                             let stop_msg = LogEntry {
                                app_name: entry.app_name.clone(), timestamp: stop_ts, level: 20, message: "🛑 Process finished".to_string(),
                                path_template: entry.path_template.clone(), to_console: false, to_file: true, 
                                rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, context: None, context_hash: 0, signal_shutdown: false
                             };
                             let (prefix, nsecs) = get_time_parts(stop_msg.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                             for d in &mut drivers { d.write(&stop_msg, 0, stop_msg.timestamp, prefix, nsecs); }
                             for d in &mut drivers { d.flush(); }
                             let _ = stdout().flush();
                             break;
                        }
                        
                        if entry.exc_hash != 0 {
                            if entry.trace_frames.is_none() { 
                                if entry.exc_hash == last_valid_exc_hash { entry.trace_frames = last_valid_traceframes.clone(); } 
                            } else { last_valid_exc_hash = entry.exc_hash; last_valid_traceframes = entry.trace_frames.clone(); }
                        }

                        let mut skip_console = false;
                        if entry.rate_limit > 0.0 {
                            let mut hasher = AHasher::default(); entry.message.hash(&mut hasher); entry.level.hash(&mut hasher); entry.exc_hash.hash(&mut hasher);
                            let msg_hash = hasher.finish(); let now = Instant::now();
                            rate_limit_map.entry(msg_hash).and_modify(|last| { if now.duration_since(*last).as_secs_f64() < entry.rate_limit { skip_console = true; } else { *last = now; } }).or_insert(now);
                        }
                        if skip_console { entry.to_console = false; }
                        
                        let current_entry_to_console = entry.to_console;

                        let is_duplicate = if let Some(ref last) = last_entry {
                            last.level == entry.level && last.message == entry.message && last.exc_hash == entry.exc_hash && 
                            last.app_name == entry.app_name && last.context_hash == entry.context_hash && current_entry_to_console == last.to_console 
                        } else { false };

                        if is_duplicate {
                            repeat_count += 1; unflushed_repeats += 1;
                            if let Some(ref mut last) = last_entry { last.timestamp = entry.timestamp; }
                            let now = Instant::now();
                            if now.duration_since(last_flush_time).as_millis() >= worker_max_flush_ms as u128 {
                                if unflushed_repeats > 0 {
                                    if let Some(ref prev) = last_entry { 
                                        let (prefix, nsecs) = get_time_parts(prev.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                        let count_arg = unflushed_repeats - 1;
                                        for d in &mut drivers { d.write(prev, count_arg, first_ts, prefix, nsecs); }
                                    }
                                    unflushed_repeats = 0; 
                                }
                                for d in &mut drivers { d.flush(); }
                                last_flush_time = now;
                            }
                            if current_entry_to_console && now.duration_since(last_console_update).as_millis() > console_refresh_rate {
                                if let Some(ref current_entry) = last_entry { 
                                    let (prefix, nsecs) = get_time_parts(current_entry.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                    last_msg_height = update_console(current_entry, repeat_count, true, first_ts, prefix, nsecs, last_msg_height);
                                    last_console_update = Instant::now(); skipped_console_update = false;
                                }
                            } else { skipped_console_update = true; }

                        } else {
                            if let Some(prev) = last_entry.take() { 
                                let (prev_prefix, prev_nsecs) = get_time_parts(prev.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);
                                if unflushed_repeats > 0 {
                                    let count_to_write = unflushed_repeats - 1;
                                    for d in &mut drivers { d.write(&prev, count_to_write, first_ts, prev_prefix, prev_nsecs); }
                                }
                                if skipped_console_update && prev.to_console { 
                                    update_console(&prev, repeat_count, true, first_ts, prev_prefix, prev_nsecs, last_msg_height); 
                                }
                                for d in &mut drivers { d.flush(); }
                                last_flush_time = Instant::now();
                            }
                            
                            repeat_count = 0; first_ts = entry.timestamp; 
                            let (prefix, nsecs) = get_time_parts(entry.timestamp, &mut cached_ts_sec, &mut cached_ts_prefix);

                            if current_entry_to_console {
                                last_msg_height = update_console(&entry, 0, false, first_ts, prefix, nsecs, 0);
                                last_console_update = Instant::now(); skipped_console_update = false;
                            } else { last_msg_height = 0; }
                            
                            last_entry = Some(entry); unflushed_repeats = 1; 
                        }
                    }
                }
            }
        }).expect("Failed to spawn lumina worker");

        LuminaEngine { 
            tx, 
            worker_thread: Arc::new(Mutex::new(Some(handle))), 
            last_push_exc_hash: Arc::new(AtomicU64::new(0)), 
            dropped_logs_count, 
            caller_cache: Arc::new(Mutex::new(HashMap::new())), 
            system_monitor, 
            app_name: app_name_arc, 
            capture_caller,
            path_template: cleanup_path_template 
        }
    }

    /// Captures current system resources (CPU, RAM, Page Faults).
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
    
    /// Calculates the difference between two snapshots and pushes a profile log entry.
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
        
        let _ = self.tx.try_send(entry);
    }

    /// Pushes a standard log message to the queue.
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
        
        match self.tx.try_send(entry) { Ok(_) => {}, Err(TrySendError::Full(_)) => { self.dropped_logs_count.fetch_add(1, Ordering::Relaxed); }, Err(_) => {} }
    }

    fn terminate(&self) {
        let _ = self.tx.send(LogEntry { 
            app_name: self.app_name.clone(), 
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
            level: 0, 
            message: String::new(), 
            path_template: self.path_template.clone(),
            to_console: false, 
            to_file: false, 
            rate_limit: 0.0, exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, context: None, context_hash: 0, 
            signal_shutdown: true 
        });
        let mut guard = self.worker_thread.lock();
        if let Some(handle) = guard.take() { let _ = handle.join(); }
    }
}