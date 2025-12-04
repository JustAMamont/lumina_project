use pyo3::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write, stdout, Stdout, IsTerminal};
use byteorder::{ReadBytesExt, LittleEndian};
use bincode::Options;
use chrono::{DateTime, Local};
use colored::*;
use crc32fast::Hasher as Crc32Hasher;
use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Ordering;
use std::thread;
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use crossterm::{cursor::MoveUp, terminal::{Clear, ClearType}, ExecutableCommand};

use crate::utils::{fast_colorize, get_level_meta, set_theme, measure_text_height, remove_style_tags};
use crate::types::{BinaryLogRecord, JsonLogRecord, Theme, RawTraceback, FileCache};

/// Magic header to identify valid Lumina binary logs.
const MAGIC_HEADER: &[u8; 4] = b"LUM1";
const FILE_BUFFER_SIZE: usize = 128 * 1024; // 128KB Read Buffer
const STDOUT_BUFFER_SIZE: usize = 128 * 1024; // 128KB Write Buffer

/// Configuration object for the Log Reader.
#[derive(Clone)]
pub struct ReaderConfig {
    pub file_pattern: Option<String>,
    pub min_level: u8,
    pub show_trace: bool,
    pub json_output: bool,
    pub follow: bool,
    pub target_levels: Option<Vec<u8>>,
    pub grep: Option<String>,
    pub trace_type: Option<String>,
    pub start_ts: Option<f64>,
    pub theme: Option<String>,
    pub export_json: Option<String>,
}

/// Helper struct for the Priority Queue (BinaryHeap).
struct HeapEntry {
    record: BinaryLogRecord,
    file_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.record.ts == other.record.ts }
}
impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.record.ts.partial_cmp(&self.record.ts)
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

/// Iterator that reads a single `.ldb` file block by block.
struct SingleFileIterator {
    reader: BufReader<File>,
    buffer: std::vec::IntoIter<BinaryLogRecord>,
    path: String,
    is_dead: bool,
    min_level: u8,
    target_levels: Option<Vec<u8>>,
    start_ts: Option<f64>,
}

impl SingleFileIterator {
    fn new(path: &str, config: &ReaderConfig) -> Option<Self> {
        let file = match File::open(path) {
            Ok(f) => f, Err(_) => return None,
        };
        let mut reader = BufReader::with_capacity(FILE_BUFFER_SIZE, file);
        
        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() || &magic != MAGIC_HEADER { return None; }
        
        Some(Self {
            reader,
            buffer: Vec::new().into_iter(),
            path: path.to_string(),
            is_dead: false,
            min_level: config.min_level,
            target_levels: config.target_levels.clone(),
            start_ts: config.start_ts,
        })
    }
    
    fn next_rec(&mut self) -> Option<BinaryLogRecord> {
        if self.is_dead { return None; }
        loop {
            while let Some(rec) = self.buffer.next() {
                if let Some(start) = self.start_ts {
                    if rec.ts < start { continue; }
                }
                
                let level_ok = if let Some(levels) = &self.target_levels {
                    levels.contains(&rec.lvl)
                } else {
                    rec.lvl >= self.min_level
                };

                if level_ok {
                    return Some(rec);
                }
            }

            let current_pos = self.reader.stream_position().unwrap_or(0);
            
            let expected_crc = match self.reader.read_u32::<LittleEndian>() {
                Ok(c) => c,
                Err(_) => { return None; }
            };

            let len = match self.reader.read_u32::<LittleEndian>() {
                Ok(l) => l as usize,
                Err(_) => {
                    let _ = self.reader.seek(SeekFrom::Start(current_pos));
                    return None;
                }
            };

            if len > 50 * 1024 * 1024 {
                let _ = self.reader.seek(SeekFrom::Start(current_pos));
                return None;
            }

            let mut compressed = vec![0u8; len];
            if self.reader.read_exact(&mut compressed).is_err() {
                let _ = self.reader.seek(SeekFrom::Start(current_pos));
                return None;
            }
            
            let mut hasher = Crc32Hasher::new();
            hasher.update(&compressed);
            if hasher.finalize() != expected_crc {
                let _ = self.reader.seek(SeekFrom::Start(current_pos));
                return None;
            }
            
            let raw = match lz4_flex::decompress_size_prepended(&compressed) {
                Ok(d) => d,
                Err(_) => {
                    eprintln!("❌ Corrupted block in file: {}", self.path);
                    self.is_dead = true;
                    return None;
                }
            };

            let opts = bincode::DefaultOptions::new().with_little_endian().with_fixint_encoding();
            match opts.deserialize::<Vec<BinaryLogRecord>>(&raw) {
                Ok(records) => {
                    self.buffer = records.into_iter();
                },
                Err(_) => {
                    self.is_dead = true;
                    return None;
                }
            }
        }
    }
}

// === Python Wrapper ===
#[pyfunction]
#[pyo3(signature = (file_pattern_arg=None, min_level=0, show_trace=false, json_output=false,
    follow=false, target_levels=None, grep=None, trace_type=None, start_ts=None, theme=None,
    export_json=None, no_color=false))]
pub fn read_binary_log(
    file_pattern_arg: Option<String>,
    min_level: u8,
    show_trace: bool,
    json_output: bool,
    follow: bool,
    target_levels: Option<Vec<u8>>,
    grep: Option<String>,
    trace_type: Option<String>,
    start_ts: Option<f64>,
    theme: Option<String>,
    export_json: Option<String>,
    no_color: bool,
) -> PyResult<()> {
    crate::utils::set_colors_enabled(!no_color);
    
    let config = ReaderConfig {
        file_pattern: file_pattern_arg,
        min_level,
        show_trace,
        json_output,
        follow,
        target_levels,
        grep,
        trace_type,
        start_ts,
        theme,
        export_json
    };

    match run_reader_impl(config) {
        Ok(_) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

// === Pure Rust Implementation ===
pub fn run_reader_impl(config: ReaderConfig) -> Result<(), Box<dyn std::error::Error>> {
    
    let theme_enum = Theme::from_str(&config.theme.clone().unwrap_or_else(|| "dark".to_string()));
    set_theme(theme_enum);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let _ = ctrlc::set_handler(move || { r.store(false, AtomicOrdering::SeqCst); });

    let search_term = config.grep.as_ref().map(|s| s.to_lowercase());
    let type_filter = config.trace_type.as_ref().map(|s| s.to_lowercase());

    let mut json_writer = if let Some(path) = &config.export_json {
        match OpenOptions::new().create(true).write(true).truncate(true).open(path) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                write!(w, "[")?;
                Some(w)
            },
            Err(e) => return Err(Box::new(e)),
        }
    } else { None };
    let mut is_first_export_record = true;

    let mut stdout_handle = BufWriter::with_capacity(STDOUT_BUFFER_SIZE, stdout());

    let mut iterators: Vec<Option<SingleFileIterator>> = Vec::new();
    let mut heap = BinaryHeap::new();
    let mut known_paths: HashSet<PathBuf> = HashSet::new();

    let scan_files = |iters: &mut Vec<Option<SingleFileIterator>>, h: &mut BinaryHeap<HeapEntry>, paths_set: &mut HashSet<PathBuf>| {
        let mut candidates = Vec::new();
        
        let patterns_to_check = if let Some(p) = &config.file_pattern {
            let path_obj = Path::new(p);
            if path_obj.is_dir() { vec![format!("{}/**/*.ldb", p.trim_end_matches('/'))] }
            else { vec![p.clone()] }
        } else { vec!["**/*.ldb".to_string()] };

        for pat in patterns_to_check {
             if let Ok(paths) = glob::glob(&pat) {
                 for entry in paths { if let Ok(p) = entry { if p.is_file() { candidates.push(p); } } }
             }
        }
        
        if candidates.is_empty() && config.file_pattern.is_none() {
             let now = Local::now();
             let current_date = now.date_naive();
             let start_date = if let Some(ts) = config.start_ts {
                 let secs = ts as i64;
                 DateTime::from_timestamp(secs, 0).map(|d| d.with_timezone(&Local).date_naive()).unwrap_or(current_date)
             } else { current_date };
             let mut d = start_date;
             while d <= current_date {
                 let fname = format!("logs/{}.ldb", d.format("%Y-%m-%d"));
                 let p = PathBuf::from(fname);
                 if p.exists() { candidates.push(p); }
                 if let Some(next_d) = d.succ_opt() { d = next_d; } else { break; }
             }
        }

        for path in candidates {
            if !paths_set.contains(&path) {
                if let Some(mut iter) = SingleFileIterator::new(path.to_str().unwrap(), &config) {
                    let file_idx = iters.len();
                    if let Some(first) = iter.next_rec() { h.push(HeapEntry { record: first, file_idx }); }
                    iters.push(Some(iter));
                    paths_set.insert(path);
                }
            }
        }
    };

    scan_files(&mut iterators, &mut heap, &mut known_paths);
    if known_paths.is_empty() && !config.follow { if !config.json_output { eprintln!("⚠️ No log files found."); } return Ok(()); }

    let mut file_cache: FileCache = HashMap::new();
    let mut pending_record: Option<BinaryLogRecord> = None;
    let mut pending_first_ts: f64 = 0.0;
    let mut last_height: usize = 0;
    let mut last_rescan = Instant::now();
    let is_tty_follow = std::io::stdout().is_terminal() && config.follow && !config.json_output;

    while running.load(AtomicOrdering::SeqCst) {
        let heap_entry = heap.pop();

        if let Some(entry) = heap_entry {
            let rec = entry.record;
            let file_idx = entry.file_idx;
            if let Some(iter) = iterators[file_idx].as_mut() {
                if let Some(next) = iter.next_rec() { heap.push(HeapEntry { record: next, file_idx }); }
            }

            let mut show = true;
            if let Some(term) = &search_term {
                let content = if let Some(tb) = &rec.traceback {
                    format!("{} {} {} {}", rec.app_name, rec.msg, tb.exc_type, tb.exc_message)
                } else { format!("{} {}", rec.app_name, rec.msg) };
                if !content.to_lowercase().contains(term) { show = false; }
            }
            if let Some(tf) = &type_filter {
                if let Some(tb) = &rec.traceback {
                    if !tb.exc_type.to_lowercase().contains(tf) { show = false; }
                } else { show = false; }
            }

            if !show { continue; }

            // CORRECTED LOGIC BLOCK
            let is_duplicate = if let Some(pending) = &pending_record {
                pending.lvl == rec.lvl && pending.msg == rec.msg && pending.traceback.is_none() && rec.traceback.is_none() && pending.app_name == rec.app_name && pending.context == rec.context
            } else {
                false
            };

            if is_duplicate {
                // If the new record is a duplicate, just update the pending record's state.
                if let Some(pending) = pending_record.as_mut() {
                    pending.count += rec.count;
                    pending.ts = rec.ts;
                    // If we're in interactive follow mode, update the screen in-place.
                    if is_tty_follow {
                        last_height = print_entry(&mut stdout_handle, pending, &config, true, last_height, pending_first_ts, &mut file_cache);
                        let _ = stdout_handle.flush();
                    }
                }
            } else {
                // If it's a new, unique record:
                // 1. Finalize and export the *previous* pending record, if it existed.
                if let Some(finalized_pending) = pending_record.take() {
                    export_record(&finalized_pending, &mut json_writer, &mut is_first_export_record);
                }

                // 2. The new record becomes the pending one.
                pending_first_ts = rec.ts;
                
                // 3. Print the new record on a new line. `is_update` is false.
                last_height = print_entry(&mut stdout_handle, &rec, &config, false, 0, pending_first_ts, &mut file_cache);
                let _ = stdout_handle.flush();
                
                // 4. Store it as the new pending record.
                pending_record = Some(rec);
            }

        } else {
            // Heap is empty, which means we've processed all historical logs.
            if !config.follow { break; }
            if !running.load(AtomicOrdering::SeqCst) { break; }
            
            // In follow mode, actively look for new records.
            let mut got_new_data = false;
            for (idx, iter_opt) in iterators.iter_mut().enumerate() {
                if let Some(iter) = iter_opt {
                    while let Some(rec) = iter.next_rec() {
                        heap.push(HeapEntry { record: rec, file_idx: idx });
                        got_new_data = true;
                    }
                }
            }

            if last_rescan.elapsed() > Duration::from_secs(2) {
                scan_files(&mut iterators, &mut heap, &mut known_paths);
                last_rescan = Instant::now();
            }

            // If after checking all files and rescanning we still have no new data, sleep.
            if !got_new_data && heap.is_empty() {
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    
    // Final print/export for the last pending record when the loop exits (e.g., via Ctrl+C).
    if let Some(last) = pending_record.take() {
        // If it was being updated interactively, we need to print its final state.
        if is_tty_follow && last.count > 1 {
            print_entry(&mut stdout_handle, &last, &config, true, last_height, pending_first_ts, &mut file_cache);
        }
        export_record(&last, &mut json_writer, &mut is_first_export_record);
    }
    
    let _ = stdout_handle.flush();
    
    if let Some(w) = &mut json_writer {
        if !is_first_export_record {
            write!(w, "\n]")?;
        } else {
            write!(w, "]")?;
        }
        let _ = w.flush();
    }

    Ok(())
}


fn export_record(
    rec: &BinaryLogRecord, 
    json_writer: &mut Option<BufWriter<File>>,
    is_first: &mut bool,
) {
    if let Some(w) = json_writer {
        if !*is_first {
            let _ = writeln!(w, ",");
        }

        let (lvl_name, _, _) = get_level_meta(rec.lvl);
        let clean_msg_raw = remove_style_tags(&rec.msg);

        let exc_string = if let Some(tb) = &rec.traceback {
            let mut s = String::new();
            s.push_str("Traceback (most recent call last):\n");
            for frame in &tb.frames {
                s.push_str(&format!("  File \"{}\", line {}, in {}\n", frame.filename, frame.lineno, frame.name));
            }
            s.push_str(&format!("{}: {}", tb.exc_type, tb.exc_message));
            Some(s)
        } else { None };

        let j = JsonLogRecord {
            ts: rec.ts, 
            lvl: lvl_name.to_string(), 
            app: rec.app_name.clone(), 
            pid: rec.pid, 
            msg: clean_msg_raw, 
            count: rec.count, 
            exc: exc_string, 
            context: rec.context.clone(), 
            file: None, 
            line: None
        };
        
        if let Ok(s) = serde_json::to_string(&j) { 
            let _ = write!(w, "{}", s); 
        }

        *is_first = false;
    }
}

fn format_traceback_for_cli(tb: &RawTraceback, cache: &mut FileCache) -> String {
    let mut out = String::with_capacity(1024);
    let is_light = { if let Ok(r) = crate::utils::CURRENT_THEME.read() { *r == Theme::Light } else { false } };

    let header = if is_light { "Traceback (most recent call last):".magenta().to_string() } 
                 else { "Traceback (most recent call last):".yellow().dimmed().to_string() };
    out.push_str(&format!("{}\n", header));

    for frame in &tb.frames {
        let fname = if is_light { frame.filename.blue().to_string() } else { frame.filename.blue().underline().to_string() };
        let line = if is_light { frame.lineno.to_string().purple().to_string() } else { frame.lineno.to_string().yellow().to_string() };
        let func = if is_light { frame.name.black().bold().to_string() } else { frame.name.cyan().to_string() };
        out.push_str(&format!("  File \"{}\", line {}, in {}\n", fname, line, func));

        if !cache.contains_key(&frame.filename) {
            if let Ok(content) = fs::read_to_string(&frame.filename) {
                cache.insert(frame.filename.clone(), content.lines().map(String::from).collect());
            } else {
                cache.insert(frame.filename.clone(), Vec::new());
            }
        }
        
        if let Some(lines) = cache.get(&frame.filename) {
            if frame.lineno > 0 && (frame.lineno as usize) <= lines.len() {
                let code_line = &lines[(frame.lineno - 1) as usize];
                let arrow = "-->".red().to_string();
                let code = if is_light { code_line.trim().black().bold().to_string() } else { code_line.trim().white().bold().to_string() };
                out.push_str(&format!("    {} {}\n", arrow, code));
            }
        }
    }
    
    out.push_str(&format!("{}: {}", tb.exc_type.red().bold(), tb.exc_message));
    out
}

fn print_entry(
    writer: &mut BufWriter<Stdout>,
    rec: &BinaryLogRecord, 
    config: &ReaderConfig,
    is_update: bool, 
    last_height: usize, 
    first_ts: f64,
    file_cache: &mut FileCache,
) -> usize {

    if config.json_output {
        let (lvl_name, _, _) = get_level_meta(rec.lvl);
        let clean_msg_raw = remove_style_tags(&rec.msg);
        let exc_string = if let Some(tb) = &rec.traceback {
            let mut s = String::new();
            s.push_str("Traceback (most recent call last):\n");
            for frame in &tb.frames {
                s.push_str(&format!("  File \"{}\", line {}, in {}\n", frame.filename, frame.lineno, frame.name));
            }
            s.push_str(&format!("{}: {}", tb.exc_type, tb.exc_message));
            Some(s)
        } else { None };
        let j = JsonLogRecord {
            ts: rec.ts, lvl: lvl_name.to_string(), app: rec.app_name.clone(), pid: rec.pid, msg: clean_msg_raw, 
            count: rec.count, exc: exc_string, context: rec.context.clone(), file: None, line: None
        };
        if let Ok(s) = serde_json::to_string(&j) { let _ = writeln!(writer, "{}", s); }
        return 1;
    }

    // Determine if we should move the cursor up. This only happens in interactive follow mode when updating a line.
    let is_tty_follow_update = std::io::stdout().is_terminal() && config.follow && is_update;

    if is_tty_follow_update && last_height > 0 {
        let _ = writer.execute(MoveUp(last_height as u16));
        let _ = writer.execute(Clear(ClearType::FromCursorDown));
    }
    
    let content_height = print_record_content(writer, rec, config.show_trace, first_ts, file_cache);
    
    // Always add a newline after printing a record's content to finalize the line.
    let _ = writeln!(writer);
    
    // The height for the next potential update is the height of the content plus the final newline.
    content_height + 1
}

fn print_record_content(
    writer: &mut BufWriter<Stdout>,
    rec: &BinaryLogRecord, 
    show_trace: bool, 
    first_ts: f64,
    file_cache: &mut FileCache
) -> usize {
    let is_light = { if let Ok(r) = crate::utils::CURRENT_THEME.read() { *r == Theme::Light } else { false } };

    let secs = rec.ts as i64;
    let nsecs = ((rec.ts - secs as f64) * 1_000_000_000.0) as u32;
    let dt = DateTime::from_timestamp(secs, nsecs).unwrap_or_default();
    let time_str = dt.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S.%3f").to_string();
    let (_, lvl_color, _) = get_level_meta(rec.lvl);
    
    let context_str = if let Some(ctx) = &rec.context {
        let mut s = String::new();
        let mut sorted_keys: Vec<&String> = ctx.keys().collect();
        sorted_keys.sort();
        for k in sorted_keys {
            if let Some(v) = ctx.get(k) {
                let safe_v = crate::utils::sanitize_input(v);
                if is_light {
                    s.push_str(&format!(" {}={}", k.black().dimmed(), safe_v.blue()));
                } else {
                    s.push_str(&format!(" {}={}", k.dimmed(), safe_v.cyan()));
                }
            }
        }
        s
    } else { String::new() };

    let suffix = if rec.count > 1 {
        let diff = rec.ts - first_ts;
        let time_fmt = if diff < 1.0 { format!("{:.0}ms", diff * 1000.0) } else { format!("{:.1}s", diff) };
        if is_light {
            format!(" (x{} │ {})", rec.count, time_fmt).black().dimmed().to_string() 
        } else {
            format!(" (x{} │ {})", rec.count, time_fmt).yellow().dimmed().to_string() 
        }
    } else { String::new() };
    
    let safe_msg_raw = crate::utils::sanitize_input(&rec.msg);
    let colored_msg = fast_colorize(&safe_msg_raw);
    let app_colored = color_by_name(&rec.app_name).bold();

    let time_colored = if is_light { time_str.black().dimmed() } else { time_str.cyan().dimmed() };

    let header_line = format!("{} │ {: <14} {: <9} │ {}{}{}", 
        time_colored, format!("[{}]", app_colored), lvl_color, colored_msg, context_str, suffix
    );
    // Use write! here, as the final newline is handled by the calling function `print_entry`
    let _ = write!(writer, "{}", header_line);
    let mut current_height = measure_text_height(&header_line);

    if let Some(tb) = &rec.traceback {
        // Add a newline to separate the main log from the traceback
        let _ = writeln!(writer);
        current_height += 1;
        if show_trace {
            let colored_exc = format_traceback_for_cli(tb, file_cache);
            let _ = write!(writer, "{}", colored_exc);
            current_height += measure_text_height(&colored_exc);
        } else {
            let last_line = format!("{}: {}", tb.exc_type.red().bold(), tb.exc_message);
            let _ = write!(writer, "  └── {}", last_line);
            current_height += 1;
        }
    }
    
    current_height
}

fn color_by_name(name: &str) -> ColoredString {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    let h = hasher.finish();
    match h % 6 {
        0 => name.cyan(), 1 => name.magenta(), 2 => name.green(), 3 => name.yellow(), 4 => name.blue(), _ => name.purple(),
    }
}