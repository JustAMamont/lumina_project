// Only import PyO3 types if building the extension
#[cfg(feature = "extension-module")]
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

const MAGIC_HEADER: &[u8; 4] = b"LUM1";
const FILE_BUFFER_SIZE: usize = 128 * 1024; 
const STDOUT_BUFFER_SIZE: usize = 128 * 1024; 

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

// === Python Wrapper (Only compiled for extension) ===
#[cfg(feature = "extension-module")]
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

// === Pure Rust Implementation (Compiled for both) ===
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
    
    // Counter to track if we actually showed anything
    let mut logs_displayed_count = 0; 

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

        for path in candidates {
            if !paths_set.contains(&path) {
                // Debug info if not in JSON/Follow mode
                if !config.json_output && !config.follow {
                    eprintln!("📄 Found log file: {:?}", path); 
                }
                
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
    
    if known_paths.is_empty() && !config.follow { 
        if !config.json_output { eprintln!("⚠️ No log files found in current directory/subdirectories."); } 
        return Ok(()); 
    }

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

            logs_displayed_count += 1;

            let is_duplicate = if let Some(pending) = &pending_record {
                pending.lvl == rec.lvl && pending.msg == rec.msg && pending.traceback.is_none() && rec.traceback.is_none() && pending.app_name == rec.app_name && pending.context == rec.context
            } else {
                false
            };

            if is_duplicate {
                if let Some(pending) = pending_record.as_mut() {
                    pending.count += rec.count;
                    pending.ts = rec.ts;
                    if is_tty_follow {
                        last_height = print_entry(&mut stdout_handle, pending, &config, true, last_height, pending_first_ts, &mut file_cache);
                        let _ = stdout_handle.flush();
                    }
                }
            } else {
                if let Some(finalized_pending) = pending_record.take() {
                    export_record(&finalized_pending, &mut json_writer, &mut is_first_export_record);
                }
                pending_first_ts = rec.ts;
                last_height = print_entry(&mut stdout_handle, &rec, &config, false, 0, pending_first_ts, &mut file_cache);
                let _ = stdout_handle.flush();
                pending_record = Some(rec);
            }

        } else {
            if !config.follow { break; }
            if !running.load(AtomicOrdering::SeqCst) { break; }
            
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

            if !got_new_data && heap.is_empty() {
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    
    if let Some(last) = pending_record.take() {
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

    if logs_displayed_count == 0 && !config.json_output && !config.follow {
        eprintln!("\n⚠️  Found {} files, but 0 records matched your filters.", known_paths.len());
        if config.start_ts.is_some() {
            eprintln!("   Hint: Try removing time filters (-m/-H/-D) or ensure logs were written recently.");
        }
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

    let is_tty_follow_update = std::io::stdout().is_terminal() && config.follow && is_update;

    if is_tty_follow_update && last_height > 0 {
        let _ = writer.execute(MoveUp(last_height as u16));
        let _ = writer.execute(Clear(ClearType::FromCursorDown));
    }
    
    let content_height = print_record_content(writer, rec, config.show_trace, first_ts, file_cache);
    
    let _ = writeln!(writer);
    
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
    let _ = write!(writer, "{}", header_line);
    let mut current_height = measure_text_height(&header_line);

    if let Some(tb) = &rec.traceback {
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


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::types::{BinaryLogRecord, LogEntry};
    use crate::engine::flush_binary_buffer;
    use crossbeam_channel::bounded;

    fn write_test_file(path: &Path, records: Vec<BinaryLogRecord>) {
        let (writer_tx, writer_rx) = bounded(10);
        let path_str = path.to_str().unwrap().to_string();

        let writer_thread = std::thread::spawn({
            let path_clone = path.to_path_buf();
            move || {
                let mut file = OpenOptions::new().create(true).write(true).open(&path_clone).unwrap();
                file.write_all(MAGIC_HEADER).unwrap();
                for msg in writer_rx.iter() {
                    if let crate::types::WriterMsg::BinaryBlock { data, .. } = msg {
                        file.write_all(&data).unwrap();
                    }
                }
            }
        });

        // The flush logic requires a LogEntry to know the target path.
        let last_log = LogEntry {
             app_name: "".into(), timestamp: 0.0, level: 0, message: "".to_string(),
             path_template: path_str.clone(), to_console: false, to_file: true, rate_limit: 0.0,
             exc_type: None, exc_message: None, exc_hash: 0, trace_frames: None, context: None, context_hash: 0, signal_shutdown: false,
        };
        
        // We pass the records directly to flush_binary_buffer for this test setup.
        let mut buf = records;
        flush_binary_buffer(&mut buf, &last_log, &writer_tx);
        drop(writer_tx); // Close the channel to unblock the writer thread.
        writer_thread.join().unwrap();
    }

    #[test]
    fn test_reader_scatter_gather_merge() {
        let dir = tempdir().unwrap();
        
        // File 1 with interleaved timestamps.
        let path1 = dir.path().join("proc1.ldb");
        let records1 = vec![
            BinaryLogRecord { ts: 100.0, msg: "p1_msg1".to_string(), count: 1, ..Default::default() },
            BinaryLogRecord { ts: 300.0, msg: "p1_msg2".to_string(), count: 1, ..Default::default() },
        ];
        write_test_file(&path1, records1);

        // File 2 with interleaved timestamps.
        let path2 = dir.path().join("proc2.ldb");
        let records2 = vec![
            BinaryLogRecord { ts: 200.0, msg: "p2_msg1".to_string(), count: 1, ..Default::default() },
            BinaryLogRecord { ts: 400.0, msg: "p2_msg2".to_string(), count: 1, ..Default::default() },
        ];
        write_test_file(&path2, records2);
        
        let export_path = dir.path().join("merged.json");
        let pattern = dir.path().join("*.ldb").to_str().unwrap().to_string();
        
        let config = ReaderConfig {
            file_pattern: Some(pattern),
            export_json: Some(export_path.to_str().unwrap().to_string()),
            ..Default::default()
        };
        
        run_reader_impl(config).unwrap();
        
        let result_json = fs::read_to_string(export_path).unwrap();
        let records: Vec<serde_json::Value> = serde_json::from_str(&result_json).unwrap();
        
        assert_eq!(records.len(), 4, "There should be 4 records after merging");
        
        // Check for strict chronological order.
        assert_eq!(records[0]["msg"], "p1_msg1");
        assert_eq!(records[1]["msg"], "p2_msg1");
        assert_eq!(records[2]["msg"], "p1_msg2");
        assert_eq!(records[3]["msg"], "p2_msg2");
        
        let timestamps: Vec<f64> = records.iter().map(|r| r["ts"].as_f64().unwrap()).collect();
        assert_eq!(timestamps, vec![100.0, 200.0, 300.0, 400.0]);
    }
    
    // Add Default for BinaryLogRecord to simplify creation in tests.
    impl Default for BinaryLogRecord {
        fn default() -> Self {
            Self {
                ts: 0.0,
                lvl: 20,
                app_name: "default_app".to_string(),
                pid: 0,
                msg: "".to_string(),
                traceback: None,
                context: None,
                count: 1,
            }
        }
    }

    // Default for ReaderConfig to simplify test setup.
    impl Default for ReaderConfig {
        fn default() -> Self {
            Self {
                file_pattern: None, min_level: 0, show_trace: false,
                json_output: false, follow: false, target_levels: None,
                grep: None, trace_type: None, start_ts: None, theme: None, export_json: None
            }
        }
    }
}