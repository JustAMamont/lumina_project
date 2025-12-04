use std::fs::{self, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::Path;
use crate::types::LogEntry;
use crate::utils::get_level_meta; 
use super::LogDriver;
use regex::Regex;
use once_cell::sync::Lazy;

static STYLE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\[([^|\]]+)\|([^\]]+)\]").unwrap()
});

pub struct TextFileDriver {
    pub writer: Option<BufWriter<std::fs::File>>,
    pub current_path: String,
    pub buffer_size: usize,
}

impl LogDriver for TextFileDriver {
    fn write(&mut self, entry: &LogEntry, repeat_count: usize, first_ts: f64, time_prefix: &str, time_nsecs: u32) {
        if !entry.to_file { return; }

        let (lvl_str, _, _) = get_level_meta(entry.level);
        let date_str = &time_prefix[0..10]; 
        let target_path = entry.path_template.replace("{date}", date_str);
        
        if self.current_path != target_path || self.writer.is_none() {
            if let Some(w) = &mut self.writer { let _ = w.flush(); }
            let path = Path::new(&target_path);
            if let Some(parent) = path.parent() { if !parent.exists() { let _ = fs::create_dir_all(parent); } }
            match OpenOptions::new().create(true).append(true).open(path) {
                Ok(f) => { 
                    self.writer = Some(BufWriter::with_capacity(self.buffer_size, f)); 
                    self.current_path = target_path; 
                },
                Err(e) => eprintln!("Lumina Error: Cannot open log file {}: {}", target_path, e),
            }
        }

        if let Some(w) = &mut self.writer {
            let clean_msg = STYLE_REGEX.replace_all(&entry.message, "$1");
            let suffix = if repeat_count > 0 {
                let diff = entry.timestamp - first_ts;
                let time_fmt = if diff < 1.0 { format!("{:.0}ms", diff * 1000.0) } else { format!("{:.1}s", diff) };
                format!(" (x{} | {})", repeat_count + 1, time_fmt)
            } else { String::new() };

            let res = writeln!(w, "{}.{:03} | {} | {}{}", time_prefix, time_nsecs, lvl_str, clean_msg, suffix)
                .and_then(|_| {
                    if entry.exc_type.is_some() {
                        let mut tb_str = String::new();
                        if let Some(frames) = &entry.trace_frames {
                            tb_str.push_str("Traceback (most recent call last):\n");
                            for frame in frames {
                                tb_str.push_str(&format!("  File \"{}\", line {}, in {}\n", frame.filename, frame.lineno, frame.name));
                            }
                        }
                        if let (Some(t), Some(m)) = (&entry.exc_type, &entry.exc_message) { 
                            tb_str.push_str(&format!("{}: {}", t, m)); 
                        }
                        writeln!(w, "{}", tb_str)?;
                    } 
                    Ok(())
                });
            if let Err(e) = res { eprintln!("Lumina Error: Failed to write to log file: {}", e); }
            if self.buffer_size == 0 { let _ = w.flush(); }
        }
    }

    fn flush(&mut self) {
        if let Some(w) = &mut self.writer { let _ = w.flush(); }
    }
}