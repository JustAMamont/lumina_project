use crate::types::{LogEntry, BinaryLogRecord, RawTraceback};
use super::LogDriver;
use std::io::{Write, Result as IoResult, Seek};
use std::fs::{File, OpenOptions};
use std::path::Path;
use bincode::Options;
use byteorder::{WriteBytesExt, LittleEndian};
use crc32fast::Hasher;
use fs2::FileExt;

const MAGIC_HEADER: &[u8; 4] = b"LUM1";
const FILE_WRITE_BUFFER_SIZE: usize = 256 * 1024; // 256KB buffer

pub struct LuminaDbDriver {
    buffer: Vec<BinaryLogRecord>,
    current_path: String,
    pid: u32,
    writer: Option<std::io::BufWriter<File>>,
}

impl LuminaDbDriver {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(2000),
            current_path: String::new(),
            pid: std::process::id(),
            writer: None,
        }
    }

    /// Opens a new file or switches to an existing one.
    fn open_or_switch_file(&mut self, path_str: &str) -> IoResult<()> {
        // 1. Flush and close the previous writer if it exists.
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            if let Ok(file) = writer.into_inner() {
                file.unlock()?;
            }
        }

        // 2. Create parent directories if they don't exist.
        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            if !parent.exists() { std::fs::create_dir_all(parent)?; }
        }

        // 3. Open the file with options for appending and creating.
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true) // Open in append mode
            .open(path)?;

        // 4. Lock the file for exclusive access from this process.
        file.lock_exclusive()?;

        let mut writer = std::io::BufWriter::with_capacity(FILE_WRITE_BUFFER_SIZE, file);

        // 5. If the file is new (position is 0), write the magic header.
        if writer.stream_position()? == 0 {
            writer.write_all(MAGIC_HEADER)?;
        }

        self.writer = Some(writer);
        self.current_path = path_str.to_string();

        Ok(())
    }

    fn flush_to_disk(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        let writer = match &mut self.writer {
            Some(w) => w,
            None => return,
        };

        let my_options = bincode::DefaultOptions::new().with_little_endian().with_fixint_encoding();
        let raw_data = match my_options.serialize(&self.buffer) {
            Ok(d) => d,
            Err(e) => { eprintln!("Lumina Serialize Error: {}", e); return; },
        };
        
        let compressed_data = lz4_flex::compress_prepend_size(&raw_data);
        
        let mut hasher = Hasher::new();
        hasher.update(&compressed_data);
        let checksum = hasher.finalize();
        
        let compressed_len = compressed_data.len() as u32;

        // Write block: [CRC32][LENGTH][DATA]
        let write_res = writer.write_u32::<LittleEndian>(checksum)
            .and_then(|_| writer.write_u32::<LittleEndian>(compressed_len))
            .and_then(|_| writer.write_all(&compressed_data));
        
        if let Err(e) = write_res {
            eprintln!("Lumina disk write error: {}", e);
        }

        self.buffer.clear();
    }
}

impl Drop for LuminaDbDriver {
    fn drop(&mut self) {
        self.flush_to_disk();
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.flush();
            if let Ok(file) = writer.into_inner() {
                let _ = file.unlock();
            }
        }
    }
}

impl LogDriver for LuminaDbDriver {
    fn write(&mut self, entry: &LogEntry, repeat_count: usize, _first_ts: f64, time_prefix: &str, _time_nsecs: u32) {
        if !entry.to_file { return; }

        let date_str = &time_prefix[0..10];
        let target_path_str = entry.path_template.replace("{date}", date_str);

        if self.current_path != target_path_str {
            self.flush_to_disk();
            if let Err(e) = self.open_or_switch_file(&target_path_str) {
                eprintln!("Lumina IO Error ({}): {}", target_path_str, e);
                self.writer = None; // Disable writing if file can't be opened
                return;
            }
        }

        let traceback = if let (Some(exc_type), Some(exc_message), Some(frames)) = 
            (&entry.exc_type, &entry.exc_message, &entry.trace_frames) {
            Some(RawTraceback {
                exc_type: exc_type.clone(),
                exc_message: exc_message.clone(),
                frames: frames.clone(),
            })
        } else {
            None
        };

        let rec = BinaryLogRecord {
            ts: entry.timestamp,
            lvl: entry.level,
            app_name: entry.app_name.to_string(),
            pid: self.pid,
            msg: entry.message.clone(),
            traceback,
            context: entry.context.clone(), 
            count: (repeat_count + 1) as u32,
        };

        self.buffer.push(rec);
    }

    fn flush(&mut self) {
        self.flush_to_disk();
        if let Some(writer) = &mut self.writer {
            let _ = writer.flush();
        }
    }
}