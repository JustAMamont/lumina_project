use crate::types::LogEntry;

pub trait LogDriver: Send {
    fn write(&mut self, entry: &LogEntry, repeat_count: usize, first_ts: f64, time_prefix: &str, time_nsecs: u32);
    fn flush(&mut self);
}

pub mod text;
pub mod binary;