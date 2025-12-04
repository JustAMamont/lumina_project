import time
import random
import math
import logging
from lumina import Lumina

# ==========================================
# 1. INITIALIZATION
# ==========================================
# Create the logger.
# capture_caller=True: Enables file:line detection (e.g., demo.py:45).
# theme="dark": Optimized color scheme for dark terminals.
lum = Lumina.get_logger(
    name="ProApp",          # Application name in logs
    retention_days=3,       # Auto-delete logs older than 3 days
    ui_refresh_rate_ms=50,  # Console UI smoothness
    flush_interval_ms=500,  # Write to disk every 0.5 seconds
    capture_caller=True,    # Show caller location
    theme="dark",           # Color theme
    colors_enabled=True
)

# Intercept standard Python logging (requests, urllib, django, etc.)
# This redirects all logging.info(), logging.warning() calls to Lumina.
lum.intercept_std_logging()

# Catch unhandled exceptions (prevents silent crashes)
lum.catch_crashes()


# ==========================================
# 2. DEMO FUNCTIONS
# ==========================================

def demo_context_logging():
    """Demonstrates passing structured data (context) to logs."""
    lum.info("--- 1. CONTEXT LOGGING ---")
    
    # Standard log
    lum.info("Server started")
    
    # Log with kwargs. These are not just text; they are stored as JSON fields.
    # In the console, they are highlighted with colors.
    lum.info("User logged in", user_id=42, ip="10.0.0.1", role="admin")
    
    # Semantic levels
    lum.success("Database connected", db="postgres", latency="2ms")
    lum.warning("Disk space low", disk="/var", free="5%")
    time.sleep(0.5)

def demo_std_logging_interception():
    """
    Demonstrates that standard Python logging calls are captured by Lumina.
    """
    lum.info("--- 2. STANDARD LOGGING INTERCEPTION ---")
    
    # We are using the standard 'logging' module here, NOT 'lum'
    logger = logging.getLogger("ExternalLib")
    
    # This will appear in Lumina's format because of lum.intercept_std_logging()
    logger.warning("This is a standard logging.warning() call!")
    logger.error("This is a standard logging.error() call!", extra={"lib_version": "1.0.0"})
    
    time.sleep(0.5)

def demo_profiler_silence():
    """
    Demonstrates filtering of small tasks.
    If a task finishes faster than 'min_duration_ms', it is NOT logged.
    This prevents log spam from micro-operations.
    """
    lum.info("--- 3. PROFILER: NOISE FILTERING ---")
    
    lum.info("Running micro-tasks (you won't see them)...")
    
    for i in range(5):
        # min_duration_ms=10: Log only if duration > 10ms.
        # We sleep 1ms. Log will be suppressed.
        with lum.profile(f"MicroTask {i}", min_duration_ms=10):
            time.sleep(0.001) 
            
    lum.success("Loop finished. Log is clean, no spam.")
    time.sleep(0.5)

def demo_profiler_slow():
    """
    Demonstrates slow task detection (Lag Detector).
    If a task is slower than 'slow_threshold_ms', it is logged as WARNING 🐌.
    """
    lum.info("--- 4. PROFILER: LAG DETECTOR ---")
    
    # We expect DB query to take 50ms. Threshold is 100ms.
    # Actual sleep is 200ms.
    # Expectation: Yellow log with a snail icon 🐌.
    with lum.profile("SQL Query", slow_threshold_ms=100, query="SELECT * FROM big_table"):
        time.sleep(0.2)
        
    lum.info("Notice the snail icon 🐌 and WARN level above.")
    time.sleep(0.5)

def demo_profiler_cpu_bound():
    """
    Simulates heavy computation.
    Metrics to observe:
    - cpu: Should be high (near 100% of a core).
    - ctx_forced: May increase (OS interrupting the greedy process).
    """
    lum.info("--- 5. PROFILER: CPU BOUND ---")
    
    with lum.profile("Calculating Primes", limit=5000):
        # Burn CPU cycles with math
        count = 0
        for num in range(2, 20000):
            if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
                count += 1
                
    time.sleep(0.5)

def demo_profiler_io_bound():
    """
    Simulates waiting (Network/Disk).
    Metrics to observe:
    - cpu: Near 0% (we are sleeping).
    - ctx_wait: HIGH (voluntary context switches).
      This indicates the program is waiting for I/O, not lagging itself.
    """
    lum.info("--- 6. PROFILER: I/O WAIT (NET/DISK) ---")
    
    with lum.profile("Download File", url="http://example.com/big.iso"):
        time.sleep(0.3) # Simulate download latency
        
    time.sleep(0.5)

def demo_profiler_memory():
    """
    Simulates memory allocation spikes.
    Metrics to observe:
    - ram: Shows the delta (+XX MB).
    - pg_faults: HIGH (Page Faults).
      This indicates the OS had to allocate physical pages for the data.
    """
    lum.info("--- 7. PROFILER: MEMORY SPIKE ---")
    
    # Allocate memory
    with lum.profile("Allocating Big List"):
        big_list = [random.random() for _ in range(5_000_000)]
        
    # Free memory to be nice
    del big_list 
    time.sleep(0.5)

def demo_exception():
    """Demonstrates pretty Traceback formatting."""
    lum.info("--- 8. ERROR HANDLING ---")
    
    try:
        lum.info("Dividing by zero...")
        x = 1 / 0
    except Exception as e:
        # exc=e automatically extracts the traceback, colorizes it,
        # and displays the file:line context.
        lum.error("Operation failed", exc=e, variable="x")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    lum.info("🚀 STARTING LUMINA DEMO")
    time.sleep(0.5)
    
    demo_context_logging()
    demo_std_logging_interception()
    demo_profiler_silence()
    demo_profiler_slow()
    demo_profiler_cpu_bound()
    demo_profiler_io_bound()
    demo_profiler_memory()
    demo_exception()
    
    lum.success("🏁 Demo finished. Logs flushed to disk.")


#============================================================
#⚡ BENCHMARK: Application Responsiveness Test
#   Measuring how long the main application thread is blocked by each log call.
#   Lower numbers mean the application remains more responsive under load.
#============================================================
#
#--- SCENARIO 1: SINGLE-THREADED LOGGING ---
#   A single thread writing 400,000 log messages.
#
#============================================================
#📋 RESULTS: SINGLE-THREADED MODE
#
#📊 STANDARD Logger STATISTICS (microseconds):
#   Median (P50):   27.16 µs  (Typical latency)
#   Tail   (P99):   40.57 µs  (1% slowest calls)
#   Spike  (Max): 18189.22 µs  (The single worst pause)
#
#📊 LUMINA Logger STATISTICS (microseconds):
#   Median (P50):    3.77 µs  (Typical latency)
#   Tail   (P99):    7.61 µs  (1% slowest calls)
#   Spike  (Max): 5168.93 µs  (The single worst pause)
#
#──────────────────── 🚀 PERFORMANCE GAIN ────────────────────
#   Typical Speedup (P50): 7.2x faster
#   Stability Gain (P99): 5.3x more stable (fewer spikes)
#
#✅ Lumina's impact is negligible. Main thread pauses are minimal.
#============================================================
#
#--- SCENARIO 2: MULTI-THREADED LOGGING (CONTENTION) ---
#   4 threads writing a total of 400,000 messages to the SAME file.
#   This test shows how the logger handles contention and locking.
#
#============================================================
#📋 RESULTS: MULTI-THREADED MODE (CONTENTION)
#
#📊 STANDARD Logger STATISTICS (microseconds):
#   Median (P50):   37.78 µs  (Typical latency)
#   Tail   (P99): 4104.31 µs  (1% slowest calls)
#   Spike  (Max): 29234.91 µs  (The single worst pause)
#
#📊 LUMINA Logger STATISTICS (microseconds):
#   Median (P50):    3.77 µs  (Typical latency)
#   Tail   (P99):   14.74 µs  (1% slowest calls)
#   Spike  (Max): 87876.70 µs  (The single worst pause)
#
#──────────────────── 🚀 PERFORMANCE GAIN ────────────────────
#   Typical Speedup (P50): 10.0x faster
#   Stability Gain (P99): 278.5x more stable (fewer spikes)
#============================================================
#
#🏁 BENCHMARK COMPLETE
