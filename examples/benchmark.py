import time
import logging
import statistics
import os
import shutil
from lumina import Lumina

ITERATIONS = 100_000
LOG_DIR = "logs/latency"

def setup_dirs():
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

def measure_latencies_std():
    """Measure how long the program HANGS on a logger.info call."""
    logger = logging.getLogger("lat_std")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.FileHandler(os.path.join(LOG_DIR, "std.log"))
    logger.addHandler(handler)
    
    latencies = []
    
    # Warmup
    for _ in range(1000): logger.info("warmup")
    
    for i in range(ITERATIONS):
        start = time.perf_counter()
        # At this moment, Python blocks and waits for the OS
        logger.info(f"User action {i} performed successfully") 
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000) # in microseconds
        
    handler.close()
    return latencies

def measure_latencies_lumina():
    """Measure how long the program HANGS on a lum.info call."""
    lum = Lumina.get_logger(
        name="lat_lum",
        path_template=os.path.join(LOG_DIR, "lum.ldb"),
        channel_capacity=ITERATIONS + 1000,
        text_enabled=False,
        db_enabled=True,
        capture_caller=False # Disable, as std logging doesn't resolve caller by default either
    )
    
    latencies = []
    
    # Warmup
    for _ in range(1000): lum.info("warmup", console=False)
    
    for i in range(ITERATIONS):
        start = time.perf_counter()
        # At this moment, Python copies data to the channel and returns instantly
        lum.info(f"User action {i} performed successfully", console=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000) # in microseconds

    lum.shutdown()
    Lumina._instance = None
    return latencies

def print_stats(name, data):
    p50 = statistics.median(data)
    p95 = statistics.quantiles(data, n=20)[18]
    p99 = statistics.quantiles(data, n=100)[98]
    p_max = max(data)
    
    print(f"\n📊 {name} STATISTICS (microseconds):")
    print(f"   Median (P50):  {p50:>6.2f} µs  (Typical case)")
    print(f"   Worst  (P99):  {p99:>6.2f} µs  (1% slowest calls)")
    print(f"   Spike  (Max):  {p_max:>6.2f} µs  (The single worst lag)")

    return p50, p99

if __name__ == "__main__":
    setup_dirs()
    print(f"⚡ APPLICATION RESPONSIVENESS TEST ({ITERATIONS} calls)")
    print("   Measuring how long the Main Thread is blocked per log call.\n")
    
    lat_std = measure_latencies_std()
    std_p50, std_p99 = print_stats("STANDARD Logging", lat_std)
    
    lat_lum = measure_latencies_lumina()
    lum_p50, lum_p99 = print_stats("LUMINA Logging", lat_lum)
    
    print("\n" + "="*60)
    print("🚀 RESPONSIVENESS GAIN:")
    print(f"   Typical speedup (P50): {std_p50 / lum_p50:.1f}x faster")
    print(f"   Stability gain  (P99): {std_p99 / lum_p99:.1f}x more stable")
    print("="*60)
    
    if lum_p99 < 10.0:
        print("✅ Lumina impact is negligible (< 10µs). Ideally suited for AsyncIO/HighLoad.")
    else:
        print("⚠️ Lumina impact is noticeable.")


# ⚡ APPLICATION RESPONSIVENESS TEST (100000 calls)
#    Measuring how long the Main Thread is blocked per log call.
# 
# 
# 📊 STANDARD Logging STATISTICS (microseconds):
#    Median (P50):   13.03 µs  (Typical case)
#    Worst  (P99):   48.42 µs  (1% slowest calls)
#    Spike  (Max):  4169.61 µs  (The single worst lag)
# 
# 📊 LUMINA Logging STATISTICS (microseconds):
#    Median (P50):    0.78 µs  (Typical case)
#    Worst  (P99):    3.29 µs  (1% slowest calls)
#    Spike  (Max):  2453.75 µs  (The single worst lag)
# 
# ============================================================
# 🚀 RESPONSIVENESS GAIN:
#    Typical speedup (P50): 16.7x faster
#    Stability gain  (P99): 14.7x more stable
# ============================================================
# ✅ Lumina impact is negligible (< 10µs). Ideally suited for AsyncIO/HighLoad.