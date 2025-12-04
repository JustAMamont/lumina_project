import time
import logging
import statistics
import os
import shutil
import threading
from lumina import Lumina

ITERATIONS = 100000  # Per thread
NUM_THREADS = 4      # Number of threads
TOTAL_ITERATIONS = ITERATIONS * NUM_THREADS
LOG_DIR = "logs/latency"

def setup_dirs():
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

# --- Standard Logger ---
def std_log_worker(logger, latencies, barrier):
    """Worker function for standard logger threads."""
    barrier.wait()  # Sync start
    for i in range(ITERATIONS):
        start = time.perf_counter()
        # This call acquires a lock, causing threads to wait for each other.
        logger.info(f"User action {i} performed by thread {threading.get_ident()}")
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)

def measure_latencies_std(multi_threaded=False):
    """Measures latency for Python's standard logging.Handler."""
    logger = logging.getLogger("lat_std")
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Ensure no old handlers
    
    path_suffix = 'mt' if multi_threaded else 'st'
    handler = logging.FileHandler(os.path.join(LOG_DIR, f"std_{path_suffix}.log"))
    logger.addHandler(handler)
    
    latencies = []
    
    # Warm-up
    for _ in range(1000): logger.info("warmup")
    
    if not multi_threaded:
        for i in range(TOTAL_ITERATIONS):
            start = time.perf_counter()
            logger.info(f"User action {i} performed successfully")
            end = time.perf_counter()
            latencies.append((end - start) * 1_000_000)
    else:
        threads = []
        barrier = threading.Barrier(NUM_THREADS)
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=std_log_worker, args=(logger, latencies, barrier))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    handler.close()
    logger.removeHandler(handler)
    return latencies

# --- Lumina Logger ---
def lumina_log_worker(lum, latencies, barrier):
    """Worker function for Lumina logger threads."""
    barrier.wait() # Sync start
    for i in range(ITERATIONS):
        start = time.perf_counter()
        # This call is non-blocking; it just sends data to a channel.
        lum.info(f"User action {i} from thread {threading.get_ident()}", console=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)

def measure_latencies_lumina(multi_threaded=False):
    """Measures latency for Lumina's async logger."""
    if Lumina._instance:
        Lumina._instance.shutdown()
    Lumina._instance = None
    
    path_suffix = 'mt' if multi_threaded else 'st'
    lum = Lumina.get_logger(
        name="lat_lum",
        path_template=os.path.join(LOG_DIR, f"lum_{path_suffix}.ldb"),
        channel_capacity=50000,
        text_enabled=False, db_enabled=True, capture_caller=False
    )
    
    latencies = []
    
    # Warm-up
    for _ in range(1000): lum.info("warmup", console=False)
    
    if not multi_threaded:
        for i in range(TOTAL_ITERATIONS):
            start = time.perf_counter()
            lum.info(f"User action {i} performed successfully", console=False)
            end = time.perf_counter()
            latencies.append((end - start) * 1_000_000)
    else:
        threads = []
        barrier = threading.Barrier(NUM_THREADS)
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=lumina_log_worker, args=(lum, latencies, barrier))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    lum.shutdown()
    Lumina._instance = None
    return latencies

def print_stats(name, data):
    p50 = statistics.median(data)
    p99 = statistics.quantiles(data, n=100)[98]
    p_max = max(data)
    
    print(f"\n📊 {name} STATISTICS (microseconds):")
    print(f"   Median (P50): {p50:>7.2f} µs  (Typical latency)")
    print(f"   Tail   (P99): {p99:>7.2f} µs  (1% slowest calls)")
    print(f"   Spike  (Max): {p_max:>7.2f} µs  (The single worst pause)")
    return p50, p99

def compare_and_print(name, lat_std, lat_lum):
    print("\n" + "="*60)
    print(f"📋 RESULTS: {name.upper()}")
    std_p50, std_p99 = print_stats("STANDARD Logger", lat_std)
    lum_p50, lum_p99 = print_stats("LUMINA Logger", lat_lum)
    
    print("\n" + "─"*20 + " 🚀 PERFORMANCE GAIN " + "─"*20)
    if lum_p50 > 0: print(f"   Typical Speedup (P50): {std_p50 / lum_p50:.1f}x faster")
    if lum_p99 > 0: print(f"   Stability Gain (P99): {std_p99 / lum_p99:.1f}x more stable (fewer spikes)")
    
    if lum_p99 < 10.0:
        print("\n✅ Lumina's impact is negligible. Main thread pauses are minimal.")
    print("="*60 + "\n")

if __name__ == "__main__":
    setup_dirs()
    print("="*60)
    print(f"⚡ BENCHMARK: Application Responsiveness Test")
    print(f"   Measuring how long the main application thread is blocked by each log call.")
    print(f"   Lower numbers mean the application remains more responsive under load.")
    print("="*60 + "\n")
    
    # --- Single-Threaded Test ---
    print("--- SCENARIO 1: SINGLE-THREADED LOGGING ---")
    print(f"   A single thread writing {TOTAL_ITERATIONS:,} log messages.")
    
    lat_std_st = measure_latencies_std(multi_threaded=False)
    lat_lum_st = measure_latencies_lumina(multi_threaded=False)
    compare_and_print("Single-Threaded Mode", lat_std_st, lat_lum_st)
    time.sleep(1)

    # --- Multi-Threaded Test ---
    print("--- SCENARIO 2: MULTI-THREADED LOGGING (CONTENTION) ---")
    print(f"   {NUM_THREADS} threads writing a total of {TOTAL_ITERATIONS:,} messages to the SAME file.")
    print(f"   This test shows how the logger handles contention and locking.")

    lat_std_mt = measure_latencies_std(multi_threaded=True)
    lat_lum_mt = measure_latencies_lumina(multi_threaded=True)
    compare_and_print("Multi-Threaded Mode (Contention)", lat_std_mt, lat_lum_mt)

    print("🏁 BENCHMARK COMPLETE")