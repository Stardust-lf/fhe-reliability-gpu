import subprocess
import re
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Optional: set library path if needed
# env = os.environ.copy()
# env["LD_LIBRARY_PATH"] = "../phantom-fhe/build/lib"


def run_test(log_dim, batch_size, num_flips, num_target_symbols):
    """
    Runs the ntt_test executable and parses bit and symbol error rates.
    Returns (bit_error, symbol_error) or (None, None) on failure.
    """
    try:
        result = subprocess.run(
            ["./ntt_test", str(log_dim), str(batch_size), str(num_flips), str(num_target_symbols)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
            # env=env
        )
        output = result.stdout
        bit_error_match = re.search(r"Bit error: \d+/\d+ = ([0-9.]+)", output)
        symbol_error_match = re.search(r"Symbol error: \d+/\d+ = ([0-9.]+)", output)
        if bit_error_match and symbol_error_match:
            return float(bit_error_match.group(1)), float(symbol_error_match.group(1))
    except subprocess.CalledProcessError as e:
        print(f"Execution failed: {e.output}")
    return None, None


def main():
    # Configuration
    log_dim = 12
    batch_size = 1
    trials = 100

    bits_range = range(1, 16)         # 1 to 15 bit flips in one symbol
    symbol_range = range(1, 8)       # 1 to 7 symbols flipped

    # Prepare tasks: (type, x, num_flips, num_target_symbols)
    tasks = []
    for bits in bits_range:
        for _ in range(trials):
            tasks.append(("flip_per_symbol", bits, bits, 1))
    for num_symbols in symbol_range:
        for _ in range(trials):
            tasks.append(("num_symbols", num_symbols, 1, num_symbols))

    # Ensure output folder
    os.makedirs("data", exist_ok=True)
    data_file = os.path.join("data", "flipimpact.csv")

    # Run tests in parallel and save to CSV
    with open(data_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["type", "x", "bit_error", "symbol_error"]);

        # Submit all tasks
        futures = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for t in tasks:
                _, x, num_flips, num_target_symbols = t
                fut = executor.submit(run_test, log_dim, batch_size, num_flips, num_target_symbols)
                futures[fut] = t

            # Collect results as they complete
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Running tests"):
                task_type, x, _, _ = futures[fut]
                bit_err, sym_err = fut.result()
                if bit_err is not None:
                    writer.writerow([task_type, x, bit_err, sym_err])

    print(f"Data saved to {data_file}")


if __name__ == "__main__":
    main()
