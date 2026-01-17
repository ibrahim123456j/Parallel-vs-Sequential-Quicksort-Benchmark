import random
import time
import statistics as stats
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------
INSERTION_SORT_THRESHOLD = 32
PARALLEL_SIZE_THRESHOLD = 50_000
MAX_PARALLEL_DEPTH = 3
TRIALS = 5
INPUT_SIZES = [100_000, 500_000, 1_000_000, 2_000_000]

_GLOBAL_POOL = None

# ----------------------------
# Sorting Algorithms
# ----------------------------
def insertion_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def median_of_three(arr: List[int]) -> int:
    a0, a1, a2 = arr[0], arr[len(arr)//2], arr[-1]
    return sorted([a0, a1, a2])[1]

def quicksort_seq(arr: List[int]) -> List[int]:
    if len(arr) <= INSERTION_SORT_THRESHOLD:
        return insertion_sort(arr)
    pivot = median_of_three(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_seq(left) + mid + quicksort_seq(right)

def _worker(args: Tuple[List[int], int]) -> List[int]:
    return quicksort_par(*args)

def quicksort_par(arr: List[int], depth: int = 0) -> List[int]:
    if len(arr) <= INSERTION_SORT_THRESHOLD:
        return insertion_sort(arr)

    pivot = median_of_three(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    can_parallel = (
        depth < MAX_PARALLEL_DEPTH
        and len(arr) >= PARALLEL_SIZE_THRESHOLD
        and _GLOBAL_POOL is not None
    )

    if can_parallel:
        left_s, right_s = _GLOBAL_POOL.map(
            _worker,
            [(left, depth + 1), (right, depth + 1)]
        )
        return left_s + mid + right_s
    else:
        return quicksort_par(left, depth) + mid + quicksort_par(right, depth)

# ----------------------------
# Benchmarking
# ----------------------------
def benchmark(data: List[int]) -> Tuple[float, float]:
    start = time.perf_counter()
    seq_sorted = quicksort_seq(data)
    t_seq = time.perf_counter() - start

    start = time.perf_counter()
    par_sorted = quicksort_par(data, 0)
    t_par = time.perf_counter() - start

    assert seq_sorted == par_sorted
    return t_seq * 1000, t_par * 1000  # ms

# ----------------------------
# Main Execution
# ----------------------------
def main():
    results = defaultdict(lambda: {"seq": [], "par": []})
    processes = cpu_count()

    print(f"Using {processes} processes")

    # Predefine input arrays ONCE
    predefined_arrays = {
        n: [random.randint(0, 1_000_000) for _ in range(n)]
        for n in INPUT_SIZES
    }

    global _GLOBAL_POOL
    with Pool(processes=processes) as pool:
        _GLOBAL_POOL = pool

        for n in INPUT_SIZES:
            data = predefined_arrays[n]

            for t in range(TRIALS):
                t_seq, t_par = benchmark(data)
                results[n]["seq"].append(t_seq)
                results[n]["par"].append(t_par)

                print(
                    f"n={n} trial={t+1} "
                    f"seq={t_seq:.1f}ms par={t_par:.1f}ms"
                )

    # ----------------------------
    # Aggregation
    # ----------------------------
    sizes = []
    seq_means = []
    par_means = []
    speedups = []
    efficiencies = []

    for n in INPUT_SIZES:
        T1 = stats.mean(results[n]["seq"])
        Tp = stats.mean(results[n]["par"])

        sizes.append(n)
        seq_means.append(T1)
        par_means.append(Tp)

        Sp = T1 / Tp
        Ep = Sp / processes

        speedups.append(Sp)
        efficiencies.append(Ep)

    # ----------------------------
    # Charts
    # ----------------------------
    plt.figure()
    plt.plot(sizes, seq_means, marker='o', label="Sequential")
    plt.plot(sizes, par_means, marker='o', label="Parallel")
    plt.xlabel("Input Size (n)")
    plt.ylabel("Time (ms)")
    plt.title("Runtime vs Input Size")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(sizes, speedups, marker='o')
    plt.xlabel("Input Size (n)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Input Size")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(sizes, efficiencies, marker='o')
    plt.xlabel("Input Size (n)")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Input Size")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()