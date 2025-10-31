import torch
import time
import fused_add_relu_ext
import matplotlib.pyplot as plt

def benchmark(func, a, b, warmup=5, repeat=20):
    # Warm-up (to stabilize CPU cache and threading)
    for _ in range(warmup):
        func(a, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Timing loop
    start = time.perf_counter()
    for _ in range(repeat):
        func(a, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    return (end - start) / repeat

if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())  # use full CPU
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Extension loaded from: {fused_add_relu_ext.__file__}")

    sizes = [1 << i for i in range(10, 23)]  # 1K → 8M
    times_ref, times_ext, speedups = [], [], []

    for n in sizes:
        a = torch.randn(n, dtype=torch.float32)
        b = torch.randn_like(a)

        ref_func = lambda x, y: torch.nn.functional.relu(x + y)
        ext_func = lambda x, y: fused_add_relu_ext.fused_add_relu(x, y)

        t_ref = benchmark(ref_func, a, b)
        t_ext = benchmark(ext_func, a, b)
        times_ref.append(t_ref)
        times_ext.append(t_ext)
        speedups.append(t_ref / t_ext if t_ext > 0 else float("inf"))

        print(f"n={n:8d}: ref={t_ref*1e3:8.3f} ms | ext={t_ext*1e3:8.3f} ms | "
              f"speedup={t_ref/t_ext:5.2f}x")

    # Validate correctness on one sample
    a = torch.randn(10000)
    b = torch.randn_like(a)
    diff = (torch.nn.functional.relu(a + b)
            - fused_add_relu_ext.fused_add_relu(a, b)).abs().max()
    print(f"Max abs diff between reference and extension: {diff:.3e}")

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title("Benchmark: Fused Add+ReLU (C++ Extension vs PyTorch)")
    ax1.plot(sizes, [t * 1e3 for t in times_ref], 'o-', label="PyTorch ReLU(a+b)")
    ax1.plot(sizes, [t * 1e3 for t in times_ext], 's-', label="C++ Extension")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Tensor size (elements)")
    ax1.set_ylabel("Execution time (ms)")
    ax1.grid(True, which="both", ls="--", lw=0.5)
    ax1.legend(loc="upper left")

    # Secondary axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(sizes, speedups, '^-', color='tab:green', label="Speedup (x)")
    ax2.set_ylabel("Speedup (times faster)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
