import torch
import time
import fused_add_relu_ext

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

    sizes = [
        1 << 10,   # 1K elements
        1 << 15,   # 32K
        1 << 18,   # 256K
        1 << 20,   # 1M
        1 << 22,   # 4M
    ]

    for n in sizes:
        a = torch.randn(n, dtype=torch.float32)
        b = torch.randn_like(a)

        ref_func = lambda x, y: torch.nn.functional.relu(x + y)
        ext_func = lambda x, y: fused_add_relu_ext.fused_add_relu(x, y)

        t_ref = benchmark(ref_func, a, b)
        t_ext = benchmark(ext_func, a, b)

        speedup = t_ref / t_ext if t_ext > 0 else float("inf")
        print(f"n={n:8d}:  torch F.relu(a+b): {t_ref*1e3:8.3f} ms  |  "
              f"ext: {t_ext*1e3:8.3f} ms  |  speedup: {speedup:5.2f}x")

    # Validate correctness on one sample
    a = torch.randn(10000)
    b = torch.randn_like(a)
    diff = (torch.nn.functional.relu(a + b)
            - fused_add_relu_ext.fused_add_relu(a, b)).abs().max()
    print(f"Max abs diff between reference and extension: {diff:.3e}")
