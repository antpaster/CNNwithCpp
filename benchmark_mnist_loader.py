import time
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import mnist_loader_ext

def time_dataloader(loader, iters=200):
    # measure time per batch over 'iters' mini-batches (or full epoch if smaller)
    start = time.perf_counter()
    it = iter(loader)
    for i in range(iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    return (end - start) / iters

# --- Standard torchvision pipeline ---
def benchmark_torchvision(batch_size=64, num_workers=4, samples=1000):
    transform = T.ToTensor() # default converts PIL->FloatTensor [0,1]
    ds = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    per_batch = time_dataloader(loader, iters=min(samples, len(loader)))
    return per_batch

# --- Fast loader (C++) pipeline ---
def benchmark_fastloader(images_path, labels_path, batch_size=64, num_workers=4, samples=1000):
    images, labels = mnist_loader_ext.load_mnist(images_path, labels_path, True, 0.1307, 0.3081)
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    per_batch = time_dataloader(loader, iters=min(samples, len(loader)))
    return per_batch

if __name__ == "__main__":
    # paths: replace with your downloaded idx files (from torchvision or MNIST website)
    train_images = "data/MNIST/raw/train-images-idx3-ubyte"
    train_labels = "data/MNIST/raw/train-labels-idx1-ubyte"

    print("Warming caches and building dataset (torchvision)...")
    t_tv = benchmark_torchvision(batch_size=64, num_workers=2, samples=200)
    print(f"torchvision ToTensor per-batch: {t_tv*1000:.3f} ms (batch_size=64, num_workers=2)")

    print("Now using C++ loader...")
    t_fast = benchmark_fastloader(train_images, train_labels, batch_size=64, num_workers=2, samples=200)
    print(f"C++ loader per-batch: {t_fast*1000:.3f} ms (batch_size=64, num_workers=2)")

    print(f"Speedup: {t_tv / t_fast:.2f}x")
