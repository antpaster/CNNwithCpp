import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cpp_mnist_dataset_ext_many_workers as cpp_mnist_dataset_ext
# import cpp_mnist_dataset_ext_many_workers_openmp as cpp_mnist_dataset_ext

def benchmark_loader(name, dataset, batch_size=64, num_workers=2, num_batches=500):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    it = iter(loader)
    # Warm up
    for _ in range(10):
        next(it)
    start = time.time()
    for i, batch in enumerate(it):
        if i >= num_batches:
            break
    elapsed = time.time() - start
    ms_per_batch = (elapsed / num_batches) * 1000
    print(f"{name} per-batch: {ms_per_batch:.3f} ms (batch_size={batch_size}, num_workers={num_workers})")
    return ms_per_batch

def main():
    batch_size = 64
    num_workers = 6
    num_batches = 500  # enough for stable timing

    print("Warming up and building torchvision dataset...")
    torchvision_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    torch_time = benchmark_loader("torchvision ToTensor", torchvision_ds, batch_size, num_workers)

    print("Now using C++ MNIST loader...")
    cpp_ds = cpp_mnist_dataset_ext.MNISTDataset(
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte"
    )

    cpp_time = benchmark_loader("C++ loader", cpp_ds, batch_size, num_workers)

    print(f"Speedup: {torch_time / cpp_time:.2f}x")

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
