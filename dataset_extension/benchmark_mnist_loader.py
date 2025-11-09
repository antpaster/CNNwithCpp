import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import cpp_mnist_dataset_ext_many_workers as cpp_mnist_dataset_ext
# import cpp_mnist_dataset_ext_many_workers_openmp as cpp_mnist_dataset_ext
import cpp_mnist_dataset_ext_many_workers_openmp_with_aug as cpp_mnist_dataset_ext

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
    num_workers = 8
    num_batches = 500  # enough for stable timing

    print("Warming up and building torchvision dataset...")
    # Baseline Python
    torchvision_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    torch_time = benchmark_loader("torchvision ToTensor", torchvision_ds, batch_size, num_workers)

    # C++ extension
    print("\nC++ MNIST with augmentation (OpenMP)...")
    cpp_ds = cpp_mnist_dataset_ext.MNISTDataset(
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte"
    )

    cpp_ds.enable_augmentation(True)
    cpp_ds.set_pad(2)
    cpp_ds.set_flip_prob(0.5)
    cpp_ds.set_normalize(True)
    cpp_time = benchmark_loader("C++ OpenMP loader", cpp_ds, batch_size, num_workers)

    print(f"Speedup with augmentation: {torch_time / cpp_time:.2f}x")

if __name__ == "__main__":
    # torch.set_num_threads(1)
    main()
