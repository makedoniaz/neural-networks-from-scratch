from .idx_dataset import IDXDataset

class FashionMNISTDataset(IDXDataset):
    """Fashion MNIST dataset loader using IDX format files.

    The Fashion MNIST dataset contains 70,000 grayscale images of fashion items
    (10 classes) at 28x28 pixel resolution.
    """

    CLASS_NAMES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]