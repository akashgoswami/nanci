import numpy as np
from tensorflow.keras.datasets import mnist
import os

def save_mnist_binary():
    # Download MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    os.makedirs('./mnist', exist_ok=True)

    # Save training images
    with open('./mnist/mnist_train_images.bin', 'wb') as f:
        f.write(x_train.astype(np.uint8).tobytes())
    
    # Save training labels
    with open('./mnist/mnist_train_labels.bin', 'wb') as f:
        f.write(y_train.astype(np.uint8).tobytes())
    
    # Save test images
    with open('./mnist/mnist_test_images.bin', 'wb') as f:
        f.write(x_test.astype(np.uint8).tobytes())
    
    # Save test labels
    with open('./mnist/mnist_test_labels.bin', 'wb') as f:
        f.write(y_test.astype(np.uint8).tobytes())
    
    # Print dataset information
    print("Dataset information:")
    print(f"Training images: {len(x_train)} ({len(x_train) * 28 * 28} bytes)")
    print(f"Training labels: {len(y_train)} bytes")
    print(f"Test images: {len(x_test)} ({len(x_test) * 28 * 28} bytes)")
    print(f"Test labels: {len(y_test)} bytes")

if __name__ == "__main__":
    save_mnist_binary()