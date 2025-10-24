from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from PIL import Image
import os

# Create folder to store exported test images
os.makedirs("test_images", exist_ok=True)

(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Save first 20 test images as PNG
for i in range(20):
    img = Image.fromarray(x_test[i])
    img.save(f"test_images/img_{i}.png")

print("Saved 20 Fashion MNIST test images inside 'test_images' folder!")
