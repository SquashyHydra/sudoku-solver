import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create directories for each digit
dataset_path = "sudoku_dataset"
font_path = f"fonts/Arial.ttf"

os.makedirs(dataset_path, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(dataset_path, str(i)), exist_ok=True)

# Function to generate synthetic digit images
def generate_sudoku_digit(digit, font_path=font_path, size=28, save_path=None):
    image = Image.new('L', (size, size), color=255)  # White background
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size - 4)
    w, h = draw.textsize(str(digit), font=font)
    draw.text(((size - w) / 2, (size - h) / 2), str(digit), fill=0, font=font)
    
    # Add random noise and distortions
    image = image.rotate(np.random.uniform(-10, 10), fillcolor=255)
    image = image.transform((size, size), Image.QUAD, (np.random.uniform(0, 5), np.random.uniform(0, 5),
                                                       size - np.random.uniform(0, 5), np.random.uniform(0, 5),
                                                       size - np.random.uniform(0, 5), size - np.random.uniform(0, 5),
                                                       np.random.uniform(0, 5), size - np.random.uniform(0, 5)),
                             fillcolor=255)
    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    if save_path:
        cv2.imwrite(save_path, image)
    
    return image

# Generate dataset
for digit in range(10):
    for i in range(1000):  # Generate 1000 samples per digit
        file_name = os.path.join(dataset_path, str(digit), f"{digit}_{i}.png")
        generate_sudoku_digit(digit, save_path=file_name)

print("Dataset generated successfully!")