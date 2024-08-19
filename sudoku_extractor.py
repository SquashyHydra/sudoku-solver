import numpy as np
import cv2
from threading import Thread

class viewImage(Thread):
    def __init__(self, thresh):
        Thread.__init__(self)
        self.setName = "ViewImage-Thread"
        self.image = thresh

    def show_image(self):
        cv2.imshow('Processed Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run(self):
        self.show_image()

test_image = r"C:/Users/squas/Pictures/screenshots/Screenshot 2024-08-19 082938.png"

import tensorflow as tf
import numpy as np
import cv2

# Load the pretrained MNIST model
model = tf.keras.models.load_model('path_to_your_mnist_model.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell = thresh[y:y+h, x:x+w]
        cell = cv2.resize(cell, (28, 28))
        cells.append(cell)
    
    return np.array(cells)

def predict_digits(cells, model):
    cells = cells.astype('float32') / 255.0
    cells = np.expand_dims(cells, axis=-1)
    predictions = model.predict(cells)
    digits = np.argmax(predictions, axis=1)
    return digits

def reconstruct_sudoku_grid(digits, grid_size=9):
    sudoku_grid = np.zeros((grid_size, grid_size), dtype=int)
    for i, digit in enumerate(digits):
        row = i // grid_size
        col = i % grid_size
        sudoku_grid[row, col] = digit
    return sudoku_grid

# Main execution
cells = preprocess_image('path_to_your_sudoku_image.png')
digits = predict_digits(cells, model)
sudoku_grid = reconstruct_sudoku_grid(digits)
print("Sudoku grid:\n", sudoku_grid)