import tensorflow as tf
import numpy as np
import cv2

test_image = r'/home/squashyhydra/sudoku-solver/image test/Screenshot 2024-08-19 082938.png'
model_path = r'/home/squashyhydra/sudoku-solver/mnist_model.keras'

# Load the pretrained MNIST model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def extract_cells_from_grid(thresh):
    # Find contours of the entire grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Assume the largest contour is the grid itself
    grid_contour = contours[0]
    
    # Approximate the grid contour to a rectangle
    epsilon = 0.1 * cv2.arcLength(grid_contour, True)
    grid_approx = cv2.approxPolyDP(grid_contour, epsilon, True)
    
    if len(grid_approx) == 4:
        # Apply perspective transformation to get a straightened grid
        src_pts = np.array([grid_approx[0], grid_approx[1], grid_approx[2], grid_approx[3]], dtype="float32")
        dst_pts = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        grid_image = cv2.warpPerspective(thresh, M, (450, 450))
    else:
        # Use the original image if grid approximation fails
        grid_image = thresh
    
    # Divide the grid into 81 cells (9x9)
    cells = []
    step_size = grid_image.shape[0] // 9
    for i in range(9):
        for j in range(9):
            cell = grid_image[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size]
            cell = cv2.resize(cell, (28, 28))
            cells.append(cell)
    
    return np.array(cells)

def predict_digits(cells, model, blank_threshold=50):
    digits = []
    for cell in cells:
        if np.mean(cell) < blank_threshold:  # If the cell is mostly blank
            digits.append(0)
        else:
            cell = cell.astype('float32') / 255.0
            cell = np.expand_dims(cell, axis=-1)
            prediction = model.predict(np.array([cell]))
            digit = np.argmax(prediction, axis=1)[0]
            digits.append(digit)
    return digits

def reconstruct_sudoku_grid(digits, grid_size=9):
    sudoku_grid = np.zeros((grid_size, grid_size), dtype=int)
    for i, digit in enumerate(digits):
        row = i // grid_size
        col = i % grid_size
        sudoku_grid[row, col] = digit
    return sudoku_grid

# Main execution
thresh = preprocess_image(test_image)
cells = extract_cells_from_grid(thresh)
digits = predict_digits(cells, model)
sudoku_grid = reconstruct_sudoku_grid(digits)

view = True
if view: print(f"Extracted Numbers:\n{digits}\n");print("Sudoku grid:\n", sudoku_grid)