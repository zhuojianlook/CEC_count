import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Title of the app
st.title('Honeycomb Cell Detection and Counting')

# Upload the image file
uploaded_file = st.file_uploader("Upload an image of the cell culture", type=['png', 'jpg', 'jpeg', 'tif'])

# Slider for thickness control
dilation_size = st.slider("Select Border Thickness (Dilation Iterations)", min_value=1, max_value=5, value=2)

# Slider for dilation size after the final closed gaps step
final_dilation_size = st.slider("Skeleton Dilation (Final Closed Gaps)", min_value=1, max_value=10, value=2)

# Function to enhance contrast
def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

# Function to ensure borders are connected
def ensure_connected_borders(image, dilation_size):
    _, borders = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)  # Small kernel for thin dilation
    borders = cv2.dilate(borders, kernel, iterations=dilation_size)
    closing_kernel = np.ones((4, 4), np.uint8)  # Larger kernel for closing gaps
    borders = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
    return borders

# Function to apply dilation to the final closed skeleton
def dilate_final_skeleton(skeleton, dilation_size):
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    return dilated_skeleton

# Function to detect and count honeycomb cells
def count_honeycomb_cells(image, dilated_skeleton):
    # Invert the image to detect closed regions (honeycomb cells)
    inverted_skeleton = cv2.bitwise_not(dilated_skeleton)

    contours, _ = cv2.findContours(inverted_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours and mark cells
    image_with_contours = image.copy()
    
    # Draw contours
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)  # Red contours
    
    # Count the number of cells
    cell_count = len(contours)
    
    return image_with_contours, cell_count

# Function to plot images
def plot_image(title, image, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

# Display the original and processed images
if uploaded_file is not None:
    # Read the image from the uploaded file
    image = np.array(Image.open(uploaded_file))
    
    # Step 1: Enhance contrast
    enhanced_image = enhance_contrast(image)
    plot_image("Original Image", enhanced_image)
    
    # Step 2: Ensure the borders are connected and thickened
    connected_borders = ensure_connected_borders(enhanced_image, dilation_size)
    plot_image("Connected Borders (Thickened)", connected_borders)
    
    # Step 3: Apply dilation to the final closed skeleton to control thickness
    dilated_skeleton = dilate_final_skeleton(connected_borders, final_dilation_size)
    plot_image("Final Closed Gaps with Dilation", dilated_skeleton)
    
    # Step 4: Count honeycomb cells and draw contours
    image_with_contours, cell_count = count_honeycomb_cells(image, dilated_skeleton)
    plot_image(f"Honeycomb Cells (Count: {cell_count})", image_with_contours, cmap=None)

    st.write(f"Number of detected honeycomb cells: {cell_count}")
