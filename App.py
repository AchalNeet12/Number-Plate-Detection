import streamlit as st
import cv2
import imutils
import pytesseract
from PIL import Image
import numpy as np

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract path

def process_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smooth the image
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(gray_image, 30, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    screenCnt = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y+h, x:x+w]
            return new_img, True

    return image, False

# Streamlit app
st.title("Number Plate Detection")

uploaded_file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    result_image, found = process_image(image)

    if found:
        st.image(result_image, caption="Detected Number Plate", use_column_width=True)
        # Extract text from the number plate
        text = pytesseract.image_to_string(result_image, config='--psm 8')
        st.write("Detected Number Plate Text:")
        st.write(text)
    else:
        st.write("No number plate detected.")
