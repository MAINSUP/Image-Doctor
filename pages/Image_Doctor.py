import streamlit as st
import os
import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage  # Replace PyQt5 with PySide6 if needed
import pymupdf # PyMuPDF
from scipy.ndimage import rotate
import cv2
from stqdm import stqdm
from io import BytesIO
import img2pdf


# st.set_page_config(page_title="📚 Image Doctor")
st.markdown("""<h1 style='text-align: center;'>🩺🖼️ Image Doctor</h1> 
<style>
div.stButton > button:first-child {
    height: 50px;
    width: 200px;
}
</style>
""", unsafe_allow_html=True)
# st.subheader("📚 Image Doctor")
container_pdf, container_chat = st.columns([50, 50])
padding = st.slider(
    "Specify image border padding value",
    min_value=0, max_value=50, value=10, step=5)
# on = st.toggle("Does your image contain noise?")
# if on:
#    noise = "y"
# else:
#    noise = "n"


def prepare_image_for_scipy(input_image):
    """
    Prepares an input image for scipy.ndimage processing using OpenCV.
    Automatically detects if input is a file path, QPixmap, or fitz.Pixmap.

    Parameters:
        input_image: str | QPixmap | fitz.Pixmap
            Path to an image file, a QPixmap object, or a fitz.Pixmap.

    Returns:
        numpy.ndarray:
            The image as a NumPy array compatible with scipy.ndimage.
    """
    # Check if the input is a file path
    if isinstance(input_image, str) and os.path.isfile(input_image):
        # Read the image using OpenCV (handles most formats)
        image_array = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        image = Image.open(input_image)
        if image_array is None:
            raise ValueError(f"Failed to read image from path: {input_image}")
    elif input_image is not None and hasattr(input_image, "read"):
        # Read the file as a byte array
        file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)

        # Decode the image using OpenCV
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        image = input_image
        if image_array is None:
            raise ValueError("Failed to decode image from uploaded file.")
    elif isinstance(input_image, QPixmap):
        # Convert QPixmap to QImage
        qimage = input_image.toImage()
        # Convert QImage to NumPy array
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)  # 3 bytes per pixel (RGB)
        image_array = np.array(ptr).reshape((height, width, 3))
    elif isinstance(input_image, pymupdf.Pixmap):
        # Convert fitz.Pixmap to NumPy array
        if input_image.n < 4:  # Grayscale or RGB
            image_array = np.frombuffer(input_image.samples, dtype=np.uint8).reshape(input_image.h, input_image.w, input_image.n)
        else:  # CMYK or other unsupported formats
            image_array = np.frombuffer(input_image.samples, dtype=np.uint8).reshape(input_image.h, input_image.w, 4)[:, :, :3]  # Convert to RGB
    else:
        raise ValueError("Input must be a file path, QPixmap, or fitz.Pixmap object.")
    return image, image_array


def detect_and_crop_with_padding(image, padding):
    # Read the image
    #image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection with adjusted thresholds
    edges = cv2.Canny(gray, 30, 100)

    # Dilate edges to enlarge contours
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours and filter by area
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Get the bounding box of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original.shape[1] - x, w + 2 * padding)
        h = min(original.shape[0] - y, h + 2 * padding)

        # Crop the image
        cropped_image = original[y:y+h, x:x+w]

        # Special negative padding
        neg_pad = -60
        x = max(0, x - neg_pad)
        y = max(0, y - neg_pad)
        w = min(original.shape[1] - x, w + 2 * neg_pad)
        h = min(original.shape[0] - y, h + 2 * neg_pad)
        extra_crop_img = original[y:y+h, x:x+w]
    else:
        cropped_image = original  # No contours found, return original image
    # Ensure the array is compatible with scipy.ndimage
    # cropped_image = np.ascontiguousarray(cropped_image, dtype=np.float32)

    return cropped_image, extra_crop_img


def good_image(cropped_image,extra_crop_img, noise):
    delta = 0.25
    limit = 5
    angles = np.arange(-limit, limit + delta, delta)
    print("Checking page alignment and removing any noise")
    scores = []
    wd, ht = extra_crop_img.size
    pix = np.array(extra_crop_img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)  # getting best score
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.format(best_angle))
    # correct skew
    (h, w) = cropped_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    if noise == "y":
        rotated = cv2.warpAffine(cropped_image, M, (wd, ht), flags=cv2.INTER_CUBIC,  # rotate image to the best angle
                                 borderMode=cv2.BORDER_REPLICATE)
        ret, mask = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.fastNlMeansDenoising(mask, None, 20, 20, 15)  # performing denoising
        kernel = np.array([[10, -1, 5],
                           [-1, -1, -1],
                           [5, -1, 10]])  # sharpening image according to kernel matrix
        sharp_img = cv2.filter2D(src=dst, ddepth=-1, kernel=kernel)  # manually tuned coefficient matrix
        # do adaptive threshold on gray image
        img_gray = cv2.cvtColor(sharp_img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 25)
        # make background of input white where thresh is white
        result = sharp_img.copy()
        result[thresh == 255] = (255, 255, 255)
        warped_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        result =  cv2.warpAffine(cropped_image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE) # rotate original image to the best angle
        warped_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, warped_rgb


def find_score(arr, angle):  # utility function to return array of skewness scores
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def prepare_cv2_image_for_img2pdf(cv2_image, output_format="JPEG"):
    """
    Prepares a cv2 image (NumPy array) for img2pdf by converting it to raw bytes.

    Parameters:
        cv2_image: NumPy array representing the image (from cv2).
        output_format: The desired image format for img2pdf (default: 'JPEG').

    Returns:
        A BytesIO object containing the image in the specified format.
    """
    from PIL import Image

    try:
        # Convert the cv2 image (BGR) to PIL format (RGB)
        pil_image = Image.fromarray(cv2_image)
        # Save the PIL image into a BytesIO buffer in the specified format
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format=output_format)
        img_buffer.seek(0)  # Reset buffer pointer
        return img_buffer
    except Exception as e:
        raise ValueError(f"Failed to process cv2 image for img2pdf: {e}")


imgfiles = st.file_uploader("Upload your images here:",
                            type=["jpeg", "jpg", "png", '.heic', '.heif', '.HEIC', '.HEIF'],
                            accept_multiple_files=True)

processed_bytes = []
names = []
img_bgr = []
for file in stqdm(imgfiles):
    origin_image, image_array = prepare_image_for_scipy(file)
    cropped_image, extra_crop  = detect_and_crop_with_padding(image_array, padding=padding)
    img_cropped = Image.fromarray((255 - 255 * cropped_image).astype("uint8")).convert("RGB")
    extra_crop_img = Image.fromarray((255 - 255 * extra_crop).astype("uint8")).convert("RGB")
    result, warped_rgb = good_image(cropped_image,extra_crop_img, noise=None)
    with st.sidebar:
        st.image([origin_image, warped_rgb], caption=["Original image", "Processed image"])
    _, buffer = cv2.imencode(".png", result)
    processed_bytes.append(BytesIO(buffer))
    names.append(origin_image.name)
    img_bgr.append(warped_rgb)

left, right= st.columns(2)
for name, buffer in zip(names, processed_bytes):
    # Provide a download button for the processed image
    left.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name='{}_corrected.png'.format(name),
        mime="image/png"
    )
for name, cv2_im in zip(names, img_bgr):
    # Provide a download button for the processed image
    img_buffer = prepare_cv2_image_for_img2pdf(cv2_im, output_format="JPEG")
    pdf_bytes = img2pdf.convert(img_buffer.getvalue())
    right.download_button(
        label="Download Processed Image as pdf",
        data=pdf_bytes,
        file_name='{}_corrected.pdf'.format(name),
        mime="PDF/pdf"
    )

