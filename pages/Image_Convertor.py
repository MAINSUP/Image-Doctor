from io import BytesIO
import streamlit as st
from PIL import Image
import img2pdf
import numpy as np
from pillow_heif import open_heif

st.logo(
    "pages/dse_logo.png",
    link="https://dse.enterprises",
    size="large"
    # icon_image=LOGO_URL_SMALL,
)
container_pdf, container_chat = st.columns([50, 50])
st.header("You are at Image Convertor")


def convert_img_to_pdf(imgfiles):
    """
    Converts uploaded image files to individual PDFs.

    Parameters:
        imgfiles: List of image file-like objects (uploaded via Streamlit).

    Returns:
        A list of BytesIO objects containing the generated PDFs.
    """
    pdf_buffers = []  # To store PDF data for each image
    fnames = []
    for imgfile in imgfiles:
        try:
            # Convert image to PDF
            pdf_buffer = BytesIO()
            pdf_buffer.write(img2pdf.convert(imgfile.read()))
            pdf_buffer.seek(0)  # Reset buffer pointer
            pdf_buffers.append(pdf_buffer)
            fnames.append(imgfile.name)
        except Exception as e:
            st.error(f"Failed to convert image {imgfile.name}: {e}")
    return pdf_buffers, fnames


def convert_heic_to_jpeg(imgfile):
    """
    Converts a .heic or .heif image file to JPEG format.

    Parameters:
        imgfile: A file-like object (uploaded via Streamlit).

    Returns:
        A BytesIO object containing the JPEG image.
    """
    try:
        # Open the HEIC/HEIF file
        heif_file = open_heif(imgfile, convert_hdr_to_8bit=False, bgr_mode=True)
        # Convert to a NumPy array and then to a PIL Image
        np_array = np.asarray(heif_file)
        pil_image = Image.fromarray(np_array)
        # Save the image as JPEG to a BytesIO buffer
        jpeg_buffer = BytesIO()
        pil_image.save(jpeg_buffer, format="JPEG")
        jpeg_buffer.seek(0)  # Reset buffer pointer
        return jpeg_buffer, imgfile.name
    except Exception as e:
        st.error(f"Failed to process file {imgfile.name}: {e}")
        return None


imgfiles = st.file_uploader("Upload your images here:",
                            type=["jpeg", "jpg", "png", '.heic', '.heif', '.HEIC', '.HEIF'],
                            accept_multiple_files=True)
_, centre, __= st.columns(3)
centre.button("Reset", type="primary", use_container_width=True)
left, right = st.columns(2)
if left.button("Convert images to pdf", use_container_width=True):
       st.info("Convertion of images")
       pdf_buffers, fnames = convert_img_to_pdf(imgfiles)
       for buffer, name in zip(pdf_buffers, fnames):
           # Provide a download button for the processed files
           st.download_button(
               label="Download Generated PDFs",
               data=buffer,
               file_name='{}.pdf'.format(name),
               mime="PDF/pdf"
           )

if right.button("Convert images from .heic to .jpg", use_container_width=True):
    for file in imgfiles:
        jpeg_buffer, fname = convert_heic_to_jpeg(file)
        st.download_button(
            label="Download Generated Images",
            data=jpeg_buffer,
            file_name='{}.jpg'.format(fname),
            mime="Image/jpg"
        )
        st.info("Convertion of images")

