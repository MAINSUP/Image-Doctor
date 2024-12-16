from io import BytesIO
import img2pdf
import pillow_heif
import streamlit as st
from PIL import Image
from pillow_heif import open_heif, register_heif_opener

register_heif_opener()

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
            if imgfile.name.lower().endswith(('.heic', '.heif')):
                heif_file = pillow_heif.read_heif(imgfile)
                # Convert to PIL Image
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                )
                # Save the image as JPEG to a BytesIO buffer
                jpeg_buffer = BytesIO()
                image.save(jpeg_buffer, format="JPEG")
                jpeg_buffer.seek(0)
                pdf_buffer.write(img2pdf.convert(jpeg_buffer))
            pdf_buffer.write(img2pdf.convert(imgfile.read()))
            pdf_buffer.seek(0)  # Reset buffer pointer
            pdf_buffers.append(pdf_buffer)
            fnames.append(imgfile.name)
        except Exception as e:
            print(e) # st.error(f"Failed to convert image {imgfile.name}: {e}")
    return pdf_buffers, fnames


def convert_heic_to_jpeg(imgfile):
    """
    Converts a .heic or .heif image file to JPEG format.

    Parameters:
        imgfile: A file-like object (uploaded via Streamlit).

    Returns:
        A BytesIO object containing the JPEG image and the original file name.
    """
    try:
        # Check if the uploaded file is a .heic or .heif file
        if imgfile.name.lower().endswith(('.heic', '.heif')):
            # Open the HEIC/HEIF file using pillow_heif
            heif_file = open_heif(imgfile)
            # Ensure image mode is RGB (pillow_heif supports it natively)
            rgb_image = heif_file.to_pillow()
            # Save the RGB image to a BytesIO buffer as JPEG
            jpeg_buffer = BytesIO()
            rgb_image.save(jpeg_buffer, format="JPEG")
            jpeg_buffer.seek(0)  # Reset buffer pointer
            return jpeg_buffer, imgfile.name
        else:
            st.warning("Uploaded file is not a valid HEIC/HEIF image.")
            return None
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
               label="Download Generated PDF",
               data=buffer,
               file_name='{}.pdf'.format(name),
               mime="PDF/pdf"
           )

if right.button("Convert images from .heic to .jpg", use_container_width=True):
    for file in imgfiles:
        jpeg_buffer, fname = convert_heic_to_jpeg(file)
        st.download_button(
            label="Download Generated Image",
            data=jpeg_buffer,
            file_name='{}.jpg'.format(fname),
            mime="Image/jpg"
        )
        st.info("Converted images to JPG")

