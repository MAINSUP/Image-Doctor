import streamlit as st
from io import BytesIO
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from pikepdf import PdfImage
from stqdm import stqdm
from streamlit_pdf_viewer import pdf_viewer
import pikepdf
from PIL import UnidentifiedImageError
import traceback


compressed_pdfs = []
names = []

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
    height: 50px;
}
div.stButton > button:hover {
    background-color: #00ff00;
    height: 50px;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


def merge_pdf(sorted_files):
    """
    Merges multiple PDF files into a single PDF.

    Parameters:
        sorted_files: List of tuples (file name, file content) in the desired order.

    Returns:
        BytesIO object containing the merged PDF.
    """
    pdf_merge = PdfMerger()
    try:
        for _, pdf_content in sorted_files:
            pdf_merge.append(PdfReader(BytesIO(pdf_content)))

        # Save the merged PDF to a BytesIO object
        merged_pdf_buffer = BytesIO()
        pdf_merge.write(merged_pdf_buffer)
        pdf_merge.close()
        merged_pdf_buffer.seek(0)  # Reset buffer pointer
        return merged_pdf_buffer
    except Exception as e:
        raise ValueError(f"Failed to merge PDFs: {e}")

def reduce_pdfimage_quality(pdffile, quality):
    output_pdf = BytesIO()
    try:
        with pikepdf.open(pdffile) as pdf:
            for page in pdf.pages:
                for image_key, raw_image in page.images.items():
                    try:
                        # Access the raw image stream
                        rawimage = page.images[image_key]  # The raw object/dictionary
                        pdfimage = PdfImage(rawimage)
                        # Read the raw bytes from the image stream

                        try:
                            pillowimage = pdfimage.as_pil_image()
                            # Convert to RGB if necessary (some images may be in CMYK or other formats)
                            if pillowimage.mode != "RGB":
                                pillowimage = pillowimage.convert("RGB")

                            # Compress the image using Pillow
                            compressed_image_buffer = BytesIO()
                            pillowimage.save(compressed_image_buffer, format="JPEG", quality=quality)
                            compressed_image_buffer.seek(0)
                            # Replace the image stream in the PDF
                            pdfimage.obj.write(compressed_image_buffer.getvalue())  # Extract raw bytes
                        except UnidentifiedImageError:
                            st.warning(f"Failed to process image (unsupported format): {image_key}")
                            continue
                    except Exception as image_error:
                        st.warning(f"Failed to process an image: {traceback.format_exc()}")

            # Save the modified PDF to a BytesIO buffer
            pdf.save(output_pdf)
        output_pdf.seek(0)  # Reset buffer pointer
        return output_pdf
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {traceback.format_exc()}")
        return None


def lossless_pdf_compression(pdffiles):
    """
        Compresses the content streams of PDF files losslessly.

        Parameters:
            pdffiles: A list of uploaded file-like objects (e.g., from Streamlit's uploader).

        Returns:
            A list of BytesIO objects, each containing a processed PDF.
        """
    processed_pdfs = []
    names = []
    for pdf in stqdm(pdffiles):
        try:
            # Read the input PDF
            reader = PdfReader(pdf)
            writer = PdfWriter()

            # Compress each page
            for page in reader.pages:
                page.compress_content_streams()  # Compress the content stream
                writer.add_page(page)  # Add the page to the writer

            # Write the compressed PDF to a BytesIO buffer
            compressed_pdf_buffer = BytesIO()
            writer.write(compressed_pdf_buffer)
            compressed_pdf_buffer.seek(0)  # Reset buffer pointer
            processed_pdfs.append(compressed_pdf_buffer)
            names.append(file.name)
        except Exception as e:
            st.error(f"Failed to process a PDF: {e}")
    return processed_pdfs, names


container_pdf, container_chat = st.columns([50, 50])
st.header("You are at PDF Editor")

pdffiles = st.file_uploader("Upload your PDFs here:", type="pdf", accept_multiple_files=True)
if pdffiles:
    # Store file names and content in a list
    file_list = [(file.name, file.getvalue()) for file in pdffiles]

    # Sidebar to display files and arrange them
    st.sidebar.header("Arrange Files for Merging")
    uploaded_file_names = [file[0] for file in file_list]

    # Multiselect to arrange files
    sorted_options = st.sidebar.multiselect(
        "Drag to reorder files for merging",
        uploaded_file_names,
        default=uploaded_file_names  # Show in the original order
    )

    # Create the sorted file list based on user selection
    sorted_files = [file for file in file_list if file[0] in sorted_options]

    if sorted_files and len(sorted_files) == len(file_list):  # Ensure all files are selected
        st.write("Ready to merge the following PDFs (in selected order):")
        for name, _ in sorted_files:
            st.write(f"- {name}")

_, centre, __ = st.columns(3)
centre.button("Reset", type="primary", use_container_width=True)
# options = st.sidebar.multiselect(
#        "Arrange files for merging",
#        file_list, file_list)
quality = centre.slider(
        "Specify quality value",
        min_value=50, max_value=300, value=80, step=5)
left, middle, right = st.columns(3)
if left.button("Merge PDF files", f'<span class="big-font">Big Font Button</span>',icon="🖇️", use_container_width=True):
    merged_pdf_buffer = merge_pdf(sorted_files)
    left.markdown("Merged")
    # Provide a download button for the processed files
    st.download_button(
            label="Download Merged PDFs",
            data=merged_pdf_buffer,
            file_name='merged.pdf',
            mime="PDF/pdf"
        )

if middle.button("Reduce PDF size by processing images", icon="🎞", use_container_width=True):

    for file in pdffiles:
        compressed_pdf = reduce_pdfimage_quality(file, quality=quality)
        compressed_pdfs.append(compressed_pdf)
        names.append(file.name)
        for name, buffer in zip(names, compressed_pdfs):
            # Provide a download button for the processed files
            st.download_button(
                label="Download Processed PDFs",
                data=buffer,
                file_name='{}_resized.pdf'.format(name),
                mime="PDF/pdf"
            )

    middle.markdown("Ruduced file size")
if right.button("Perform lossless PDF compression", icon="🗜️", use_container_width=True):
    processed_pdfs, fnames = lossless_pdf_compression(pdffiles)
    for buffer, name in zip(processed_pdfs,fnames):
        st.download_button(
            label="Download Processed PDFs",
            data=buffer,
            file_name='{}_resized.pdf'.format(name),
            mime="PDF/pdf"
        )
    right.markdown("Ruduced file size")


