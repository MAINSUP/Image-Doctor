import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit_pdf_viewer import pdf_viewer

st.logo("pages/dse_logo.png", link="https://dse.enterprises", size="large")


def save_logo_temp(logo_file):
    """Save uploaded logo as a temporary file and return the path."""
    temp_folder = "temp_folder"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    temp_path = os.path.join(temp_folder, logo_file.name)
    with open(temp_path, "wb") as f:
        f.write(logo_file.getvalue())  # Convert UploadedFile to bytes
    return temp_path


def add_link_to_pdf(uploaded_pdf, link_inputs, logo_paths, selected_pages, x_coor, y_coor, width_size, height_size):
    """Add logo with a clickable link to a PDF."""

    # Validate input PDF
    reader = PdfReader(uploaded_pdf)
    if len(reader.pages) == 0:
        raise ValueError("The uploaded PDF has no pages.")

    writer = PdfWriter()

    # Create overlay with logos
    overlay_buffer = BytesIO()
    c = canvas.Canvas(overlay_buffer, pagesize=A4)

    for link_url, path, x, y, width, height in zip(link_inputs, logo_paths, x_coor, y_coor, width_size, height_size):
        c.drawImage(path, x, y, width=width, height=height)
        c.linkURL(link_url, (x, y, x + width, y + height), relative=1)

    c.save()

    # Read overlay PDF
    overlay_buffer.seek(0)
    overlay_reader = PdfReader(overlay_buffer)
    if len(overlay_reader.pages) == 0:
        return None  # Prevent error

    for page_num, page in enumerate(reader.pages):
        if page_num in selected_pages:
            overlay_page = overlay_reader.pages[0]
            page.merge_page(overlay_page)  # Ensure this is supported in your PyPDF2 version
        writer.add_page(page)

    output_pdf = BytesIO()
    writer.write(output_pdf)
    output_pdf.seek(0)
    return output_pdf


# ---- Streamlit UI ----

st.header("PDF Links Editor")

pdffiles = st.file_uploader("Upload your PDFs:", type="pdf", accept_multiple_files=True)

num_logos = st.slider("Number of logos:", 1, 10, 1)

if pdffiles:
    num_pages = []
    for obj in pdffiles:
        pdf_data = BytesIO(obj.getvalue())  # Ensure we get the full PDF data
        pdf_data.seek(0)  # Reset the pointer to the start
        if pdf_data.getbuffer().nbytes == 0:
            st.error("Uploaded PDF is empty. Please check the file and try again.")
            st.stop()
        read = PdfReader(pdf_data)  # Read the PDF correctly
        num_pages.append(len(read.pages))

    page_choices = [f"PDF {i + 1} - Page {j + 1}" for i, pages in enumerate(num_pages) for j in range(pages)]
    selected_pages = st.multiselect("Select page(s) for links:", page_choices, default=[page_choices[0]])

# ---- Logo & Link Inputs ----

logo_paths = []
link_inputs = []
x_coor, y_coor, width_size, height_size = [], [], [], []

for i in range(num_logos):
    logo = st.file_uploader(f"Upload logo {i + 1}", type=["png", "jpg", "jpeg"])
    if logo:
        temp_logo_path = save_logo_temp(logo)
        logo_paths.append(temp_logo_path)

    link_inputs.append(st.text_input(f"Enter link for logo {i + 1}"))
    x_coor.append(st.number_input(f"X position {i + 1}", 0, 600, 500, key=f"x_{i + 1}"))
    y_coor.append(st.number_input(f"Y position {i + 1}", 0, 800, 690, key=f"y_{i + 1}"))
    width_size.append(st.number_input(f"Width {i + 1}", 10, 200, 30, key=f"width_{i + 1}"))
    height_size.append(st.number_input(f"Height {i + 1}", 10, 100, 30, key=f"height_{i + 1}"))

# ---- PDF Processing ----

file_names, linked_pdfs, spages = [], [], []

if pdffiles and logo_paths:  # Only proceed if there are PDFs and logos
    for j, file in enumerate(pdffiles):
        pdf_data = BytesIO(file.read())

        for page_choice in selected_pages:
            pdf_index, page_number = page_choice.split(" - ")
            if int(pdf_index.split(" ")[1]) - 1 == j:
                spages.append(int(page_number.split(" ")[1]) - 1)

        modified_pdf = add_link_to_pdf(pdf_data, link_inputs, logo_paths, spages, x_coor, y_coor, width_size,
                                       height_size)

        if modified_pdf:
            linked_pdfs.append(modified_pdf)
            file_names.append(file.name)

# ---- PDF Viewer & Download ----

try:
    with st.sidebar:
        for pdf in linked_pdfs:
            binary_data = pdf.read()
            pdf_viewer(input=binary_data, width=300)

    for buffer, name in zip(linked_pdfs, file_names):
        st.download_button(
            label=f"Download {name}_linked.pdf",
            data=buffer,
            file_name=f"{name}_linked.pdf",
            mime="application/pdf"
        )
except AttributeError:
    pass