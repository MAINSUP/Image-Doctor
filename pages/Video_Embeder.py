import streamlit as st
import pymupdf
import io
import os
import cv2
from PIL import Image
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer

# JavaScript for playing embedded video
jscript = """
try {
    this.exportDataObject({ cName: 'video.mp4', nLaunch: 2 });
} catch (err) {
    app.alert('Error opening video: ' + err);
}
"""


def save_video_temp(video_file):
    """Save uploaded video as a temporary file and return the path."""
    temp_folder = "temp_folder"
    os.makedirs(temp_folder, exist_ok=True)
    temp_path = os.path.join(temp_folder, video_file.name)
    with open(temp_path, "wb") as f:
        f.write(video_file.getvalue())
    return temp_path


def extract_first_frame(video_path):
    """Extracts the first frame of the video and returns it as a PIL image."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def embed_video_in_pdf(pdf_bytes, video_paths, page_numbers, x_coords, y_coords, widths, heights):
    """
    Embeds videos into a PDF and adds interactive buttons.
    """
    doc = pymupdf.open("pdf", pdf_bytes)

    for j, (video_path, page_num, x, y, width, height) in enumerate(
            zip(video_paths, page_numbers, x_coords, y_coords, widths, heights)):
        page = doc[page_num]

        # Extract first frame and insert into PDF
        frame_img = extract_first_frame(video_path)
        if frame_img:
            frame_img_path = f"temp_frame_{j + 1}.png"
            frame_img.save(frame_img_path)
            img_rect = pymupdf.Rect(x, y, x + width, y + height)
            page.insert_image(img_rect, filename=frame_img_path)

        # Embed video file in PDF
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
        doc.embfile_add(f"video_{j + 1}.mp4", video_data)

        # Create a widget (button) for video playback
        widget = pymupdf.Widget()
        widget.field_type = pymupdf.PDF_WIDGET_TYPE_BUTTON
        widget.field_flags = pymupdf.PDF_BTN_FIELD_IS_PUSHBUTTON
        widget.rect = pymupdf.Rect(x, y, x + width, y + height)  # Position and size
        widget.script = jscript  # JavaScript for launching video
        widget.field_name = f"Play Video {j + 1}"
        widget.fill_color = (0, 0, 1)  # Blue button for visibility
        page.add_widget(widget)  # Add the button to the page

    # Save modified PDF to a BytesIO object
    output_pdf = io.BytesIO()
    doc.save(output_pdf)
    output_pdf.seek(0)
    return output_pdf


def main():
    st.title("PDF Video Embedder")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    num_videos = st.slider("Number of videos:", 1, 5, 1)

    video_paths, page_numbers, x_coords, y_coords, widths, heights = [], [], [], [], [], []

    if pdf_file:
        pdf_data = io.BytesIO(pdf_file.getvalue())
        reader = PdfReader(pdf_data)
        num_pages = len(reader.pages)

        for i in range(num_videos):
            with st.expander(f"Video {i + 1} Settings"):
                video = st.file_uploader(f"Upload video {i + 1}", type=["mp4", "avi", "mov"], key=f"video_{i}")
                if video:
                    video_paths.append(save_video_temp(video))
                page_numbers.append(st.number_input(f"Page Number", 1, num_pages, 1, key=f"page_{i}") - 1)
                x_coords.append(st.number_input(f"X position", 0, 600, 100, key=f"x_{i}"))
                y_coords.append(st.number_input(f"Y position", 0, 800, 500, key=f"y_{i}"))
                widths.append(st.number_input(f"Width", 10, 200, 100, key=f"width_{i}"))
                heights.append(st.number_input(f"Height", 10, 100, 75, key=f"height_{i}"))

        if st.button("Embed Videos and Download PDF"):
            modified_pdf = embed_video_in_pdf(pdf_file.getvalue(), video_paths, page_numbers, x_coords, y_coords,
                                              widths, heights)
            with st.sidebar:
                binary_data = modified_pdf.read()
                pdf_viewer(input=binary_data, width=300)
            st.download_button(
                label="Download PDF with Videos",
                data=modified_pdf,
                file_name="video_embedded.pdf",
                mime="application/pdf"
            )


if __name__ == "__main__":
    main()
