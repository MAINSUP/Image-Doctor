from contextlib import contextmanager
import streamlit as st

st.logo(
    "pages/dse_logo.png",
    link="https://dse.enterprises",
    size="large"
    # icon_image=LOGO_URL_SMALL,
)
# st.set_page_config(page_title="📚 Image Doctor")
st.markdown("""<h1 style='text-align: center;'>📚 Image Doctor</h1> <style>
div.stButton > button:first-child {
    height: 50px;
    width: 200px;
}
</style>""", unsafe_allow_html=True)
st.subheader("Chose application to start")
container_pdf, container_chat = st.columns([50, 50])

HORIZONTAL_STYLE = """
<style class="hide-element">
    /* Hides the style container and removes the extra spacing */
    .element-container:has(.hide-element) {
        display: none;
    }
    /*
        The selector for >.element-container is necessary to avoid selecting the whole
        body of the streamlit app, which is also a stVerticalBlock.
    */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) {
        display: flex;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: baseline;
    }
    /* Buttons and their parent container all have a width of 704px, which we need to override */
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) div {
        width: max-content !important;
    }
    /* Just an example of how you would style buttons, if desired */
    /*
    div[data-testid="stVerticalBlock"]:has(> .element-container .horizontal-marker) button {
        border-color: red;
    }
    */
</style>
"""

@contextmanager
def st_horizontal():
    st.markdown(HORIZONTAL_STYLE, unsafe_allow_html=True)
    with st.container():
        st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
        yield


with st_horizontal():
    app1 = st.button("Image Doctor")
    app2 = st.button("PDF Editor")
    app3 = st.button("Image Convertor")

    if app1:
        st.switch_page("pages/Image_Doctor.py")
    if app2:
        st.switch_page("pages/PDF_Editor.py")
    if app3:
        st.switch_page("pages/Image_Convertor.py")
