import streamlit as st
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import os
import hmac

st.title("Mimir POC")


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

image_file = st.file_uploader(
    label="Upload image",
    type=["jpg", "png", "jpeg"],
)


if image_file is not None:
    client = vision.ImageAnnotatorClient(
        client_options={"api_key": os.environ["GCP_API_KEY"]}
    )

    image = vision.Image(content=image_file.read())
    response: vision.AnnotateImageResponse = client.landmark_detection(image=image)

    with Image.open(image_file) as annotated_image:
        draw = ImageDraw.Draw(annotated_image)
        for landmark in response.landmark_annotations:
            polygon = [
                (vertex.x, vertex.y) for vertex in landmark.bounding_poly.vertices
            ]
            draw.polygon(
                xy=polygon,
                width=3,
                outline="red",
            )
            st.write(landmark.description)
            st.write(landmark.score)
            location = landmark.locations[0].lat_lng
            font = ImageFont.load_default(size=annotated_image.width // 32)
            draw.text(
                xy=polygon[0],
                text=f"{landmark.description}\n{landmark.score:.3f}\n({location.latitude:.4f}, {location.longitude:.4f})",
                fill="red",
                font=font,
            )
        st.image(annotated_image, use_container_width=True)
