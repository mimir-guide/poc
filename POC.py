import streamlit as st
from google.cloud import vision
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
import os
import hmac
from dataclasses import dataclass
import nest_asyncio

nest_asyncio.apply()

st.title("Mimir POC")

GCP_API_KEY = os.getenv("GCP_API_KEY")

if GCP_API_KEY is None:
    st.error("API_KEY not set")
    st.stop()


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


@dataclass
class NarrativeFeature:
    landmark: str
    language: str


class Narrative(BaseModel):
    story: str = Field(description="The story about the landmark")


def get_agent(api_key: str):
    model = GeminiModel("gemini-2.0-flash-exp", api_key=api_key)
    agent = Agent(
        model=model,
        deps_type=NarrativeFeature,
        result_type=Narrative,
        system_prompt="""
        You are a tour guide. You are telling the user about the history of the given landmark.
        You should tell the history in the manner of a story, and you can mix in a little bit of fiction.
        The goal is to teach the user about some history, while keep it entertaining.
        Keep the story short.""",
    )

    @agent.system_prompt
    def get_landmark(ctx: RunContext[NarrativeFeature]) -> str:
        return f"The landmark is {ctx.deps.landmark!r}."

    @agent.system_prompt
    def get_language(ctx: RunContext[NarrativeFeature]) -> str:
        return f"The story should be told in {ctx.deps.language!r}."

    return agent


def main(api_key: str):
    if image_file is not None:
        client = vision.ImageAnnotatorClient(client_options={"api_key": api_key})

        image = vision.Image(content=image_file.read())
        response: vision.AnnotateImageResponse = client.landmark_detection(image=image)

        st.image(image_file, use_container_width=True)

        for landmark in response.landmark_annotations:
            message = f"""
            ### {landmark.description} ({landmark.score:.3f})\n

            ({landmark.locations[0].lat_lng.latitude:.4f}, {landmark.locations[0].lat_lng.longitude:.4f})

            ---
            """

            st.markdown(message)

        if len(response.landmark_annotations) >= 1:
            agent = get_agent(api_key)
            st.warning("Select the values below slowly between selections")
            landmark = st.selectbox(
                "Landmark",
                [landmark.description for landmark in response.landmark_annotations],
            )
            language = st.selectbox(
                "Language", ["English", "Finnish", "Swedish", "Vietnamese", "Japanese"]
            )

            deps = NarrativeFeature(
                landmark=landmark,
                language=language,
            )
            result = agent.run_sync("Tell me about this landmark.", deps=deps)
            st.write(result.data.story)


if __name__ == "__main__":
    main(GCP_API_KEY)
