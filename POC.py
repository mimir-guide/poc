import os
from typing import List, Self
from dataclasses import dataclass

import streamlit as st
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent, RunContext
from google.cloud import texttospeech as tts
from PIL import Image, ImageDraw
import nest_asyncio

nest_asyncio.apply()

st.title("Mimir POC")
GCP_API_KEY = os.getenv("GCP_API_KEY")

language = st.selectbox(
    "Language", ["English", "Finnish", "Swedish", "Spanish", "Chinese", "Vietnamese"]
)


@dataclass
class DepContext:
    lang: str


class BoundingBox(BaseModel):
    x0: float = Field(
        ..., description="X coordinate of the top-left corner", ge=0.0, le=1.0
    )
    y0: float = Field(
        ..., description="Y coordinate of the top-left corner", ge=0.0, le=1.0
    )
    x1: float = Field(
        ..., description="X coordinate of the bottom-right corner", ge=0.0, le=1.0
    )
    y1: float = Field(
        ..., description="Y coordinate of the bottom-right corner", ge=0.0, le=1.0
    )

    @model_validator(mode="after")
    def check_coordinates(self) -> Self:
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("Invalid bounding box coordinates")
        return self

    def scale_coordinate(self, w_scale: int = 1, h_scale: int = 1) -> List[int]:
        return [
            int(self.x0 * w_scale),
            int(self.y0 * h_scale),
            int(self.x1 * w_scale),
            int(self.y1 * h_scale),
        ]


class Narative(BaseModel):
    landmark: str = Field(..., description="Landmark name")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    story: str = Field(..., description="A brief history of the landmark")


class Naratives(BaseModel):
    naratives: List[Narative] = Field(..., description="A list of detected landmarks")


language_codes = {
    "English": "en-US",
    "Finnish": "fi-FI",
    "Swedish": "sv-SE",
    "Spanish": "es-ES",
    "Chinese": "zh-CN",
    "Vietnamese": "vi-VN",
}

voices = {
    "English": "en-US-Chirp3-HD-Aoede",
    "Finnish": "fi-FI-Standard-A",
    "Swedish": "sv-SE-Standard-A",
    "Spanish": "es-ES-Chirp3-HD-Aoede",
    "Chinese": "cmn-CN-Chirp3-HD-Aoede",
    "Vietnamese": "vi-VN-Chirp3-HD-Aoede",
}


geo_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    deps_type=DepContext,
    result_type=Naratives,
    system_prompt="""
    You are a geolocation expert.
    Your task is to analyze the image and provide a list of landmarks with the coordinates
    of the bounding boxes where the landmarks are,
    and a brief history of each landmark.
    The bounding boxes should be as close as possible to the landmarks.
    The coordinates should be in the format [x0, y0, x1, y1], where (x0, y0) is the top-left corner,
    and (x1, y1) is the bottom-right corner. The coordinates should be normalized to the range [0, 1].
    The story should be upbeat, like a funny tour guide. Keep the story between 200 to 500 words.
    There should be details about the history, but not boring like a lecture.
    """,
)


@geo_agent.system_prompt
def set_language(ctx: RunContext[DepContext]) -> str:
    return f"The language for the narrative is {ctx.deps.lang}."


tts_client = tts.TextToSpeechClient(client_options={"api_key": GCP_API_KEY})

image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image is not None:
    result = geo_agent.run_sync(
        [
            "Tell me about the landmarks in this image. Locate the bounding boxes of the landmarks.",
            BinaryContent(image.read(), media_type="image/png"),
        ],
        deps=DepContext(lang=language),
    )

    with Image.open(image) as pil_image:
        canvas = ImageDraw.Draw(pil_image)
        w, h = pil_image.width, pil_image.height

        for landmark in result.data.naratives:
            box = landmark.bounding_box.scale_coordinate(w, h)
            canvas.rectangle(box, outline="red", width=5)

        st.image(pil_image)

    for i, landmark in enumerate(result.data.naratives):
        st.write(landmark.landmark)
        st.write(landmark.story)

        tts_response = tts_client.synthesize_speech(
            input=tts.SynthesisInput(text=landmark.story),
            voice=tts.VoiceSelectionParams(
                language_code=language_codes[language],
                name=voices[language],
            ),
            audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.OGG_OPUS),
        )
        st.audio(data=tts_response.audio_content, format="audio/wav", autoplay=i == 0)
