import json

import boto3
import streamlit as st

st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Generation Demo")

REGION = "us-west-2"

# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


# Bedrock api call to stable diffusion
def generate_image_sd(text, style):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


def generate_image_titan(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Titan
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "textToImageParams": {"text": text},
        "taskType": "TEXT_IMAGE",
        "imageGenerationConfig": {
            "cfgScale": 10,
            "seed": 0,
            "quality": "standard",
            "width": 512,
            "height": 512,
            "numberOfImages": 1,
        },
    }

    body = json.dumps(body)

    modelId = "amazon.titan-image-generator-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


model = st.selectbox("Select model", ["Amazon Titan", "Stable Diffusion"])
prompt = st.text_input("Enter prompt")
# List of Stable Diffusion Preset Styles
sd_presets = [
    "None", "3d-model", "analog-film", "anime", "cinematic", "comic-book",
    "digital-art", "enhance", "fantasy-art", "isometric", "line-art",
    "low-poly", "modeling-compound", "neon-punk", "origami", "photographic",
    "pixel-art", "tile-texture"
]

# Select box for styles (only if Stable Diffusion is selected)
if model == "Stable Diffusion":
    style = st.selectbox("Select Style", sd_presets)
else:
    style = "None"
import base64
from PIL import Image
from io import BytesIO

def base64_to_pil(base64_str):
    """
    Converts a base64 string to a PIL Image object.
    """
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image
if st.button("Generate Image"):
    if model == "Amazon Titan":
        image_base64 = generate_image_titan(prompt)
    else:
        image_base64 = generate_image_sd(prompt, style)
    # Convert base64 to PIL Image
    image = base64_to_pil(image_base64)
    
    # Display the image
    st.image(image)
