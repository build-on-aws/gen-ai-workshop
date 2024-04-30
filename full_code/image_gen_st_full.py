import streamlit as st
import boto3
import json
import base64
import io
from PIL import Image

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


# Bedrock api call to stable diffusion
def generate_image_sd(bedrock_client, text, style):
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

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results


def generate_image_titan(bedrock_client, text):
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

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


def initialize_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name=REGION)


def get_prompt_and_style():
    style = st.selectbox("Select style", sd_presets)
    prompt = st.text_input("Enter prompt")
    return prompt, style


def get_prompt():
    prompt = st.text_input("Enter prompt")
    return prompt


def main():

    bedrock_client = initialize_bedrock_client()

    model = st.selectbox("Select model", ["Stable Diffusion", "Amazon Titan"])

    if model == "Amazon Titan":
        prompt = get_prompt()
    elif model == "Stable Diffusion":
        prompt, style = get_prompt_and_style()
    else:
        st.error("Please select a model")

    if st.button("Generate"):

        if model == "Stable Diffusion":

            results = generate_image_sd(bedrock_client, prompt, style)
            img = convert_base64_to_image(results)
            img.save("image.png")
            st.image(img)
        elif model == "Amazon Titan":
            results = generate_image_titan(bedrock_client, prompt)
            img = convert_base64_to_image(results)
            img.save("image.png")
            st.image(img)


if __name__ == "__main__":
    main()
