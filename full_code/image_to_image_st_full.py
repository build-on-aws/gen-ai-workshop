import base64
import io
import json
import os

import boto3
import streamlit as st
from PIL import Image

REGION = "us-east-1"

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def image_to_base64(img) -> str:
    """Convert a PIL Image or local image file path to a base64 string for Amazon Bedrock"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        print("Converting PIL Image to base64 string")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")


# Turn base64 string to image with PIL
def base64_to_pil(base64_string):
    """
    Purpose:
        Turn base64 string to image with PIL
    Args/Requests:
         base64_string: base64 string of image
    Return:
        image: PIL image
    """

    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image


def nova_update_image(change_prompt, init_image_b64):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Nova
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """

    body = json.dumps(
        {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": change_prompt,
                "images": [init_image_b64],
                "similarityStrength": 0.7,  # Range: 0.2 to 1.0
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 8.0,
            },
        }
    )

    model_id = "amazon.nova-canvas-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


def update_image_pipeline(user_image, change_prompt, model):

    # Turn image to base64 string
    init_image_b64 = image_to_base64(user_image)

    if model == "Amazon Nova":
        updated_image = nova_update_image(change_prompt, init_image_b64)

    # convert updated_image to PIL image
    updated_image = base64_to_pil(updated_image)

    # save updated image
    updated_image.save("updated_image.png")

    return updated_image


st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Generation Demo - Image to Image")

model = st.selectbox("Select model", ["Amazon Nova"])

# TODO insert your comments
# Add a file uploader for images
user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
change_prompt = st.text_input(
    "Enter a prompt to change the image",
    "",
)

col1, col2 = st.columns(2)  # Column 1 for input image, Column 2 for output image

# show user image
if user_image is not None:
    user_image = Image.open(user_image)
    col1.image(user_image)
    # Button to generate new image
    if col1.button("Update Image"):
        new_image = update_image_pipeline(user_image, change_prompt, model)
        col2.image(new_image)
else:
    col2.write("No image uploaded")
