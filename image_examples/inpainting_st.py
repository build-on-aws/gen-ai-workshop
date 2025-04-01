import base64
import io
import json
import os

import boto3
import streamlit as st
from PIL import Image, ImageOps

REGION = "us-east-1"

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def inpaint_mask(img, box):
    """Generates a segmentation mask for inpainting"""
    img_size = img.size
    assert len(box) == 4  # (left, top, right, bottom)
    assert box[0] < box[2]
    assert box[1] < box[3]
    return ImageOps.expand(
        Image.new(mode="RGB", size=(box[2] - box[0], box[3] - box[1]), color="black"),
        border=(box[0], box[1], img_size[0] - box[2], img_size[1] - box[3]),
        fill="white",
    )


def gen_mask_from_image(user_image):
    img_size = user_image.size
    box = (
        (img_size[0] - 300) // 2,
        img_size[1] - 300,
        (img_size[0] + 300) // 2,
        img_size[1] - 200,
    )

    # Mask
    mask = inpaint_mask(user_image, box)
    return mask


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


# Bedrock api call to stable diffusion
def sd_inpaint_image(change_prompt, init_image_b64, mask):
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
        "text_prompts": ([{"text": change_prompt, "weight": 1.0}]),
        "cfg_scale": 10,
        "init_image": init_image_b64,
        "mask_source": "MASK_IMAGE_BLACK",
        "mask_image": image_to_base64(mask),
        "seed": 0,
        "start_schedule": 0.6,
        "steps": 50,
    }

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


def titan_inpaint_image(change_prompt, init_image_b64, mask):
    """
    Purpose:
        Uses Bedrock API to generate an Image using Titan
    Args/Requests:
         text: Prompt
    Return:
        image: base64 string of image
    """
    body = {
        "inPaintingParams": {
            "text": change_prompt,
            "image": init_image_b64,
            "maskImage": image_to_base64(mask),
        },
        "taskType": "INPAINTING",
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

    modelId = "amazon.titan-image-generator-v2:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("images")[0]
    return results


def inpaint_image_pipeline(user_image, change_prompt, mask, model):
    # Turn image to base64 string
    init_image_b64 = image_to_base64(user_image)

    # Invoke Bedrock to inpaint the image
    if model == "Stable Diffusion":
        updated_image = sd_inpaint_image(change_prompt, init_image_b64, mask)
    elif model == "Amazon Titan":
        updated_image = titan_inpaint_image(change_prompt, init_image_b64, mask)

    # convert updated_image to PIL image
    updated_image = base64_to_pil(updated_image)

    # save updated image
    updated_image.save("inpainted_image.png")

    return updated_image


st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Generation Demo - Inpainting")

model = st.selectbox("Select model", ["Stable Diffusion", "Amazon Titan"])

# Streamlit file uploader for only for images
user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Get user prompt to change image
change_prompt = st.text_input("Enter prompt to change image")

# Create three columns, one to show the uploaded image, 2 for the mask, and 3 for the new image
col1, col2, col3 = st.columns(3)

# Show the uploaded image
if user_image is not None:
    user_image = Image.open(user_image)
    col1.image(user_image)

    if model == "Stable Diffusion":
        mask = Image.open("sd_mask.png")
    else:
        mask = gen_mask_from_image(user_image)

    # TODO Finish App with Q
