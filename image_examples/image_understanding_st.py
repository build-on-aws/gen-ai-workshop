import streamlit as st
import boto3
import json
import base64
import io
from PIL import Image


st.title("Building with Bedrock")  # Title of the application
st.subheader("Image Understanding Demo")

REGION = "us-west-2"

# Define bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)


def call_claude_sonnet(base64_string):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": "Provide a caption for this image"},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


# function to convert PIL image to base64 string
def pil_to_base64(image, format="png"):
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        return base64.b64encode(buffer.getvalue()).decode()


# Streamlit file uploader for only for images
user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Creat two columns, one to show the uploaded image, another for the new image
col1, col2 = st.columns(2)

# Show the uploaded image
if user_image is not None:
    user_image = Image.open(user_image)
    # TODO Finish App with Q
    col1.image(user_image, use_column_width=True)

    if col1.button("Get caption"):
        base64_image = pil_to_base64(user_image)
        caption = call_claude_sonnet(base64_image)
        col2.write(caption)

else:
    col2.write("No image uploaded")
