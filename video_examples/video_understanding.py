# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import json

import boto3


def create_bedrock_client(region="us-west-2"):
    """Create a Bedrock Runtime client in the specified AWS Region."""
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
    )


def encode_video_to_base64(video_path):
    """Open video file and encode it as a Base64 string."""
    with open(video_path, "rb") as video_file:
        binary_data = video_file.read()
        # TODO return base64 encoded data
        return


def create_request_payload(base64_string, system_prompt, user_prompt, temperature=0.7):
    """Create the request payload for the model."""
    system_list = [{"text": system_prompt}]
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {"bytes": base64_string},
                    }
                },
                {"text": user_prompt},
            ],
        }
    ]
    inf_params = {"temperature": temperature}

    return {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }


def invoke_model_and_get_response(client, model_id, request_payload):
    """Invoke the model and get response."""
    response = client.invoke_model(modelId=model_id, body=json.dumps(request_payload))
    return json.loads(response["body"].read())


def print_response(model_response):
    """Print the full response and content text."""
    print("[Full Response]")
    print(json.dumps(model_response, indent=2))
    content_text = model_response["output"]["message"]["content"][0]["text"]
    print("\n[Response Content Text]")
    print(content_text)


def main():
    MODEL_ID = "us.amazon.nova-pro-v1:0"
    VIDEO_PATH = "the-sea.mp4"
    SYSTEM_PROMPT = (
        "You are great at coming up with catchy titles to inspire the audience."
    )
    USER_PROMPT = "Provide a 3 titles"

    client = create_bedrock_client()
    base64_string = encode_video_to_base64(VIDEO_PATH)
    request_payload = create_request_payload(base64_string, SYSTEM_PROMPT, USER_PROMPT)
    model_response = invoke_model_and_get_response(client, MODEL_ID, request_payload)
    print_response(model_response)


if __name__ == "__main__":
    main()
