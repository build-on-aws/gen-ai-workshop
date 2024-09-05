import time

import boto3

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


def generate_conversation(model_id, system_prompts, messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    print(f"Generating message with model {model_id}")

    # Inference parameters to use.
    temperature = 0.5

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    # top_k = 200
    # additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        # additionalModelRequestFields=additional_model_fields,
    )

    # Log token usage.
    token_usage = response["usage"]
    print(f"Input tokens: {token_usage['inputTokens']}")
    print(f"Output tokens: {token_usage['outputTokens']}")
    print(f"Total tokens: {token_usage['totalTokens']}")
    print(f"Stop reason: {response['stopReason']}")

    text_response = response["output"]["message"]["content"][0]["text"]

    return text_response


model_ids = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0"
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0",
]


def summarize_text(text):
    """
    Function to summarize text using a generative AI model.
    """

    model_id = "meta.llama3-1-70b-instruct-v1:0"
    # Setup the system prompts and messages to send to the model.
    system_prompts = [
        {"text": "You are an app that creates summaries of text in 50 words or less."}
    ]
    message_1 = {
        "role": "user",
        "content": [{"text": f"Summarize the following text: {text}."}],
    }

    messages = [message_1]

    result = generate_conversation(model_id, system_prompts, messages)

    return result


def sentiment_analysis(text):
    """
    Function to return a JSON object of sentiment from a given text.
    """
    # TODO Can you fill in the function?
    result = None
    return result


def perform_qa(question, text):
    """
    Function to perform a Q&A operation based on the provided text.
    """
    # TODO Can you fill in the function?
    result = None
    return result


if __name__ == "__main__":
    # Sample text for summarization
    text = "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with"

    print("\n=== Summarization Example ===")
    summary = summarize_text(text)
    print(f"Summary:\n{summary}")
    time.sleep(2)

    print("\n=== Sentiment Analysis Example ===")
    sentiment_analysis_json = sentiment_analysis(text)
    print(f"Sentiment_Analysis JSON:\n{sentiment_analysis_json}")
    time.sleep(2)

    print("\n=== Q&A Example ===")

    q1 = "How many companies have models in Amazon Bedrock?"
    print(q1)
    answer = perform_qa(q1, text)
    print(f"Answer: {answer}")
    time.sleep(2)

    q2 = "Can Amazon Bedrock support RAG?"
    print(q2)
    answer = perform_qa(q2, text)
    print(f"Answer: {answer}")
    time.sleep(2)

    q3 = "When was Amazon Bedrock announced?"
    print(q3)
    answer = perform_qa(q3, text)
    print(f"Answer: {answer}")
