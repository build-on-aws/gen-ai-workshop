import boto3
import json
import time

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


# Call Mistral model
def call_mistral_8x7b(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mixtral-8x7b-instruct-v0:1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results

# Call Mistral model
def call_mistral_7b(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8
    }

    body = json.dumps(prompt_config)

    modelId = "mistral.mistral-7b-instruct-v0:2"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("outputs")[0].get("text")
    return results

# Call AI21 labs model
def call_ai21(prompt):
    prompt_config = {
        "prompt": prompt,
        "maxTokens": 5147,
        "temperature": 0.7,
        "stopSequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "ai21.j2-ultra-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completions")[0].get("data").get("text")
    return results


def claude_prompt_format(prompt: str) -> str:
    # Add headers to start and end of prompt
    return "\n\nHuman: " + prompt + "\n\nAssistant:"


# Call Claude model
def call_claude_sonet(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results

# Call Claude model
def call_claude(prompt):
    prompt_config = {
        "prompt": claude_prompt_format(prompt),
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-v2:1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("completion")
    return results


# Call Cohere model
def call_cohere(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    body = json.dumps(prompt_config)

    modelId = "cohere.command-text-v14"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("generations")[0].get("text")
    return results


# Call Titan model
def call_titan(prompt):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results


def call_llama2(prompt):
    prompt_config = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "top_p": 0.9,
        "temperature": 0.2,
    }

    body = json.dumps(prompt_config)

    modelId = "meta.llama2-13b-chat-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body["generation"].strip()
    return results


def summarize_text(text):
    """
    Function to summarize text using a generative AI model.
    """
    prompt = f"Summarize the following text in 50 words or less: {text}"
    result = call_titan(prompt)
    return result


def sentiment_analysis(text):
    """
    Function to return a JSON object of sentiment from a given text.
    """
    # TODO Can you write a prompt to help answer these questions?
    result = None
    return result


def perform_qa(question, text):
    """
    Function to perform a Q&A operation based on the provided text.
    """
    # TODO Can you write a prompt to help answer these questions?
    result = None
    return result


if __name__ == "__main__":
    # Sample text for summarization
    text = "Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with"

    print("\n=== Summarization Example ===")
    summary = summarize_text(text)
    print(f"Summary:\n {summary}")
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
