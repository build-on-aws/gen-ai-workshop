import os

import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_unstructured import UnstructuredLoader

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)


def chunk_doc_to_text(doc_loc: str):
    loader = UnstructuredLoader(doc_loc)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(docs)

    return texts


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


def rag_with_bedrock(query):

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1",
    )
    pdf_loc = "well_arch.pdf"

    if os.path.exists("local_index"):
        local_vector_store = FAISS.load_local(
            "local_index", embeddings, allow_dangerous_deserialization=True
        )
    else:
        texts = chunk_doc_to_text(pdf_loc)
        local_vector_store = FAISS.from_documents(
            texts, embeddings, allow_dangerous_deserialization=True
        )
        local_vector_store.save_local("local_index")

    docs = local_vector_store.similarity_search(query)
    context = ""

    for doc in docs:
        context += doc.page_content

    prompt = f"""Use the following pieces of context to answer the question at the end.

    {context}

    Question: {query}
    Answer:"""

    model_id = "us.amazon.nova-pro-v1:0"
    # Setup the system prompts and messages to send to the model.
    system_prompts = [
        {
            "text": "You are an expert AWS Solutions Architect that helps customers solve their problems."
        }
    ]
    message_1 = {
        "role": "user",
        "content": [{"text": f"{prompt}"}],
    }

    messages = [message_1]

    result = generate_conversation(model_id, system_prompts, messages)

    return result


query = "What can you tell me about Amazon RDS?"
print(query)
print(rag_with_bedrock(query))
