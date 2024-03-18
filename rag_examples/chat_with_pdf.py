import json
import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
)

REGION = "us-west-2"

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


def chunk_doc_to_text(doc_loc: str):
    loader = UnstructuredFileLoader(doc_loc)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(docs)

    return texts


def call_claude_sonnet(prompt):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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

    return call_claude_sonnet(prompt)


query = "What can you tell me about Amazon RDS?"
print(query)
print(rag_with_bedrock(query))
