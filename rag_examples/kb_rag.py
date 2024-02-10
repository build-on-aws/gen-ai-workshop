import boto3

KB_ID = "TODO"
QUERY = "What can you tell me about Amazon EC2?"
REGION = "us-west-2"
MODEL = "anthropic.claude-v2:1"
NUM_RESULTS = 10

# Setup bedrock
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=REGION,
)


docs_only_response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": QUERY},
    retrievalConfiguration={
        "vectorSearchConfiguration": {"numberOfResults": NUM_RESULTS}
    },
)

for doc in docs_only_response["retrievalResults"]:
    print(f"Citation:\n{doc}\n")

text_response = bedrock_agent_runtime.retrieve_and_generate(
    input={"text": QUERY},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": MODEL,
        },
    },
)

print(f"Output:\n{text_response['output']['text']}\n")
for citation in text_response["citations"]:
    print(f"Citation:\n{citation}\n")
