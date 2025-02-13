import random

import boto3

AGENT_ID = "INSERT_AGENT_ID"
QUERY = "What are some features of Amazon S3?"
REGION = "us-east-1"

# Setup bedrock
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=REGION,
)


def generate_random_15digit():
    number = ""

    for _ in range(15):
        number += str(random.randint(0, 9))

    return number


def format_retrieved_references(references):
    # Extracting the text and link from the references
    for reference in references:
        content_text = reference.get("content", {}).get("text", "")
        s3_uri = reference.get("location", {}).get("s3Location", {}).get("uri", "")

        # Formatting the output
        formatted_output = "Reference Information:\n"
        formatted_output += f"Content: {content_text}\n"
        formatted_output += f"S3 URI: {s3_uri}\n"

        print(formatted_output)


def process_stream(stream):
    try:
        # print("Processing stream...")
        trace = stream.get("trace", {}).get("trace", {}).get("orchestrationTrace", {})

        if trace:
            # print("This is a trace")
            knowledgeBaseInput = trace.get("invocationInput", {}).get(
                "knowledgeBaseLookupInput", {}
            )
            if knowledgeBaseInput:
                print(
                    f'Looking up in knowledgebase: {knowledgeBaseInput.get("text", "")}'
                )
            knowledgeBaseOutput = trace.get("observation", {}).get(
                "knowledgeBaseLookupOutput", {}
            )
            if knowledgeBaseOutput:
                retrieved_references = knowledgeBaseOutput.get(
                    "retrievedReferences", {}
                )
                if retrieved_references:
                    print("Formatted References:")
                    format_retrieved_references(retrieved_references)

        # Handle 'chunk' data
        if "chunk" in stream:
            print("This is the final answer:")
            text = stream["chunk"]["bytes"].decode("utf-8")
            print(text)

    except Exception as e:
        print(f"Error processing stream: {e}")
        print(stream)


def run_agent():

    response = bedrock_agent_runtime.invoke_agent(
        sessionState={
            "sessionAttributes": {},
            "promptSessionAttributes": {},
        },
        agentId=AGENT_ID,
        agentAliasId="TSTALIASID",
        sessionId=str(generate_random_15digit()),
        endSession=False,
        enableTrace=True,
        inputText=QUERY,
    )
    print(response)

    results = response.get("completion")

    for stream in results:
        process_stream(stream)


if __name__ == "__main__":
    run_agent()
