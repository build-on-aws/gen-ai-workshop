import os
import pandas
import boto3

S3_BUCKET = os.environ["S3_BUCKET"]
S3_OBJECT = os.environ["S3_OBJECT"]

def lambda_handler(event, context):
    # Print the received event to the logs
    print("Received event: ")
    print(event)

    # Initialize response code to None
    response_code = None

    # Extract the action group, api path, and parameters from the prediction
    action = event["actionGroup"]
    api_path = event["apiPath"]
    inputText = event["inputText"]
    httpMethod = event["httpMethod"]

    print(f"inputText: {inputText}")

    # Check the api path to determine which tool function to call
    if api_path == "/get_num_records":
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_OBJECT, "/tmp/data.csv")
        df = pandas.read_csv("/tmp/data.csv")

        # Get count of dataframe
        count = len(df)

        response_body = {"application/json": {"body": str(count)}}
        response_code = 200
    else:
        # If the api path is not recognized, return an error message
        body = {"{}::{} is not a valid api, try another one.".format(action, api_path)}
        response_code = 400
        response_body = {"application/json": {"body": str(body)}}

    # Print the response body to the logs
    print(f"Response body: {response_body}")

    # Create a dictionary containing the response details
    action_response = {
        "actionGroup": action,
        "apiPath": api_path,
        "httpMethod": httpMethod,
        "httpStatusCode": response_code,
        "responseBody": response_body,
    }

    # Return the list of responses as a dictionary
    api_response = {"messageVersion": "1.0", "response": action_response}

    return api_response

