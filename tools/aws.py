from langchain.agents import Tool
import boto3
import json
import os

lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION'))
function_name_1 = 'riassume-turnon-ec2-lambda'
function_name_2 = 'riassume-turnoff-ec2-lambda'
payload = {}

def turnon_ec2(input : str) -> str:
    # Convert the payload to JSON
    payload_json = json.dumps(payload)

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=function_name_1,
        InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
        Payload=payload_json
    )

    # Read the response
    response_payload = response['Payload'].read()
    response_dict = json.loads(response_payload)
    print(response_dict)
    return "Started!"

def turnoff_ec2(input : str) -> str:
    # Convert the payload to JSON
    payload_json = json.dumps(payload)

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=function_name_2,
        InvocationType='RequestResponse',  # Use 'Event' for asynchronous invocation
        Payload=payload_json
    )

    # Read the response
    response_payload = response['Payload'].read()
    response_dict = json.loads(response_payload)
    print(response_dict)
    return "Stopped!"

ec2_shutdown_tools = Tool(
    name="Accendi Ec2",
    func=turnon_ec2,
    description="Utile per accendere un istanza EC2 (VM) sul cloud AWS"
)

ec2_turnon_tools = Tool(
    name="Spegni Ec2",
    func=turnoff_ec2,
    description="Utile per spegnere un istanza EC2 (VM) sul cloud AWS"
)
