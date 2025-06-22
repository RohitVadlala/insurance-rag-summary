import boto3
import json

def query_bedrock_claude(prompt: str) -> str:
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    # Claude expects this format with "\n\nHuman:" and "\n\nAssistant:"
    claude_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"

    body = json.dumps({
        "prompt": claude_prompt,
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 1.0,
        "stop_sequences": ["\n\nHuman:"]
    })

    response = client.invoke_model(
        modelId="anthropic.claude-instant-v1",
        contentType="application/json",
        accept="application/json",
        body=body
    )

    result = json.loads(response['body'].read())
    return result['completion']

