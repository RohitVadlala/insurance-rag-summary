import boto3

def extract_text_from_s3(bucket, document):
    # textract = boto3.client('textract')
    textract = boto3.client('textract', region_name='us-east-1')


    response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': document}}
    )

    lines = [block["Text"] for block in response["Blocks"] if block["BlockType"] == "LINE"]
    return "\n".join(lines)

