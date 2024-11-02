import logging
import os
import sys
import tempfile
import boto3
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

S3_MODEL_PATH = "s3://cslab-resources-1/mobilenet_v1_1.0_224_quant.tflite"
S3_LABEL_PATH = "s3://cslab-resources-1/labels_mobilenet_quant_v1_224.txt"

logging.basicConfig(level=logging.INFO)

# Create a image classification model
def create_model(model_file_path):
    interpreter = tflite.Interpreter(
        model_path=model_file_path, num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def load_labels(label_file_path):
    with open(label_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def download_s3_file(s3_client, s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    s3_client.download_file(bucket, key, tmp.name)
    return tmp.name

logging.info('Initializing the image classification model')
dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')
model_path = download_s3_file(s3, S3_MODEL_PATH)
label_path = download_s3_file(s3, S3_LABEL_PATH)
model = create_model(model_path)
input_details = model.get_input_details()
output_details = model.get_output_details()
labels = load_labels(label_path)
tablename = os.environ["dynamodb_table_name"]


# logic for inference a image with a local image path,
# return the image class with the probability
def inference(image_path):
    image = Image.open(image_path).resize((224,224))
    image = np.expand_dims(image, axis=0)
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    inference_ret = np.squeeze(model.get_tensor(output_details[0]['index']))
    label_index = inference_ret.argsort()[-1]
    image_class_name = labels[label_index]
    probability = inference_ret[label_index]
    return (image_class_name, probability)

def write_result_to_table(image_path, image_class_name):
    dynamodb.put_item(TableName=tablename,
                      Item={'image_name': {'S': image_path},
                            'image_class_name': {'S': image_class_name}})


# Lambda handler, it download the image from S3 and do the inference
def lambda_handler(event, context):
    logging.info('The image classification lambda model is triggered by S3')
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        logging.info(f'Processing image from S3 path: {bucket}/{key}')

        try:
            # load image
            with tempfile.NamedTemporaryFile() as tmp:
                s3.download_file(bucket, key, tmp.name)

                (prediction_label, prediction_prob) = inference(tmp.name)

                write_result_to_table(f"{bucket}/{key}", prediction_label)
        except Exception as e:
            logging.error(f"Fail to process image with path: {bucket}/{key}",e)


# Main for local testing
if __name__ == "__main__":
    print(download_s3_file(s3, S3_LABEL_PATH))
