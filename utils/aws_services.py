import os
import io
import boto3
import random
import time
import json
import logging
from urllib.request import urlopen


from config.environments import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION


# AWS Texttract
class AWSTexttract:

    def __init__(self):
        self.client = boto3.client(
            "textract",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

    def get_text(self, file_path):

        if type(file_path) == str:
            # cioè se passo il path del file
            with open(file_path, "rb") as file:
                img_test = file.read()
                bytes_test = bytearray(img_test)
                print("Image loaded", file_path)
            response = self.client.detect_document_text(Document={"Bytes": bytes_test})
        else:
            # se passo il formato PIL
            buf = io.BytesIO()
            file_path.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            response = self.client.detect_document_text(Document={"Bytes": byte_im})

        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text += item["Text"] + "\n"

        return text


# AWS Transcribe
class AWSTranscribe:

    def __init__(self, job_uri, region):
        self.transcribe = boto3.client(
            "transcribe",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=region,
        )
        self.job_verification = False
        self.job_uri = job_uri

    def generate_job_name(self):
        return "stt_" + str(time.time_ns()) + "_" + str(random.randint(0, 500))

    def check_job_name(self, job_name):
        """
        Check if the transcribe job name is existed or not
        """
        self.job_verification = True
        # all the transcriptions
        existed_jobs = self.transcribe.list_transcription_jobs()
        for job in existed_jobs["TranscriptionJobSummaries"]:
            if job_name == job["TranscriptionJobName"]:
                self.job_verification = False
            break
        # if job_verification == False:
        #     command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
        #     if command.lower() == "y" or command.lower() == "yes":
        #         self.transcribe.delete_transcription_job(TranscriptionJobName=job_name)
        # elif command.lower() == "n" or command.lower() == "no":
        #     job_name = input("Insert new job name? ")
        #     self.check_job_name(job_name)
        # else:
        #     print("Input can only be (Y/N)")
        #     command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
        return job_name

    def amazon_transcribe(self, job_uri, job_name, audio_file_name, language):
        """
        For single speaker
        """
        # Usually, I put like this to automate the process with the file name
        # "s3://bucket_name" + audio_file_name
        # Usually, file names have spaces and have the file extension like .mp3
        # we take only a file name and delete all the space to name the job
        job_uri = os.path.join("s3://" + job_uri, audio_file_name)
        # file format
        file_format = audio_file_name.split(".")[-1]

        # check if name is taken or not
        job_name = self.check_job_name(job_name)
        print("Transctiption started from:")
        print(job_uri)
        self.transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": job_uri},
            MediaFormat=file_format,
            LanguageCode=language,
        )

        while True:
            result = self.transcribe.get_transcription_job(
                TranscriptionJobName=job_name
            )
            if result["TranscriptionJob"]["TranscriptionJobStatus"] in [
                "COMPLETED",
                "FAILED",
            ]:
                break
            time.sleep(15)
        if result["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
            response = urlopen(
                result["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            )
            data = json.loads(response.read())
        return data["results"]["transcripts"][0]["transcript"]


# AWS S3
class AWSS3:

    def __init__(self, bucket=None):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        self.bucket = bucket

    def read_metadata(self, key, id):
        response = self.s3_client.head_object(Bucket=self.bucket, Key=key)
        return response["Metadata"][id]

    def list_items(self, key):
        list = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=key)
        return list.get("Contents", [])

    def upload_file(self, fileobj, key):
        """Upload a file to an S3 bucket"""
        try:
            # the first argument is the file path

            self.s3_client.upload_fileobj(
                fileobj, self.bucket, key, ExtraArgs={"Metadata": {"Name": key}}
            )
            logging.info("File Successfully Uploaded on S3")
            return True
        except FileNotFoundError:
            time.sleep(9)
            logging.error("File not found.")
            return False

    def delete_file(self, object_name):
        """Delete a file from an S3 bucket
        :param object_name: S3 object name
        :return: True if file was deleted, else False
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            return True
        except Exception as e:
            logging.error(e)
            return False

    def download_file(self, object_name, file_name):
        """
        Download a file from an S3 bucket
        :param bucket: Bucket to download from
        :param object_name: S3 object name
        :param file_name: File to download, path
        :return: True if file was downloaded, else False
        """
        # Download the file
        try:
            # IMPORTANT: It's necessary to check if the directory exists for the upload to work properly

            # Extract the directory path from the file_name
            directory = os.path.dirname(file_name)
            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            response = self.s3_client.download_file(self.bucket, object_name, file_name)
            if response is None:
                # Download successful
                return True
        except Exception as e:
            logging.error(f"Error downloading file from S3: {e}")
            return False

    def copy_file(self, target_bucket, target_key, dest_bucket):
        """Copy a file from an S3 bucket
        :param dest: S3 destination
        :param target: S3 target
        :return: True if file was copied, else False
        """
        try:
            copy_source = {"Bucket": target_bucket, "Key": target_key}
            bucket = boto3.resource("s3").Bucket(
                dest_bucket
            )  # resource method https://stackoverflow.com/questions/70293628/attributeerror-s3-object-has-no-attribute-bucket
            bucket.copy(copy_source, target_key)
            return True
        except Exception as e:
            logging.error(e)
            return False


# Lambda
class AWSLambda:
    def __init__(self):
        self.lambda_client = boto3.client(
            "lambda",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

    def invoke_lambda(self, function_name, payload):
        """Invoke a lambda function"""
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=payload,
            )
            logging.info("Lambda invoked")
            body = response["Payload"].read()
            json_object = json.loads(body)
            return json_object["body"]
        except Exception as e:
            logging.error(e)
            return False
