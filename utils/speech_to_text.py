import re
import os
import logging
import moviepy.editor as mp
import speech_recognition as sr
from langchain_community.llms import openai

from utils.aws_services import AWSTranscribe
from config.environments import OPENAI_API_KEY
from config.constants import FILE_EXTENSIONS


class SpeechToText:

    def __init__(self, model: str, bucket: str = "newsp4-transcribe-docs-bucket"):
        # Setup the logger
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s :%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.model = model
        self.bucket = bucket
        self.SUPPORTED_MODELS = {
            "whisper-base": "Whisper Base Model (commented out)",
            "transcribe": "AWS Transcribe Model",
            "gpt-3.5-turbo": "GPT 3.5 Turbo Model",
        }
        self.VIDEO_EXTENSIONS = set(ext for ext in FILE_EXTENSIONS["video"])
        self.AUDIO_EXTENSIONS = set(ext for ext in FILE_EXTENSIONS["audio"])
        self.TEXT_EXTENSIONS = set(ext for ext in FILE_EXTENSIONS["text"])

    # Function to extract audio from video file
    def extract_audio(self, video_file: str) -> str:
        try:
            video = mp.VideoFileClip(video_file)
            audio = video.audio
            audio_file = os.path.splitext(video_file)[0] + ".wav"
            audio.write_audiofile(audio_file)

            self.logger.info(
                f"Video file '{video_file}' converted to audio file '{audio_file}'"
            )
            return audio_file

        except FileNotFoundError as e:
            self.logger.error(f"File '{video_file}' not found")
            raise e

        except Exception as e:
            self.logger.error(
                f"Error in extracting audio from video file '{video_file}'"
            )
            raise e

    # Function to perform speech to text conversion
    def speech_to_text(self, audio_file) -> str:
        """Records audio from an audio file then pass it to google cloud speech to text API to extract text"""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                # Was recognizer.recognize_google(audio) but it is not defined in the module
                # IMPORTANT: This will rquire google credentials in json file to be added to environment values
                text = recognizer.recognize_google_cloud(audio)
                self.logger.info(
                    f"Text has been extracted using google cloud from audio file '{audio_file}'"
                )

                # Remove the temporary WAV file if it was created
                if audio_file.endswith(".wav") and audio_file != os.path.basename(
                    audio_file
                ):
                    os.remove(audio_file)

                return text

        except FileNotFoundError as e:
            self.logger.error(f"File '{audio_file}' not found")
            raise e

        except Exception as e:
            self.logger.error(
                f"Error in speech to text conversion of audio file '{audio_file}'"
            )
            raise e

    # Function to clean text
    def clean_text(self, text: str) -> str:
        try:
            # Remove stammering words (words repeated more than twice)
            cleaned_text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)
            # Remove punctuation and special characters
            cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned_text)
            self.logger.info(f"Text has been cleaned.")
            return cleaned_text

        except Exception as e:
            self.logger.error(f"Error in cleaning text '{text}'")
            raise e

    def openai_api(self, text: str):
        """Using OpenAI API to improvise text"""
        try:
            prompt = "Be precise and rewrite the context and topic in thie video in simple words"
            content = prompt + " " + text

            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            # Note: This has been commented out temporarily to test the module

            # response = client.chat.completions.create(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": content,
            #         }
            #     ],
            #     model=self.model,
            #     )
            improvised_text = client(content)
            print(improvised_text)
            self.logger.info(f"Text has been improvised using OpenAI API.")

            return improvised_text

        except Exception as e:
            self.logger.error(f"Error in improvising text '{text}'")
            raise e

    def transcribe(self, file_path: str) -> str:
        """
        Takes a file path in ingress and returns a text in output
        """
        try:
            cleaned_text = ""
            file_extension = os.path.splitext(file_path)[-1].lower()

            if self.model not in self.SUPPORTED_MODELS:
                self.logger.error(
                    f"Model '{self.model}' is not supported. Supported models: {', '.join(self.SUPPORTED_MODELS.keys())}"
                )
                raise ValueError("Unsupported model")

            # if self.model == 'whisper-base':
            #     model = whisper.load_model("base")
            #     text = model.transcribe(file_path)
            #     return text['text']

            elif self.model == "transcribe":
                transcribe = AWSTranscribe(self.bucket, "us-east-1")
                job_name = transcribe.generate_job_name()
                text = transcribe.amazon_transcribe(
                    self.bucket, job_name, file_path, "it-IT"
                )
                return text

            elif self.model.startswith("gpt"):
                # if file_path is:
                # an audio file skip extract_audio
                # a text skip it to speech_to_text
                if file_extension in self.VIDEO_EXTENSIONS:
                    # Step 1: Extract audio from video
                    audio_file = self.extract_audio(file_path)
                    # Step 2: Convert speech to text
                    text = self.speech_to_text(audio_file)
                    # Step 3: Clean the text
                    cleaned_text = self.clean_text(text)

                elif file_extension in self.AUDIO_EXTENSIONS:
                    # Convert audio file to WAV if it's not in WAV format
                    if file_extension != "wav":
                        wav_file = os.path.splitext(file_path)[0] + ".wav"
                        audio = mp.AudioFileClip(file_path)
                        audio.write_audiofile(wav_file)
                        file_path = wav_file
                    text = self.speech_to_text(file_path)
                    cleaned_text = self.clean_text(text)

                elif file_extension in self.TEXT_EXTENSIONS:
                    with open(file_path, "r") as f:
                        text = f.read()
                        cleaned_text = self.clean_text(text)

                # Step 4: Input text to OpenAI API
                improvised_text = self.openai_api(cleaned_text)
                return improvised_text

            else:
                pass
                # TODO implement new models!!

        except Exception as e:
            self.logger.error(f"Error in transcribing file '{file_path}'")
            raise e
