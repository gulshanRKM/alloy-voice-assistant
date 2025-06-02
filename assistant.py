import base64
from threading import Lock, Thread
import time
import numpy
import cv2
import openai
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        from mss import mss
        with mss() as sct:
            while self.running:
                monitor = sct.monitors[1]  # Primary monitor
                sct_img = sct.grab(monitor)
                img = numpy.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                with self.lock:
                    self.screenshot = img

                time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None

        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)

        return screenshot

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="shimmer",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

        player.close()

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_base64}"},
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

# Initialize screenshot capturing
desktop_screenshot = DesktopScreenshot().start()

# Initialize model
model = ChatOpenAI(model="gpt-4o")
assistant = Assistant(model)

# Initialize speech recognizer
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

# Background callback

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, desktop_screenshot.read(encode=True))
    except UnknownValueError:
        print("There was an error processing the audio.")

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# Main loop
try:
    while True:
        screenshot = desktop_screenshot.read()
        if screenshot is not None:
            cv2.imshow("Desktop", screenshot)
        if cv2.waitKey(1) in [27, ord("q")]:
            break
except KeyboardInterrupt:
    pass
finally:
    desktop_screenshot.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
    