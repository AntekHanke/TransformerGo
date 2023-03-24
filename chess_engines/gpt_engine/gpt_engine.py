import os

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

start_message = (
    "Let's play a game in explaining what is on a picture.  I have a picture, but i won't share it with you. I want you to ask me questions about it to understand better what is on the picture. "
    'You will start by asking me exactly 5 very specific questions about the picture and I will respond to each of them. Please, make these questions very simple e.g. question "Can you describe the color or colors present in the picture?" is too hard, and it could be simplified to "Is the picture colorful?"'
    "Please do not suggest the answer. Then you will ask me next 5 questions and I will respond to them as well. "
    "After some  rounds I will ask you to sum up what is on the picture. An example start question is: 1. What is in the picture?. Remember to ask me exactly 5 questions at once. "
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that ask questions about the image and summarize its knowledge about it.",
    },
    {"role": "assistant", "content": start_message},
]

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

openai.