from pydantic import BaseModel
from typing import Literal
import requests
import json

url = 'http://[::1]:5000/predictions'
headers = { "accept": "application/json" }

tasks = []

task = { "input": {
         "task": "image_captioning",
         "image": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
         }
       }
tasks.append(task)

task = { "input": {
        "task": "visual_question_answering",
        "image": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        "question": "where is the woman?"
         }
       }
tasks.append(task)

task = { "input": {
        "task": "image_text_matching",
        "image": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        "caption": "a dog and a women are sitting at the beach"
         }
       }
tasks.append(task)


for task in tasks:
    try:
        response = requests.post(url, headers=headers, json=task)
        print("------------------------------------------> " + task['input']['task'] )
        print(json.dumps(response.json()))
    except Exception as e:
        print(e)

