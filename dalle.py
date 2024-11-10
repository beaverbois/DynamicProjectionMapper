import requests
import json
import os

class DallE:
    def __init__(self):
        self.api_key = open("API_key.txt", "r").read()  # Get the API key from env variable

        if not self.api_key:
            print("API key is missing!")
            exit()

        self.url = "https://api.openai.com/v1/images/generations"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generateImage(self, prompt):
        # The payload (data) to send with the POST request
        data = {
            "prompt": prompt,
            "n": 1,  # Number of images to generate
            "size": "1024x1024"  # Image size
        }

        # Send POST request to OpenAI API
        response = requests.post(self.url, headers=self.headers, json=data)

        if response.status_code == 200:
            # Print response JSON 
            response_data = response.json()
            print("Image URL:", response_data['data'][0]['url'])

            image_url = response_data['data'][0]['url']
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                # Save, return image
                with open("images/textures/"+prompt+".jpg", "wb") as file:
                    file.write(image_response.content)
                print("Image saved")
                return "images/textures/"+prompt+".jpg"
            else:
                print("Failed to download image.")
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
