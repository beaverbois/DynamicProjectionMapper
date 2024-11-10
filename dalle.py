import requests
import json
import os

class DallE:
    def __init__(self):
        self.api_key = open("API_key.txt", "r").read()  # Get the API key from environment variable

        # Check if the API key is available
        if not self.api_key:
            print("API key is missing!")
            exit()

        # The endpoint URL for image generation
        self.url = "https://api.openai.com/v1/images/generations"

        # The headers for the request
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generateImage(self, prompt):
        # The payload (data) to send with the POST request
        data = {
            "prompt": prompt,
            "n": 1,  # Number of images to generate
            "size": "1024x1024"  # Image size (you can also use "256x256", "512x512", etc.)
        }

        # Send the POST request to the OpenAI API
        response = requests.post(self.url, headers=self.headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Print the response JSON (which contains the image URL)
            response_data = response.json()
            print("Image URL:", response_data['data'][0]['url'])

            # Optionally, download and save the image
            image_url = response_data['data'][0]['url']
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                # Save, return the image
                with open("images/textures/"+prompt+".jpg", "wb") as file:
                    file.write(image_response.content)
                print("Image saved")
                return "images/textures/"+prompt+".jpg"
            else:
                print("Failed to download image.")
        else:
            # Print error details if the request fails
            print(f"Request failed with status code {response.status_code}: {response.text}")
