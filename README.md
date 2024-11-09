# RealityMap
An interactive augmented reality interface that blends digital art with the real world using projection mapping that dynamically maps real-time computer graphics onto physical surfaces.

# Setup
Make a venv `python -m venv .venv`, then activate it `source activate .venv/bin/activate`.
Run `pip install -r requirements.txt` to install required packages.

You must have a webcam attached to your computer. Ideally, using USB3 for the highest resolution.
Run `ffmpeg -f avfoundation -list_devices true -i ""` to list connected webcams.