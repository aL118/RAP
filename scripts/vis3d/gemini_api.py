import base64
from google import genai

client = genai.Client()

# Load a local image
image_path = "/fs/nexus-projects/sim2real/aliu/RAP/data/test/ambulance/frames/000073.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {"type": "text", "text": "Explain the content of this image."},
        {
            "type": "image",
            "data": image_b64,
            "mime_type": "image/jpeg"
        }
    ]
)
print(interaction.output_text)