import io

import requests
from PIL import Image

url = "http://localhost:8000/api/v1/screen"

# Sample dummy passport TD3 MRZ lines
data = {
    "mrz_line1": "P<INDKUMAR<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "mrz_line2": "M1234567<8IND9501015M3001018<<<<<<<<<<<<<<0"
}

files = {
    "document_image": ("passport_sample.jpg", io.BytesIO(), "image/jpeg")
}

image = Image.new("RGB", (800, 500), "white")
image_buffer = files["document_image"][1]
image.save(image_buffer, format="JPEG")
image_buffer.seek(0)

print("Testing API locally...")
response = requests.post(url, data=data, files=files, timeout=30)
print(response.status_code)
print(response.json())
