import io

import requests
from PIL import Image

def main() -> None:
    url = "http://localhost:8000/api/v1/screen"
    data = {
        "mrz_line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "mrz_line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    }
    image_buffer = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    files = {"document_image": ("passport_sample.jpg", image_buffer, "image/jpeg")}

    print("Testing API locally...")
    response = requests.post(url, data=data, files=files, timeout=30)
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
