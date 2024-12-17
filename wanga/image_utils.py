import base64
import urllib.parse
from io import BytesIO

import requests
from PIL import Image

__all__ = ["url_to_base64"]


def is_url(url: str) -> bool:
    return urllib.parse.urlparse(url).scheme in ("http", "https")


def url_to_base64(url: str) -> str:
    if not is_url(url):
        return url
    image = BytesIO(requests.get(url).content)
    image = Image.open(image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
