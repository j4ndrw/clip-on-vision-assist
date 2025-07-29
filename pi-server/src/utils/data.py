import base64


def as_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")
