from http import HTTPStatus
from flask import Flask, Response
from flask_cors import CORS

from src.control_center.routes import llm, wifi, bluetooth, peripheral

app = Flask(__name__)
CORS(app)

@app.post("/api/healthcheck")
def healthcheck():
    return Response(status=HTTPStatus.OK)

app.register_blueprint(bluetooth.bp, url_prefix="/api/bluetooth")
app.register_blueprint(wifi.bp, url_prefix="/api/wifi")
app.register_blueprint(llm.bp, url_prefix="/api/llm")
app.register_blueprint(peripheral.bp, url_prefix="/api/peripheral")

if __name__ == "__main__":
    app.run(port=42068) # disappointing
