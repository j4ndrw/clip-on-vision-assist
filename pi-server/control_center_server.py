from http import HTTPStatus

from flask import Flask, Response, json
from flask_cors import CORS

from src.control_center.routes import bluetooth, hotspot, llm, peripheral, testbed, wifi

app = Flask(__name__)
CORS(app)


@app.get("/api/healthcheck")
def healthcheck():
    return Response(response=json.dumps({}), status=HTTPStatus.OK)


app.register_blueprint(bluetooth.bp, url_prefix="/api/bluetooth")
app.register_blueprint(wifi.bp, url_prefix="/api/wifi")
app.register_blueprint(llm.bp, url_prefix="/api/llm")
app.register_blueprint(peripheral.bp, url_prefix="/api/peripheral")
app.register_blueprint(hotspot.bp, url_prefix="/api/hotspot")
app.register_blueprint(testbed.bp, url_prefix="/api/testbed")

if __name__ == "__main__":
    app.run(port=42068)  # disappointing
