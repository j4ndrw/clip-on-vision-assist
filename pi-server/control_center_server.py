from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.post("/api/control-center/under-construction")
def under_construction():
    return "Under construction"
