from flask import Flask, render_template, Response
import cv2
import time

class VideoCamera(object):
    def __init__(self):
        time.sleep(2.0)
        self.camera = cv2.VideoCapture(0)

    def get_frame(self):
        _, frame = self.camera.read()
        _, png = cv2.imencode(".png", frame)
        return png.tobytes()

pi_camera = VideoCamera()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera: VideoCamera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(pi_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
