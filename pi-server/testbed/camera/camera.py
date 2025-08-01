from flask import Flask, render_template, Response
import v4l2py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    with v4l2py.Device.from_id(0) as camera:
        capture = v4l2py.VideoCapture(camera)
        capture.set_format(640, 480, "MJPG")
        for frame in camera:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.data + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
