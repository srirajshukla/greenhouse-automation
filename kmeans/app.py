import base64
import io

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2

from segment import segmentation

app = Flask(__name__)

@app.route("/api/hello")
def hello():
    return "Hello World!"

@app.route("/api/kmeans", methods=['POST'])
def handle_kmeans():
    # print(request.headers)
    # print(request.data)
    # print(request.args)
    # print(request.form)
    # print(request.endpoint)
    # print(request.method)
    # print(request.remote_addr)
    img_b64 = request.form.get('image')
    file = base64.decodebytes(img_b64.encode())
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # calls the acutal segmentation algorithm
    segmented = segmentation(img)

    # now convert the nparray to 
    img = Image.fromarray(segmented.astype('uint8'))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64)[2:-1]})

# @app.after_request
# def add_header(r):
#     r.headers.add('Access-Control-Allow-Origin', '*')
#     r.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     r.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')


if __name__ == "__main__":
    app.run(debug=True)