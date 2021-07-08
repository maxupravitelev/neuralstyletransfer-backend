#########################################################################################
### Built upon: https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization
#########################################################################################


#########################################################################################
### handle imports
import time
#import json
import sys

# flask imports
from flask import Response
from flask import Flask, request
#from flask_cors import CORS
from flask import jsonify


# tensorflow imports
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#########################################################################################


#########################################################################################
### init flask
app = Flask(__name__)
#CORS(app)

#########################################################################################

#########################################################################################
### init tensorflow & helper functions
print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

#########################################################################################


#########################################################################################
### API routes

# route for steering the vehicle
@app.route('/steer', methods=['POST'])
def post_image():


    message = "ping"


    return jsonify(message)

# route for generated video ouput
@app.route("/")
def get_image():


    message = "ping"
    print("ping")


    return jsonify(message)


#########################################################################################


#########################################################################################
### start the flask app
if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port="6475", ssl_context='adhoc', debug=True,
            threaded=True, use_reloader=False)
        #socketio.run(app, host='0.0.0.0', port=6475, debug=True)
    except KeyboardInterrupt:
        #check for exit
        time.sleep(1)
        print("exit program")
        sys.exit(0)
