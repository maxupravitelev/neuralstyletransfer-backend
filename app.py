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
from flask import Flask, request, send_file
from flask_cors import CORS
from flask import jsonify

import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # disable gpu if necessary

#########################################################################################


#########################################################################################
### init flask
app = Flask(__name__)
CORS(app)

#########################################################################################

#########################################################################################
### init tensorflow & helper functions
print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))


def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""

  image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...] # add 0 channel 

  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

#########################################################################################


#########################################################################################
### API routes

@app.route('/api/images', methods=['POST'])
def post_images():
    
    global load_image

    # create new folder for handling multiple concurrent request
    # TODO

    # save received images for processing
    request.files["contentImage"].save("contentImage.jpg")    
    request.files["styleImage"].save("styleImage.jpg")

    # load images 
    content_image_url = 'contentImage.jpg' # @param {type:"string"}
    style_image_url = 'styleImage.jpg'  # @param {type:"string"}
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the 
    # recommended image size for the style image (though, other sizes work as 
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    squeezed_image = tf.squeeze(stylized_image)
    tf.keras.preprocessing.image.save_img("1.jpg", squeezed_image)

    return jsonify("message")
    #return send_file("1.jpg", mimetype='image/jpg')

# return generated output on request
@app.route('/api/images/generated_output', methods=['GET'])
def get_generated_output():
    return send_file("1.jpg", mimetype='image/jpg')


# route for req testing
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
        app.run(host="0.0.0.0", port="6475", 
                #ssl_context='adhoc', 
                debug=True,
                threaded=True, use_reloader=False)
        #socketio.run(app, host='0.0.0.0', port=6475, debug=True)
    except KeyboardInterrupt:
        #check for exit
        time.sleep(1)
        print("exit program")
        sys.exit(0)
