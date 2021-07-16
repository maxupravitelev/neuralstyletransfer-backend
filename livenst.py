import cv2
import time
import functools
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # disable gpu

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Load TF-Hub module.
#hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_handle = 'models/arbv1'
hub_module = hub.load(hub_handle)

## init capture
frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)

cap.set(3, frame_width)
cap.set(4, frame_height)

# warmup cam
time.sleep(1)

# set fps in loop
fps = 1 / 1

# Built upon: https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
def scale_image(image, max_dim = 512):

  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio * 1, tf.int32)

  # Resizes the image based on the scaling constant generated above
  img = tf.image.resize(image, new_shape)
  return img[tf.newaxis, ...]

# Built upon: https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization
@functools.lru_cache(maxsize=None)
def load_image(image_path):

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)

  img = scale_image(img)

  return img

output_image_size = 384  # @param {type:"integer"}
i = 15
style_image_url = f"images/f/{i}.jpg"  # @param {type:"string"}
style_image = load_image(style_image_url)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

frame_counter = 0

while (cap.isOpened()):
    #time.sleep(fps)

    _, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)

    content_image = scale_image(frame_tensor)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

    stylized_image = outputs[0]
    squeezed_image = tf.squeeze(stylized_image)

    output_image = cv2.cvtColor(squeezed_image.numpy() * 255, cv2.COLOR_RGB2BGR)
    output_image = cv2.resize(output_image, (frame_width, frame_height))
    cv2.imshow('frame', output_image.astype(np.uint8))
    
    frame_counter += 1

    if cv2.waitKey(1) == 27:
      break

cap.release()
cv2.destroyAllWindows()