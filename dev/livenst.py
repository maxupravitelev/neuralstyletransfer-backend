import cv2
import time
import functools
import os

import tensorflow as tf
import tensorflow_hub as hub

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
frame_width = 1024
frame_height = 768

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

time.sleep(1)
fps = 1 / 2

#Source: https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
def img_scaler(image, max_dim = 512):

  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio * 1, tf.int32)

  # Resizes the image based on the scaling constant generated above
  img = tf.image.resize(image, new_shape)
  return img[tf.newaxis, ...]

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)
  #img = crop_center(img)
  img = img_scaler(img)

  #img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  #return img[tf.newaxis, ...]
  return img

@functools.lru_cache(maxsize=None)
def load_tensor(tensor):


  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  # img = tf.io.decode_image(
  #     tf.io.read_file(image_path),
  #     channels=3, dtype=tf.float32)

  #img = img_scaler(tensor)
  img = tensor

  #img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img[tf.newaxis, ...]

output_image_size = 384  # @param {type:"integer"}
style_img_size = (256, 256)  # Recommended to keep it at 256.
i = 1
style_image_url = f"images/f/{i}.jpg"  # @param {type:"string"}


while (cap.isOpened()):
    time.sleep(fps)

    _, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
    #print(frame.shape)
    #print(frame_tensor.shape)

    content_image = img_scaler(frame_tensor)
    print(content_image.shape)
    style_image = load_image(style_image_url, style_img_size)
    print(style_image.shape)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    #stylized_image = outputs

    stylized_image = outputs[0]
    squeezed_image = tf.squeeze(stylized_image)
    #tf.keras.preprocessing.image.save_img(f"output/k/{i}.jpg", squeezed_image)

    cv2.imshow('frame', squeezed_image)
    #print(frame.shape)

    if cv2.waitKey(1) == 27:
      break

cap.release()
cv2.destroyAllWindows()