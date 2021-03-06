# Source: https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization

import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['CUDA_VISIBLE_DEVICES']='-1'    # disable gpu

print('TF Version: ', tf.__version__)
print('TF-Hub version: ', hub.__version__)
print('Eager mode enabled: ', tf.executing_eagerly())
print('GPU available: ', tf.config.list_physical_devices('GPU'))

# @title Define image loading and visualization functions  { display-mode: 'form' }

def crop_center(image):
  '''Returns a cropped square image.'''
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

#Source: https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
def img_scaler(image, max_dim = 512):

  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio * 1, tf.int32)

  # Resizes the image based on the scaling constant generated above
  return tf.image.resize(image, new_shape)


@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  '''Loads and preprocesses images.'''
  # Cache image file locally.
  #image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  image_path = image_url
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)
  
  print(img.shape)
  #img = crop_center(img)
  img = img_scaler(img)

  #img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img[tf.newaxis, ...]


# Load TF-Hub module
#hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_handle = 'models/arbv1'
hub_module = hub.load(hub_handle)


def stylize_batch(mode, place_in_folder):

  if mode == "stylize_by_all_filters":
      foldername = 'filter'
      files_in_folder = len([file for file in os.listdir(f'{foldername}/')])
      output_folder = 'output/all_filter'
  else: 
      foldername = 'images'
      files_in_folder = len([file for file in os.listdir(f'{foldername}/')])
      output_folder = 'output/filter' + str(place_in_folder)

  if not os.path.exists(output_folder):
      os.mkdir(output_folder)

  for i in range(54, files_in_folder):

      if mode == 'batch_by_filter':

          content_image_url = f'images/frame{place_in_folder}.png' # @param {type:'string'}
          style_image_url = f'filter/{i}.jpg'  # @param {type:'string'}
    
      else:
        content_image_url = f'images/frame{i}.png' # @param {type:'string'}
        style_image_url = f'filter/{place_in_folder}.jpg'  # @param {type:'string'}
      
      #output_image_size = 384  # @param {type:'integer'}
      output_image_size = 1024  # @param {type:'integer'}


      # The content image size can be arbitrary.
      content_img_size = (output_image_size, output_image_size)
      # The style prediction model was trained with image size 256 and it's the 
      # recommended image size for the style image (though, other sizes work as 
      # well but will lead to different results).
      style_img_size = (256, 256)  # Recommended to keep it at 256.

      content_image = load_image(content_image_url, content_img_size)
      print('c')
      style_image = load_image(style_image_url, style_img_size)
      print('s')
      print(style_image.shape)
      style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

      #outputs = hub_module(content_image, style_image)
      #stylized_image = outputs[0]

      outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
      #stylized_image = outputs

      stylized_image = outputs[0]
      squeezed_image = tf.squeeze(stylized_image)
      
      tf.keras.preprocessing.image.save_img(f'{output_folder}/frame{i}.png', squeezed_image)

stylize_batch("stylize_by_all_filters", 50)