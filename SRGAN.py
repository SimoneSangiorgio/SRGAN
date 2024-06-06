import tensorflow as tf
from tensorflow import keras
from PIL import Image
from PIL import ImageEnhance, ImageOps

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse



training_images_path = "C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/training/"
testing_images_path = "C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/testing/"
res_im_path = "C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/results/"
models_path = "C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/models/"



def residual_block(x):

    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)
    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)
    res = Add()([res, x])
    
    return res
def build_generator():

    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)
    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)
    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)
    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)
    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)
    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)
    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model



def build_discriminator():

    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)
    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)
    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)
    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)
    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)
    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)
    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)
    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)
    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)
    # Last dense layer - for classification
    output = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model



def build_vgg():
    
    input_shape = (256, 256, 3)

    vgg = keras.applications.VGG19(include_top = False ,  input_shape = input_shape , weights="imagenet")
    features = vgg.get_layer(index = 9).output

    model = keras.Model(inputs=[vgg.inputs], outputs=[features])
    return model



def input_pipeline(data_path , batch_size , highres_shape , lowres_shape):
  
  all_images = glob.glob(data_path + "*")
  cntall = len(all_images)

  def gen():
    while True:
      
      all_highres = []
      all_lowres = []
      
      idxes = np.random.choice(cntall , batch_size , replace = False)
      for idx in idxes:

        fname = all_images[idx]

        orig = Image.open(fname)

        # Data augmentation
        if np.random.random() < 0.5:
          # Adjust brightness
          enhancer = ImageEnhance.Brightness(orig)
          orig = enhancer.enhance(np.random.uniform(0.7, 1.3))

        if np.random.random() < 0.5:
          # Rotate image
          orig = orig.rotate(np.random.uniform(-15, 15))

        if np.random.random() < 0.5:
          # Zoom image
          x_center = orig.width / 2
          y_center = orig.height / 2
          width = orig.width * np.random.uniform(0.9, 1.1)
          height = orig.height * np.random.uniform(0.9, 1.1)
          left = x_center - width / 2
          top = y_center - height / 2
          right = x_center + width / 2
          bottom = y_center + height / 2
          orig = orig.crop((left, top, right, bottom))

        high_img = orig.resize(highres_shape , resample=Image.BICUBIC)
        low_img = orig.resize(lowres_shape , resample=Image.BICUBIC)

        if np.random.random() < 0.5:
          high_img = ImageOps.mirror(high_img)
          low_img = ImageOps.mirror(low_img)

        all_highres.append(np.asarray(high_img , dtype = np.float32))
        all_lowres.append(np.asarray(low_img , dtype = np.float32))

        high_res_ret = np.array(all_highres)/127.5 - 1
        low_res_ret = np.array(all_lowres)/127.5 - 1

      yield (high_res_ret , low_res_ret)

  return tf.data.Dataset.from_generator(gen , (tf.float32 , tf.float32)).prefetch(5)



def save_images(data_path , lowres , highres , orig):
  lowres = np.squeeze( (lowres.numpy() + 1)/2.0 )
  highres = np.squeeze( (highres + 1)/2.0 )
  orig = np.squeeze( (orig.numpy() + 1)/2.0 )

  fig = plt.figure(figsize=(12 , 4))

  ax = fig.add_subplot(1, 3, 1)
  ax.imshow(lowres)
  ax.axis("off")
  ax.set_title("Low-resolution")

  ax = fig.add_subplot(1, 3, 2)
  ax.imshow(orig)
  ax.axis("off")
  ax.set_title("Original")

  ax = fig.add_subplot(1, 3, 3)
  ax.imshow(highres)
  ax.axis("off")
  ax.set_title("Generated")

  plt.savefig(data_path)


#Parameters of Training
mode = 'evaluate'
epochs = 10000
batch_size = 1
common_optimizer = Adam(0.0002, 0.5)
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

# Build and compile the VGG19
vgg = build_vgg()
vgg.trainable = False
vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

#Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

# Build the generator network
generator = build_generator()

"""Build and compile the adversarial model"""

# Input layers for high-resolution and low-resolution images
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)

# Generate high-resolution images from low-resolution images
generated_high_resolution_images = generator(input_low_resolution)

# Extract feature maps of the generated images
features = vgg(generated_high_resolution_images)

# Get the probability of generated high-resolution images
probs = discriminator(generated_high_resolution_images)

# Create an adversarial model
adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])

# Get the list of trainable variables
variables = adversarial_model.trainable_variables

# Build the optimizer with the list of trainable variables
common_optimizer.build(variables)

# Compile the adversarial model
adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)



if mode == 'train':

  def just_train():

      dataloader = iter(input_pipeline(training_images_path , batch_size , high_resolution_shape[:2] , low_resolution_shape[:2]))

      for epoch in range(epochs):

        """Train the discriminator network"""

        # Sample a batch of images
        high_resolution_images, low_resolution_images = next(dataloader)

        # Generate high-resolution images from low-resolution images
        generated_high_resolution_images = generator.predict(low_resolution_images)

        # Generate batch of real and fake labels
        real_labels = np.ones((batch_size, 16, 16, 1))
        fake_labels = np.zeros((batch_size, 16, 16, 1))

        # Train the discriminator network on real and fake images
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)

        d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)


        # Calculate total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        """Train the generator network"""

        # Sample a batch of images
        high_resolution_images, low_resolution_images = next(dataloader)

        # Extract feature maps for real high-resolution images
        image_features = vgg.predict(high_resolution_images)

        # Train the generator network
        g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],[real_labels, image_features])

        print("Epoch {} : g_loss: {} , d_loss: {}".format(epoch+1 , g_loss[0] , d_loss[0]))

        # Save image of first epoch
        if (epoch+1) == 1:
            high_resolution_images, low_resolution_images = next(dataloader)

            # Normalize image
            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(res_im_path + "img_{}_{}".format(epoch+1, index),low_resolution_images[index], generated_images[index] , high_resolution_images[index])

        # Sample and save images after every 100 epochs
        if (epoch+1) % 100 == 0:
            high_resolution_images, low_resolution_images = next(dataloader)

            # Normalize images
            generated_images = generator.predict_on_batch(low_resolution_images)

            for index, img in enumerate(generated_images):
                save_images(res_im_path + "img_{}_{}".format(epoch+1, index),low_resolution_images[index], generated_images[index] , high_resolution_images[index])

            # Save models
            generator.save_weights(models_path+"generator_{}.h5".format(epoch+1))
            discriminator.save_weights(models_path+"discriminator_{}.h5".format(epoch+1))

  just_train()



if mode == "evaluate":

  generator.load_weights(os.path.join(models_path , "generator_9700.h5"))

  dataloader = iter(input_pipeline(testing_images_path , batch_size , high_resolution_shape[:2] , low_resolution_shape[:2]))

  for i in range(1):
      high_resolution_images, low_resolution_images = next(dataloader)
      generated_images = generator.predict_on_batch(low_resolution_images)
      save_images("C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/imm/" + "img{}.jpeg".format(i), low_resolution_images[0], generated_images[0] , high_resolution_images[0])

      upscaled_images_bilinear = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
      upscaled_images_bicubic = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      upscaled_image_lanczos = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

      save_images("C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/imm/" + "img{}_bilinear.jpeg".format(i), low_resolution_images[0], upscaled_images_bilinear, high_resolution_images[0])
      save_images("C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/imm/" + "img{}_bicubic.jpeg".format(i), low_resolution_images[0], upscaled_images_bicubic, high_resolution_images[0])
      save_images("C:/Users/simon/OneDrive/Desktop/Super Resolution/SRGAN/imm/" + "img{}_lanczos.jpeg".format(i), low_resolution_images[0], upscaled_image_lanczos, high_resolution_images[0])

      high_resolution_images_resized = cv2.resize(np.array(high_resolution_images[0]), (upscaled_images_bilinear.shape[1], upscaled_images_bilinear.shape[0]))
      
      psnr_value_bilinear = psnr(high_resolution_images_resized, np.array(upscaled_images_bilinear))
      mse_value_bilinear = mse(high_resolution_images_resized, np.array(upscaled_images_bilinear))

      psnr_value_bicubic = psnr(high_resolution_images_resized, np.array(upscaled_images_bicubic))
      mse_value_bicubic = mse(high_resolution_images_resized, np.array(upscaled_images_bicubic))

      psnr_value_lanczos = psnr(high_resolution_images_resized, np.array(upscaled_image_lanczos))
      mse_value_lanczos = mse(high_resolution_images_resized, np.array(upscaled_image_lanczos))

      psnr_value = psnr(np.array(high_resolution_images), np.array(generated_images))    
      mse_value = mse(np.array(high_resolution_images), np.array(generated_images))

      print(f"SRGAN - PSNR: {psnr_value}, MSE: {mse_value}")
      print(f"Bilinear - PSNR: {psnr_value_bilinear}, MSE: {mse_value_bilinear}")
      print(f"Bicubic - PSNR: {psnr_value_bicubic}, MSE: {mse_value_bicubic}")
      print(f"Lanczos - PSNR: {psnr_value_lanczos}, MSE: {mse_value_lanczos}")


  
  psnr_total = 0
  mse_total = 0
  psnr_total_bilinear = 0
  mse_total_bilinear = 0
  psnr_total_bicubic = 0
  mse_total_bicubic = 0
  psnr_total_lanczos = 0
  mse_total_lanczos = 0
  num_images = 100

  for i in range(num_images):
      high_resolution_images, low_resolution_images = next(dataloader)
      generated_images = generator.predict_on_batch(low_resolution_images)
      upscaled_images_bilinear = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
      upscaled_images_bicubic = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      upscaled_image_lanczos = cv2.resize(np.array(low_resolution_images[0]), dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

      high_resolution_images_resized = cv2.resize(np.array(high_resolution_images[0]), (upscaled_images_bilinear.shape[1], upscaled_images_bilinear.shape[0]))
      
      psnr_value_bilinear = psnr(high_resolution_images_resized, np.array(upscaled_images_bilinear))
      mse_value_bilinear = mse(high_resolution_images_resized, np.array(upscaled_images_bilinear))

      psnr_total_bilinear += psnr_value_bilinear
      mse_total_bilinear += mse_value_bilinear

      psnr_value_bicubic = psnr(high_resolution_images_resized, np.array(upscaled_images_bicubic))
      mse_value_bicubic = mse(high_resolution_images_resized, np.array(upscaled_images_bicubic))

      psnr_total_bicubic += psnr_value_bicubic
      mse_total_bicubic += mse_value_bicubic

      psnr_value_lanczos = psnr(high_resolution_images_resized, np.array(upscaled_image_lanczos))
      mse_value_lanczos = mse(high_resolution_images_resized, np.array(upscaled_image_lanczos))

      psnr_total_lanczos += psnr_value_lanczos
      mse_total_lanczos += mse_value_lanczos      

      psnr_value = psnr(np.array(high_resolution_images), np.array(generated_images))
      mse_value = mse(np.array(high_resolution_images), np.array(generated_images))
      
      psnr_total += psnr_value
      mse_total += mse_value

  psnr_average_bilinear = psnr_total_bilinear / num_images
  mse_average_bilinear = mse_total_bilinear / num_images

  psnr_average_bicubic = psnr_total_bicubic / num_images
  mse_average_bicubic = mse_total_bicubic / num_images

  psnr_average_lanczos = psnr_total_lanczos / num_images
  mse_average_lanczos = mse_total_lanczos / num_images

  psnr_average = psnr_total / num_images
  mse_average = mse_total / num_images

  print(f"SRGAN - Average PSNR: {psnr_average}, Average MSE: {mse_average}")
  print(f"Bilinear - Average PSNR: {psnr_average_bilinear}, Average MSE: {mse_average_bilinear}")
  print(f"Bicubic - Average PSNR: {psnr_average_bicubic}, Average MSE: {mse_average_bicubic}")
  print(f"Lanczos - Average PSNR: {psnr_average_lanczos}, Average MSE: {mse_average_lanczos}")




if mode == "generate":

  generator.load_weights(os.path.join(models_path , "generator_9700.h5"))

  def divide_image(input_image_path):
      image = Image.open(input_image_path).convert('RGB')
      width, height = image.size
      print(f"Original Dimensions: {width}x{height}")
      print(f"Upscaled Dimensions: {width*4}x{height*4}")

      # Calculate dimensions with padding
      padded_width = (width + 47) // 48 * 48
      padded_height = (height + 47) // 48 * 48
      padding_pixels_width = padded_width - width
      padding_pixels_height = padded_height - height

      # Create a new image with reflection padding
      padded_image_array = np.pad(image, ((0, padding_pixels_height), (0, padding_pixels_width), (0, 0)), mode='edge')
      padded_image = Image.fromarray(padded_image_array)


      # Divides the image into 62x62 batches and applies upscaling
      pieces = []
      for i in range(0, padded_height, 48):
          for j in range(0, padded_width, 48):
              box = (j, i, j+48, i+48)
              piece = padded_image.crop(box)
              piece = np.pad(piece, ((8, 8), (8, 8), (0, 0)), mode='edge')                
              upscaled_piece = np.asarray(piece, dtype=np.float32) / 127.5 - 1
              upscaled_piece = generator.predict(np.expand_dims(upscaled_piece, axis=0))
              upscaled_piece = np.squeeze(upscaled_piece, axis=0)    
              upscaled_piece = Image.fromarray(((upscaled_piece + 1) * 127.5).astype('uint8')) 
              upscaled_piece = upscaled_piece.crop((32, 32, 224, 224))
              upscaled_piece = np.asarray(upscaled_piece, dtype=np.float32) / 127.5 - 1


              pieces.append(upscaled_piece)



      return pieces, width, height


  def reassemble_images(pieces, original_width, original_height):
      # Calculates the size of the final image
      image_width = pieces[0].shape[1] 
      image_height = pieces[0].shape[0]
      num_images = len(pieces)
      num_images_per_row = (original_width + 47) // 48
      total_width = image_width * num_images_per_row
      total_height = image_height * (num_images // num_images_per_row)

      # Create a new image to contain all images
      new_image = Image.new('RGB', (total_width, total_height))

      # Paste each image in the right place
      for index, piece in enumerate(pieces):
          x = (index % num_images_per_row) * image_width
          y = (index // num_images_per_row) * image_height
          piece = Image.fromarray(((piece + 1) * 127.5).astype('uint8')) 
          new_image.paste(piece, (x, y))

      # Removes padding
      new_image = new_image.crop((0, 0, original_width*4, original_height*4))

      # shows the final image
      new_image.show()


  pieces, original_width, original_height = divide_image("C:/Users/simon/OneDrive/Desktop/images.jpg")
  reassemble_images(pieces, original_width, original_height)





