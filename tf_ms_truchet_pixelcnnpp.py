# Build a small Pixel CNN++ model to train on maze dataset.

import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from ms_truchet import MultiScaleTruchetPattern
import numpy as np

def generate_multiscale_truchet():
    how_many_tiles = 2
    of_size = 24
    multiscaleTruchetTiling = MultiScaleTruchetPattern(how_many_tiles, of_size, 'white','black')
    while True:
        img_out = multiscaleTruchetTiling.paint_a_multiscale_truchet()
        img_out = img_out.convert("L")
        img_arr = np.asarray(img_out)
        yield img_arr[:, :, np.newaxis]

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

batch_size = 16
image_shape = (64, 64, 1)

train_data = tf.data.Dataset.from_generator(
  lambda: generate_multiscale_truchet(), 
  output_signature=(
    tf.TensorSpec(shape=image_shape, dtype=tf.float32))
  )


train_it = train_data.batch(batch_size)

# Define a Pixel CNN network
dist = tfd.PixelCNN(
    image_shape=image_shape,
    num_resnet=3,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=2, #B/W
    dropout_p=.3,
)

# Define the model input
image_input = tfkl.Input(shape=image_shape)

# Define the log likelihood for the loss fn
log_prob = dist.log_prob(image_input)

# Define the model
model = tfk.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))

# Compile and train the model
model.compile(
    optimizer=tfk.optimizers.Adam(),
    metrics=[])

model.fit(train_it, epochs=10, steps_per_epoch=100, verbose=True)

# sample five images from the trained model
samples = dist.sample(5)

for i, sample in enumerate(samples):
    tf.keras.preprocessing.image.save_img('sample'+str(i)+'.gif',sample)