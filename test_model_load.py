import tensorflow as tf
model = tf.keras.models.load_model("models/vgg16_politician_final.h5")
print("Model loaded!")
