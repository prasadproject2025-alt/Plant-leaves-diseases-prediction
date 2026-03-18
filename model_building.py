import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# Load number of classes dynamically
num_classes = int(np.load("num_classes.npy"))

# Squash Activation Function
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# Capsule Layer
class CapsuleLayer(Layer):
    def __init__(self, num_capsules, dim_capsules, num_routing=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.num_routing = num_routing

    def build(self, input_shape):
        input_dim = input_shape[-1]  # ✅ Correctly taking last dimension (1280)
        
        # ✅ Correct shape: (input_dim, num_capsules, dim_capsules)
        self.W = self.add_weight(
            shape=[input_dim, self.num_capsules, self.dim_capsules],
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs, training=None):
        batch_size = K.shape(inputs)[0]  # Get batch size dynamically

        # ✅ Correct einsum equation: (batch, 1, 1280) x (1280, num_capsules, dim_capsules) -> (batch, 1, num_capsules, dim_capsules)
        u_hat = tf.einsum('bij,jkl->bikl', inputs, self.W)

        # Routing Algorithm
        b = K.zeros_like(K.sum(u_hat, axis=-1))  # (batch, 1, num_capsules)
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=2)  # Routing weights
            s = K.sum(c[..., None] * u_hat, axis=1)  # Weighted sum
            v = squash(s)  # Squashing activation
            if i < self.num_routing - 1:
                b += K.batch_dot(v, u_hat, axes=[-1, -1])  # Update routing logits

        return v  # (batch, num_capsules, dim_capsules)

# Model Definition
input_layer = Input(shape=(224, 224, 3))

efficientnet = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=input_layer)
efficientnet.trainable = False  

features = GlobalAveragePooling2D()(efficientnet.output)  # Shape: (batch, 1280)
features = Reshape((1, 1280))(features)  # ✅ Ensures correct shape for CapsuleLayer

primary_caps = CapsuleLayer(num_capsules=4, dim_capsules=4)(features)  # (batch, 1, 4, 4)

# ✅ Correct Reshape using tf.squeeze() to remove dimension 1
reshaped_caps = tf.squeeze(primary_caps, axis=1)  # (batch, 4, 4)

# Pass reshaped data into LSTM
lstm_layer = LSTM(32, return_sequences=False)(reshaped_caps)

output_layer = Dense(num_classes, activation="softmax")(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.save("hybrid_model.keras")
print("✅ Model saved successfully!")
