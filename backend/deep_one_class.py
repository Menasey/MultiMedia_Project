import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Normalizer

class DeepOneClassClassifier(BaseEstimator):
    def __init__(self, input_dim=None, latent_dim=16, epochs=100, batch_size=32):
        """
        Deep One-Class Classifier using an autoencoder with improved structure.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.normalizer = Normalizer()

    def build_model(self, input_dim=None):
        if input_dim is not None:
            self.input_dim = input_dim

        if self.input_dim is None:
            raise ValueError("input_dim must be specified to build the model.")

        # Encoder with bottleneck
        input_layer = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(self.latent_dim, activation='relu')(x)

        # Decoder
        x = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)

        # Autoencoder model
        self.autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mae')

    def fit(self, X):
        """
        Fit the model using normalized inputs.
        """
        X_norm = self.normalizer.fit_transform(X)
        if self.autoencoder is None or self.input_dim != X_norm.shape[1]:
            self.build_model(input_dim=X_norm.shape[1])
        self.autoencoder.fit(X_norm, X_norm, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def decision_function(self, X):
        """
        Return negative reconstruction error as the anomaly score.
        """
        X_norm = self.normalizer.transform(X)
        reconstructions = self.autoencoder.predict(X_norm)
        errors = tf.reduce_mean(tf.abs(X_norm - reconstructions), axis=1)  # MAE
        return -errors.numpy()
