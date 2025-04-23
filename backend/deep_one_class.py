import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.base import BaseEstimator

class DeepOneClassClassifier(BaseEstimator):
    def __init__(self, input_dim=None, latent_dim=32, epochs=50, batch_size=32):
        """
        Deep One-Class Classifier using an autoencoder.
        :param input_dim: Dimension of the input features (can be None initially).
        :param latent_dim: Dimension of the latent space.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None

    def build_model(self, input_dim=None):
        """
        Build the autoencoder model. Rebuilds the model if input_dim changes.
        :param input_dim: Dimension of the input features (optional).
        """
        if input_dim is not None:
            self.input_dim = input_dim  # Update input_dim dynamically

        if self.input_dim is None:
            raise ValueError("input_dim must be specified to build the model.")

        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(128, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)

        # Autoencoder
        self.autoencoder = models.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def fit(self, X):
        """
        Train the autoencoder on the input data.
        :param X: Input data (features).
        """
        if self.autoencoder is None or self.input_dim != X.shape[1]:
            self.build_model(input_dim=X.shape[1])  # Dynamically rebuild the model if input_dim changes
        self.autoencoder.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def decision_function(self, X):
        """
        Compute the anomaly score for the input data.
        :param X: Input data (features).
        :return: Negative reconstruction error as the anomaly score.
        """
        if self.autoencoder is None or self.input_dim != X.shape[1]:
            self.build_model(input_dim=X.shape[1])  # Ensure the model matches the input dimensions
        reconstructions = self.autoencoder.predict(X)
        reconstruction_error = tf.reduce_mean(tf.square(X - reconstructions), axis=1)
        return -reconstruction_error.numpy()  # Negative because higher error = more anomalous