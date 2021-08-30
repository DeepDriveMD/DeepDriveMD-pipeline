"""
Convolutional variational autoencoder in Keras
Reference: "Auto-Encoding Variational Bayes" (https://arxiv.org/abs/1312.6114)
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt

import numpy as np
import pandas as pd  # type: ignore[import]
import tensorflow.keras.backend as K  # type: ignore[import]
import tensorflow.keras.losses as objectives  # type: ignore[import]
from tensorflow.keras.callbacks import Callback  # type: ignore[import]
from tensorflow.keras.layers import (  # type: ignore[import]
    Conv2DTranspose,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Reshape,
)
from tensorflow.keras.models import Model  # type: ignore[import]
from tensorflow.keras.optimizers import RMSprop  # type: ignore[import]

from deepdrivemd.utils import PathLike


# save history from log
class LossHistory(Callback):  # type: ignore[misc]
    def on_train_begin(self, logs: Dict[str, float] = {}) -> None:
        self.losses: List[float] = []
        self.val_losses: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = {}) -> None:
        self.losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])

    def to_csv(self, path: PathLike) -> None:
        """Log loss values to a csv file."""
        df = pd.DataFrame({"train_loss": self.losses, "valid_loss": self.val_losses})
        df.to_csv(path, index_label="epoch")


class conv_variational_autoencoder(object):
    """Implements a convolutional variational autoencoder in Keras.

    Methods
    -------
    train(data, batch_size, epochs=1, checkpoint=False, filepath=None)
        Train network on the given data.
    save(filepath)
        Save the model weights to a file.
    load(filepath)
        Load model weights from a file.
    return_embeddings(data)
        Return the embeddings for given data.
    generate(embedding)
        Return a generated output given a latent embedding.
    """

    def __init__(  # noqa
        self,
        image_size: Tuple[int, int],
        channels: int,
        conv_layers: int,
        feature_maps: List[int],
        filter_shapes: List[Tuple[int, int]],
        strides: List[Tuple[int, int]],
        dense_layers: int,
        dense_neurons: List[int],
        dense_dropouts: List[float],
        latent_dim: int,
        activation: str = "relu",
        eps_mean: float = 0.0,
        eps_std: float = 1.0,
    ):
        """
        Parameters
        ----------
        image_size : Tuple[int, int]
            Height and width of images.
        channels : int
            Number of channels in input images.
        conv_layers : int
            Number of encoding/decoding convolutional layers.
        feature_maps : List[int]
            Number of output feature maps for each convolutional layer.
        filter_shapes : List[Tuple[int, int]]
            Convolutional filter shape for each convolutional layer.
        strides : List[Tuple[int, int]]
            Convolutional stride for each convolutional layer.
        dense_layers : int
            Number of encoding/decoding dense layers.
        dense_neurons : List[int]
            Number of neurons for each dense layer.
        dense_dropouts : List[float]
            Fraction of neurons to drop in each dense layer (between 0 and 1).
        latent_dim : int
            Number of dimensions for latent embedding.
        activation : str, default="relu"
            Activation function to use for layers.
        eps_mean : float, default=0.0
            Mean to use for epsilon (target distribution for embedding).
        eps_std : float, default=1.0
            Standard dev to use for epsilon (target distribution for embedding).

        Raises
        ------
        Exception
            :obj:`conv_layers` must equal length of :obj:`filter_shapes` list.
        Exception
            :obj:`conv_layers` must equal length of :obj:`strides` list.
        Exception
            :obj:`conv_layers` must equal length of :obj:`feature_maps` list.
        Exception
            :obj:`dense_layers` must equal length of :obj:`dense_neurons` list.
        Exception
            :obj:`dense_layers` must equal length of :obj:`dense_dropouts` list.
        """

        self.history = LossHistory()

        # tensorflow.config.experimental_run_functions_eagerly(False)

        # check that arguments are proper length
        if len(filter_shapes) != conv_layers:
            raise Exception(
                "number of convolutional layers must equal length of filter_shapes list"
            )
        if len(strides) != conv_layers:
            raise Exception(
                "number of convolutional layers must equal length of strides list"
            )
        if len(feature_maps) != conv_layers:
            raise Exception(
                "number of convolutional layers must equal length of feature_maps list"
            )
        if len(dense_neurons) != dense_layers:
            raise Exception(
                "number of dense layers must equal length of dense_neurons list"
            )
        if len(dense_dropouts) != dense_layers:
            raise Exception(
                "number of dense layers must equal length of dense_dropouts list"
            )

        # even shaped filters may cause problems in theano backend
        # even_filters = [f for pair in filter_shapes for f in pair if f % 2 == 0]
        # if K.image_dim_ordering() == 'th' and len(even_filters) > 0:
        #    warnings.warn('Even shaped filters may cause problems in Theano backend')
        # if K.image_dim_ordering() == 'channels_first' and len(even_filters) > 0:
        #    warnings.warn('Even shaped filters may cause problems in Theano backend')

        self.eps_mean = eps_mean
        self.eps_std = eps_std
        self.image_size = image_size

        # define input layer
        if K.image_data_format() == "channels_first":
            self.input = Input(shape=(channels, image_size[0], image_size[1]))
        else:
            self.input = Input(shape=(image_size[0], image_size[1], channels))

        # define convolutional encoding layers
        self.encode_conv = []
        layer = Convolution2D(
            feature_maps[0],
            filter_shapes[0],
            padding="same",
            activation=activation,
            strides=strides[0],
        )(self.input)
        self.encode_conv.append(layer)
        for i in range(1, conv_layers):
            layer = Convolution2D(
                feature_maps[i],
                filter_shapes[i],
                padding="same",
                activation=activation,
                strides=strides[i],
            )(self.encode_conv[i - 1])
            self.encode_conv.append(layer)

        # define dense encoding layers
        self.flat = Flatten()(self.encode_conv[-1])
        self.encode_dense = []
        layer = Dense(dense_neurons[0], activation=activation)(
            Dropout(dense_dropouts[0])(self.flat)
        )
        self.encode_dense.append(layer)
        for i in range(1, dense_layers):
            layer = Dense(dense_neurons[i], activation=activation)(
                Dropout(dense_dropouts[i])(self.encode_dense[i - 1])
            )
            self.encode_dense.append(layer)

        # define embedding layer
        self.z_mean = Dense(latent_dim)(self.encode_dense[-1])
        self.z_log_var = Dense(latent_dim)(self.encode_dense[-1])
        self.z = Lambda(self._sampling, output_shape=(latent_dim,))(
            [self.z_mean, self.z_log_var]
        )

        # save all decoding layers for generation model
        self.all_decoding = []

        # define dense decoding layers
        self.decode_dense = []
        layer = Dense(dense_neurons[-1], activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.z))
        for i in range(1, dense_layers):
            layer = Dense(dense_neurons[-i - 1], activation=activation)
            self.all_decoding.append(layer)
            self.decode_dense.append(layer(self.decode_dense[i - 1]))

        # dummy model to get image size after encoding convolutions
        self.decode_conv = []
        if K.image_data_format() == "channels_first":
            dummy_input = np.ones((1, channels, image_size[0], image_size[1]))
        else:
            dummy_input = np.ones((1, image_size[0], image_size[1], channels))
        dummy = Model(self.input, self.encode_conv[-1])
        conv_size = dummy.predict(dummy_input).shape
        layer = Dense(conv_size[1] * conv_size[2] * conv_size[3], activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.decode_dense[-1]))
        reshape = Reshape(conv_size[1:])
        self.all_decoding.append(reshape)
        self.decode_conv.append(reshape(self.decode_dense[-1]))

        # define deconvolutional decoding layers
        for i in range(1, conv_layers):
            if K.image_data_format() == "channels_first":
                dummy_input = np.ones((1, channels, image_size[0], image_size[1]))
            else:
                dummy_input = np.ones((1, image_size[0], image_size[1], channels))
            dummy = Model(self.input, self.encode_conv[-i - 1])
            conv_size = list(dummy.predict(dummy_input).shape)

            if K.image_data_format() == "channels_first":
                conv_size[1] = feature_maps[-i]
            else:
                conv_size[3] = feature_maps[-i]

            layer = Conv2DTranspose(
                feature_maps[-i - 1],
                filter_shapes[-i],
                padding="same",
                activation=activation,
                strides=strides[-i],
            )
            self.all_decoding.append(layer)
            self.decode_conv.append(layer(self.decode_conv[i - 1]))

        layer = Conv2DTranspose(
            channels,
            filter_shapes[0],
            padding="same",
            activation="sigmoid",
            strides=strides[0],
        )
        self.all_decoding.append(layer)
        self.output = layer(self.decode_conv[-1])

        # build model
        self.model = Model(self.input, self.output)
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # KLD loss
        self.model.add_loss(
            -0.5
            * K.mean(
                1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var),
                axis=None,
            )
        )
        self.model.compile(optimizer=self.optimizer, loss=self._vae_loss1)
        # self.model.compile(optimizer=self.optimizer)
        # self.model.compile(optimizer=self.optimizer, loss=objectives.MeanSquaredError())
        self.model.summary()

        # model for embeddings
        self.embedder = Model(self.input, self.z_mean)

        # model for generation
        self.decoder_input = Input(shape=(latent_dim,))
        self.generation = []
        self.generation.append(self.all_decoding[0](self.decoder_input))
        for i in range(1, len(self.all_decoding)):
            self.generation.append(self.all_decoding[i](self.generation[i - 1]))
        self.generator = Model(self.decoder_input, self.generation[-1])

    def _sampling(
        self, args: Tuple["npt.ArrayLike", "npt.ArrayLike"]
    ) -> "npt.ArrayLike":
        """Sampling function for embedding layer.

        Parameters
        ----------
        args : Tuple[npt.ArrayLike, npt.ArrayLike]
            The :obj:`z_mean` and :obj:`z_log_var` tensors.

        Returns
        -------
        npt.ArrayLike
            The latent codes after the reparameterization trick.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=K.shape(z_mean), mean=self.eps_mean, stddev=self.eps_std
        )
        sample: "npt.ArrayLike" = z_mean + K.exp(z_log_var) * epsilon
        return sample

    def _vae_loss1(
        self, input: "npt.ArrayLike", output: "npt.ArrayLike"
    ) -> "npt.ArrayLike":
        """Compute reconstruction loss.

        Parameters
        ----------
        input : npt.ArrayLike
            The input data.
        output : npt.ArrayLike
            The reconstructed data.

        Returns
        -------
        npt.ArrayLike
            The reconstruction loss.
        """
        input_flat = K.flatten(input)
        output_flat = K.flatten(output)
        xent_loss: "npt.ArrayLike" = (
            self.image_size[0]
            * self.image_size[1]
            * objectives.binary_crossentropy(input_flat, output_flat)
        )
        return xent_loss

    def train(
        self,
        data: "npt.ArrayLike",
        batch_size: int,
        epochs: int = 1,
        validation_data: Optional["npt.ArrayLike"] = None,
        checkpoint: bool = False,
        filepath: Optional[str] = None,
    ) -> None:
        """Train network on given data.

        Parameters
        ----------
        data : npt.ArrayLike
            Input training data.
        batch_size : int
            Batch size for training.
        epochs : int, default=1.0
            Number of epochs to train for.
        validation_data : Optional[npt.ArrayLike], optional
            Validation data to report validation accuracy during training.
        checkpoint : bool, default=False
            Whether or not to save model after each epoch.
        filepath : Optional[str], optional
            Path to save model if :obj:`checkpoint` is set to True.

        Raises
        ------
        Exception
            If :obj:`checkpoint` is :obj:`True` but :obj:`filename` is :obj:`None`.
        """
        if checkpoint and filepath is None:
            raise Exception("Please enter a path to save the network")
        # tensorflow.config.experimental_run_functions_eagerly(False)

        self.model.fit(
            data,
            data,
            batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(validation_data, validation_data),
            callbacks=[
                self.history,
                # ModelCheckpoint(
                #    "best.h5", monitor="val_loss", save_best_only=True, verbose=1
                # ),
            ],
        )

    def save(self, filepath: str) -> None:
        """Save the model weights to a file.

        Parameters
        ----------
        filepath: str
            Path to save model weights.
        """
        self.model.save_weights(filepath)

    def load(self, filepath: str) -> None:
        """Load model weights from a file.

        Parameters
        ----------
        filepath: str
            Path from which to load model weights.
        """
        self.model.load_weights(filepath)

    def decode(self, data: "npt.ArrayLike") -> "npt.ArrayLike":
        """Return the decodings for given data

        Parameters
        ----------
        data: npt.ArrayLike
            Input data.

        Returns
        -------
        npt.ArrayLike
            Array of decodings for input data.
        """
        recon: "npt.ArrayLike" = self.model.predict(data)
        return recon

    def return_embeddings(
        self, data: "npt.ArrayLike", batch_size: int = 32
    ) -> "npt.ArrayLike":
        """Return the embeddings for given data.

        Parameters
        ----------
        data: npt.ArrayLike
            Input data.
        batch_size: int
            Batch size to use during inference.

        Returns
        -------
        npt.ArrayLike
            Array of embeddings for input data.
        """
        embeddings: "npt.ArrayLike" = self.embedder.predict(data, batch_size=batch_size)
        return embeddings

    def generate(self, embedding: "npt.ArrayLike") -> "npt.ArrayLike":
        """Return a generated output given a latent embedding.

        Parameters
        ----------
        data: npt.ArrayLike
            Latent embeddings.

        Returns
        -------
        npt.ArrayLike
            Array of generated output.
        """
        preds: "npt.ArrayLike" = self.generator.predict(embedding)
        return preds
