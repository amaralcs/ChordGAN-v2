import sys
import os
import time
import io
from shutil import copy
from argparse import ArgumentParser
from yaml import safe_load, YAMLError
import matplotlib.pyplot as plt
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    TensorBoard,
    LambdaCallback,
)

from preprocess import load_data
from model import ChordGAN


logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def parse_args(argv):
    """Parses input options for this module.

    Parameters
    ----------
    argv : List
        List of input parameters to this module

    Returns
    -------
    Argparser
    """
    args = ArgumentParser()
    args.add_argument("fpath", type=str, help="Path to the dataset of the given genre.")
    args.add_argument("genre", type=str, help="Name of the genre used for training")
    args.add_argument(
        "--model_output",
        default="trained_models",
        type=str,
        help="Name of directory to output model.",
    )
    args.add_argument(
        "--config_path",
        default="src/config.yaml",
        type=str,
        help="Path to YAML file containing model configuration.",
    )
    args.add_argument(
        "--root_logdir",
        default="train_logs",
        type=str,
        help="Name of directory to output training logs.",
    )
    return args.parse_args(argv)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def setup_progress_logger(model, dataset, song_names, img_writer, sample_size=10):
    """Decorator around to setup log_progress function with the correct model and
    tf.image_writer
    """
    sample_piano_rolls, sample_chromas = [], []
    for piano_roll, chroma in dataset.take(sample_size):
        sample_piano_rolls.append(piano_roll)
        sample_chromas.append(chroma)

    sample_names = song_names[:sample_size]

    def log_progress(epoch, logs):
        """Logs the learning progress in the form of treating the midi files as images."""
        display_length = 412
        transfer_outputs = [model.generator(chroma) for chroma in sample_chromas]

        fig, axes = plt.subplots(10, 3, figsize=(15, 20))

        for (ax1, ax2, ax3), original, input_chroma, transfer, name in zip(
            axes, sample_piano_rolls, sample_chromas, transfer_outputs, sample_names
        ):
            song_name = os.path.split(name)[-1].replace(".mid", "")

            ax1.imshow(original[0, :display_length].numpy().T)
            ax1.set_title(song_name)
            ax1.axis("off")

            ax2.imshow(input_chroma[0, :display_length].numpy().T)
            ax2.set_title("input_chroma")
            ax2.axis("off")

            ax3.imshow(transfer[0, :display_length].numpy().T, cmap="viridis")
            ax3.set_title("transfer")
            ax3.axis("off")

        fig.tight_layout()
        with img_writer.as_default():
            tf.summary.image("Genre transfer progress", plot_to_image(fig), step=epoch)

    return log_progress


def get_run_logdir(root_logdir, genre):
    """Generates the paths where the logs for this run will be saved.

    Parameters
    ----------
    root_logdir : str
        The base path to use.
    genre : str
        The name of genre being used for training.

    Returns
    -------
    str, str
        The full path to the logging directory as well as the name of the current run.
    """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    model_info = "{}_{}".format(genre, run_id)
    return os.path.join(root_logdir, model_info), model_info


def load_config(config_path):
    """Loads the model config from the given path

    Parameters
    ----------
    config_path : str
        Path to yaml file containing model configuration parameters.

    Returns
    -------
    Tuple(dict, dict, dict, dict)
        Dictionaries with parameters corresponding to preprocessing, model, training and
        optimizer respectively.
    """
    with open(config_path, "r") as config_file:
        try:
            config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return (
        config["preprocessing"],
        config["model"],
        config["training"],
        config["optimizer"],
    )


def main(argv):
    """The main function for training.

    Note that this script only takes arguments related to the training. To change the model architecture,
    change the settings in config.yaml
    """
    args = parse_args(argv)
    fpath = args.fpath
    genre = args.genre
    model_output = args.model_output
    config_path = args.config_path
    root_logdir = args.root_logdir

    # debug args
    # fpath = "../data/chordGAN"
    # genre = "classical"

    log_dir, model_info = get_run_logdir(root_logdir, genre)
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    preprocess_params, model_params, training_params, optimizer_params = load_config(
        config_path
    )
    batch_size = training_params["batch_size"]
    epochs = training_params["epochs"]

    optimizer_config = {
        "class_name": optimizer_params.pop("class_name"),
        "config": optimizer_params,
    }
    optimizer = keras.optimizers.get(optimizer_config)

    note_range = (preprocess_params.pop("low_note"), preprocess_params.pop("high_note"))
    preprocess_params["note_range"] = note_range
    dataset, (_, names) = load_data(fpath, genre=genre, **preprocess_params)

    # Setup model
    model = ChordGAN(**model_params)
    model.compile(d_optimizer=optimizer, g_optimizer=optimizer)

    # Setup training monitoring
    img_writer = tf.summary.create_file_writer(log_dir + "/img")
    log_progress = setup_progress_logger(model, dataset, names, img_writer)
    callbacks = [TensorBoard(log_dir), LambdaCallback(on_epoch_end=log_progress)]

    model.fit(dataset, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save_weights(f"{model_output}/{model_info}/weights/")

    copy(config_path, f"{model_output}/{model_info}/")


if __name__ == "__main__":
    main(sys.argv[1:])
