# Initial setup to be able to load `src.cyclegan`
import sys
import os
import logging
from argparse import ArgumentParser
from yaml import safe_load, YAMLError

from preprocess import load_data
from reverse_pianoroll import piano_roll_to_pretty_midi
from model import ChordGAN

logger = logging.getLogger("convert_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s : %(name)s [%(levelname)s] : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress tensorflow info logs


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
    args.add_argument(
        "fpath",
        type=str,
        help="Path to midi files of the target genre to be converted.",
    )
    args.add_argument("genre", type=str, help="Name of the genre being converted.")
    args.add_argument("model_path", type=str, help="Path to a trained CycleGAN model.")
    args.add_argument(
        "--outpath", default="converted", type=str, help="Path to output location."
    )
    args.add_argument(
        "--config_fname",
        default="config.yaml",
        type=str,
        help="Name of the yaml config file.",
    )
    return args.parse_args(argv)


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
    """The main function for training."""
    args = parse_args(argv)
    fpath = args.fpath
    genre = args.genre
    model_path = args.model_path
    config_fname = args.config_fname

    outpath = args.outpath

    # debug args
    # fpath = "../data/chordGAN"  # dummy dir with less data
    # genre = "pop"
    # model_path = "trained_models/jazz_run_2022_07_07-11_02_11"

    model_name = model_path.split("/")[-1]
    model_fpath = os.path.join(os.getcwd(), model_path, "weights", "")

    # load config
    config_path = os.path.join(os.getcwd(), model_path, config_fname)
    preprocess_params, model_params, _, _ = load_config(config_path)

    # prepare dataset
    note_range = (preprocess_params.pop("low_note"), preprocess_params.pop("high_note"))
    preprocess_params["note_range"] = note_range
    dataset, (songs, names) = load_data(fpath, genre=genre, **preprocess_params)

    # Load model
    model_fpath = os.path.join(os.getcwd(), model_path, "weights", "")
    logger.debug(f"Loading model from {model_fpath}")
    model = ChordGAN(**model_params)
    model.load_weights(model_fpath)
    logger.debug(f"\tsuccess!")

    # Create output paths
    outpath = os.path.join(outpath, model_name, genre)
    os.makedirs(outpath, exist_ok=True)
    logger.info(f"Converting and saving results to {outpath}")

    for idx, ((_, chroma), name) in enumerate(zip(dataset, names)):
        name = os.path.split(name)[-1].split(".")[0]

        # To obtain the output in the same format as the original song, we need to
        # concatenate the various phrases which the chroma was split into
        chroma = chroma.numpy().reshape(-1, 12)
        transfer = model(chroma)
        transfer.write(f"{outpath}/{name}_{idx}_transfer.midi")


if __name__ == "__main__":
    main(sys.argv[1:])
