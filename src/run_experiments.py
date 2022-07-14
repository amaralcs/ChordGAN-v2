import sys
import os
import logging
from datetime import datetime
from itertools import product
from yaml import safe_load, YAMLError, dump
from argparse import ArgumentParser

from train import main as train_pipeline
from convert import main as style_transfer
from evaluate import main as eval_experiment

logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s : %(name)s [%(levelname)s] : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress tensorflow warning logs


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
    args.add_argument(
        "train_genre", type=str, help="Name of the genre used for training."
    )
    args.add_argument(
        "bodhi_transfer",
        type=str,
        help="Name of the Bodhidharma Genre to perform style transfer.",
    )
    args.add_argument(
        "bodhi_reference",
        type=str,
        help="Name of the Bodhidharma Genre equivalent to the trained genre.",
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
    dict
    """
    with open(config_path, "r") as config_file:
        try:
            config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return config


def dict_cartesian_product(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


def get_latest_trained_model():
    logger.info(f"Finding most recently trained model")
    model_info = [
        (fpath, datetime.strptime(fpath[-19:], "%Y_%m_%d-%H_%M_%S"))
        for fpath in os.listdir("trained_models")
    ]
    recent = sorted(model_info, key=lambda info: info[1])[-1]
    logger.info(f"Found {recent[0]}")
    return recent[0]


def main(argv):
    args = parse_args(argv)
    fpath = args.fpath
    train_genre = args.train_genre
    bodhi_transfer = args.bodhi_transfer
    bodhi_reference = args.bodhi_reference

    config_path = "src/config.yaml"
    settings = {"fs": [8, 16, 32], "n_bars": [1, 2, 4, 8]}
    config_outfile = "src/train_config.yaml"

    bodhidharma_path = "../data/bodhidharma"
    convert_outpath = "converted/bodhidharma"

    results_outpath = "results/bodhidharma"

    for conf in dict_cartesian_product(**settings):
        logger.debug(f"New conf dict: {conf}")
        cur_config = load_config(config_path)

        for option, new_value in conf.items():
            logger.debug(f"Setting {option}:{new_value}")
            cur_config["preprocessing"][option] = new_value

        logger.info(f"New preprocessing config: {cur_config['preprocessing']}")

        logger.info(f"Saving {config_outfile}")
        with open(config_outfile, "w") as f:
            dump(cur_config, f)

        logger.info("Training with new config...")
        train_pipeline([fpath, train_genre, "--config_path", config_outfile])
        logger.info("[Done]")

        model_name = get_latest_trained_model()
        model_fpath = f"trained_models/{model_name}"

        style_transfer(
            [
                bodhidharma_path,
                bodhi_transfer,
                model_fpath,
                "--outpath",
                convert_outpath,
                "--config_fname",
                "train_config.yaml",
            ]
        )
        eval_experiment(
            [
                os.path.join(bodhidharma_path, bodhi_transfer),
                os.path.join(bodhidharma_path, bodhi_reference),
                convert_outpath,
                model_name,
                "--outpath",
                results_outpath,
            ]
        )


if __name__ == "__main__":
    main(sys.argv[1:])
