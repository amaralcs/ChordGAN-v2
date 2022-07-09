from ast import parse
import sys
import os
import re
import numpy as np
import pretty_midi
from glob import glob
import json
import logging
from argparse import ArgumentParser

from eval_utils import (
    eval_chroma_similarities,
    gen_histograms,
    time_pitch_diff_hist,
    onset_duration_hist,
    eval_style_similarities,
)

logger = logging.getLogger("evaluation_logger")
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
    args.add_argument("original_fpath", type=str, help="Path to the original songs.")
    args.add_argument("transfer_fpath", type=str, help="Path to the converted songs.")
    args.add_argument("model", type=str, help="Name of model to evaluate.")
    args.add_argument("outpath", type=str, help="Path to save the results to.")

    return args.parse_args(argv)


def write_output(results, outpath, genre_a, genre_b):
    """Helper function to write the evaluation outputs.

    Parameters
    ----------
    results : dict
        Dictionary containing the results.
    outpath : str
        Path the save the evaluation results to.
    genre_a : str
        Name of genre A.
    genre_b : str
        Name of genre B.
    """
    os.makedirs(outpath, exist_ok=True)
    outfile = f"{outpath}/{genre_a}2{genre_b}_results.json"

    logger.info(f"Writing results to {outfile}")
    with open(outfile, "w") as f:
        json.dump(results, f)


def load_songs(input_fpath, target_fpath, transfer_fpath):
    """Loads the original and transferred songs from the given paths.

    Parameters
    ----------
    input_fpath : str
    target_fpath : str
    transfer_fpath : str

    Returns
    -------
    Tuple(List[prettymidi.PrettyMIDI], List[prettymidi.PrettyMIDI], List[prettymidi.PrettyMIDI])
        Containing the following:
        (songs from input genre, songs from target genre, songs from input genre converted to target genre)
    input_genre : str
        The name of the input genre, i.e. the genre that will be converted.
    target_genre : str
        The name of the target genre, i.e. genre being converted to.
    """

    # Get the names of the genres
    input_genre = os.path.split(input_fpath)[-1]
    target_genre = os.path.split(target_fpath)[-1]
    logger.debug(f"\toriginal_genre: {input_genre}")
    logger.debug(f"\transfer_genre: {target_genre}")

    # load data
    input_fpaths = glob(os.path.join(input_fpath, "*.mid*"))
    target_fpaths = glob(os.path.join(input_fpath, "*.mid*"))
    transfer_fpaths = glob(os.path.join(transfer_fpath, input_genre, "*.mid*"))

    logger.info(f"Loading inputs from {os.path.split(input_fpaths[0])[0]}")
    input_songs = [pretty_midi.PrettyMIDI(filepath) for filepath in input_fpaths]

    logger.info(f"Loading targets from {os.path.split(target_fpaths[0])[0]}")
    target_songs = [pretty_midi.PrettyMIDI(filepath) for filepath in target_fpaths]

    logger.info(f"Loading transferred from {os.path.split(transfer_fpaths[0])[0]}")
    transferred_songs = [
        pretty_midi.PrettyMIDI(filepath) for filepath in transfer_fpaths
    ]

    return (input_songs, target_songs, transferred_songs), input_genre, target_genre


def compute_style_metric(
    name,
    song_tup,
    hist_func,
    input_genre,
    target_genre,
    **kwargs,
):
    """

    Parameters
    ----------
    name : str
        Name of the metric calculation, to be used when writing the results.
        Input songs.
    song_tup : tuple(list[pretty_midi.PrettyMIDI], list[pretty_midi.PrettyMIDI], list[pretty_midi.PrettyMIDI])
        Tuple containing the following:
            (songs from input genre, songs from target genre, songs from input genre converted to target genre)
    hist_func : function
        Histogram metric to compute. One of (`time_pitch_diff_hist`. `onset_duration_hist`)
    input_genre : str
        Name of original genre.
    target_genre : str
        Name of the genre transferred to.
    kwargs :
        Keyword arguments to pass to `metric_func`.

    Returns
    -------
    List[np.array]
        The computes time-pitch histograms for each input song.
    """
    inputs, targets, transfers = song_tup

    input_histograms = gen_histograms(inputs, hist_func=hist_func, **kwargs)
    target_histograms = gen_histograms(targets, hist_func=hist_func, **kwargs)
    transfer_histograms = gen_histograms(transfers, hist_func=hist_func, **kwargs)
    input_reference_hist = input_histograms.mean(axis=0)
    targets_reference_hist = target_histograms.mean(axis=0)
    transfer_reference_hist = transfer_histograms.mean(axis=0)

    results = {
        f"macro_{name}": {
            f"{input_genre}2{target_genre}": eval_style_similarities(
                [transfer_reference_hist], targets_reference_hist
            ),
        },
        f"per_song_{name}": {
            f"{input_genre}2{target_genre}": eval_style_similarities(
                transfer_histograms, targets_reference_hist
            ),
        },
    }
    return results


def main(argv):
    """Main function to compute evaluation metrics"""
    # args = parse_args(argv)
    # original_fpath = args.original_fpath
    # transfer_fpath = args.transfer_fpath
    # model = args.model
    # outpath = args.outpath

    chroma_args = dict(sampling_rate=12, window_size=24, stride=12, use_velocity=False)
    hist_kwargs = dict(max_time=4, bin_size=1 / 6, normed=True)

    # Test args
    input_fpath = "../data/chordGAN/jazz"
    target_fpath = "../data/chordGAN/pop"
    transfer_fpath = "converted"
    model = "pop_run_2022_07_07-10_43_07"
    outpath = "results"

    transfer_fpath = os.path.join(transfer_fpath, model)
    song_tup, original_genre, transfer_genre = load_songs(
        input_fpath, target_fpath, transfer_fpath
    )

    logger.info("Computing chroma_similarities...")
    # Note that for the chroma similarities we only need the original inputs
    # and the transferred songs (indices 0 and 2 of `song_tup`)
    results = {
        "chroma_similarities": {
            f"{original_genre}2{transfer_genre}": eval_chroma_similarities(
                *song_tup[0, 2], **chroma_args
            )
        }
    }

    logger.info(f"Computing time-pitch histograms")
    time_pitch_results = compute_style_metric(
        "time_pitch_diff",
        song_tup,
        time_pitch_diff_hist,
        original_genre,
        transfer_genre,
        **hist_kwargs,
    )
    results.update(time_pitch_results)

    logger.info(f"Computing onset-duration histograms")
    onset_duration_results = compute_style_metric(
        "onset_duration",
        song_tup,
        onset_duration_hist,
        original_genre,
        transfer_genre,
    )
    results.update(onset_duration_results)

    outpath = f"{outpath}/{model}"
    write_output(results, outpath, original_genre, transfer_genre)
    logger.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
