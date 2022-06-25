import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pretty_midi
import librosa
import librosa.display

from preprocess import load_data


if __name__ == "__main__":
    fpath = "../data/ChordGAN"
    genre = "pop"

    dataset, (songs, names) = load_data(fpath, genre=genre, shuffle=False)
