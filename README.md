# ChordGAN-v2

A re-work of the [ChordGAN Music Style Transfer model](https://github.com/conanlu/chordgan).

## Features

- [✔️] TensorFlow 2
- [✔️] Documented code
- [✔️] Automated script for style transfer
- [✔️] Automated script for evaluating transfer results
  
## Usage

### Setup
1. Clone the git repo 
2. 
   a) download the files from the [original paper](https://conanlu.github.io/chordgan/datasets)
   b) Provide own files. 
   c) Alternatively use [Midiworld-scraper](https://github.com/amaralcs/midiworld-scraper) to fetch midi files from [midiworld](https://midiworld.com).

Create a directory where each subfolder is named with the genre of the midi files it contains. For example:
```
data
 |
 |-- pop
 |    |
 |    |-- file1.mid
 |    |-- file2.mid
 |    |-- ...
 |-- jazz
 |    |
 |    |-- file1.mid
 |    |-- file2.mid
 |    |-- ...
 |...
```

3. Create conda environment and activate it
```sh
conda env create -n chordgan --file chordgan_env.yaml
conda activate chordgan
```

### Train a model
Set the preprocessing, training and model configurations in `src/config.yaml` then run:
```sh
python src/train.py \
    path/to/your/data \             # Path to directory containing midi files
    genre_name \                    # Name of genre/folder to train on
    --model_output trained_models \ # Where to store weights
    --config_path src/config.yaml \ # Path to config file
    --root_logdir train_logs        # Where to store training logs
```
The `trained_models` folder will contain a copy of the `config.yaml` used for initializing the model and processing the data. This is used for reloading the model when performing style tranfer.

### Style transfer
Once you have a trained model, you can use it to convert songs from another genre into the genre of the trained model:
```sh
python src/convert.py \
    path/to/your/data \             # Path to directory containing midi files
    genre_name \                    # Name of genre/folder to convert
    path/to/trained/model \         # Path to a trained model
    --outpath converted \           # Where to store the transfer results
    --config_fname config.yaml      # Name of the config file used during training
```

### Evaluating Results
*Note:* To evaluate the results of transfer, we need three inputs:
- Pre transfer songs (i.e. the inputs to the style transfer step)
- Post transfer songs
- Reference songs

If you would like to compare results of various runs you can use:
```sh
python src/evaluate.py \
    path/to/pre/transfer \          # The original files from the genre you wish to convert
    path/to/reference    \          # Songs from the same genre as the model was trained on
    path/to/post/transfer \         # Outputs of style transfer (previous step)
    path/to/trained/model \         # Path to a trained model
    --outpath results               # Where to save the results
```

#### Metrics for evaluation
| Metric                | Reference                                                                                                                                                                                                                                                                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chroma Similarity     | Lu, W. T. and Su, L. (2018) ‘Transferring the style of homophonic music using recurrent neural networks and autoregressive models’, in Proceedings of the 19th International Society for Music Information Retrieval Conference, ISMIR 2018, pp. 740–746. Available at: https://github.com/s603122001/Music-Style-Transfer (Accessed: 25 June 2022). |
| Tonnetz Distance      | Lu, C. and Dubnov, S. (2021) ‘ChordGAN: Symbolic Music Style Transfer with Chroma Feature Extraction’.                                                                                                                                                                                                                                               |
| Time-pitch difference | Cifka, O., Simsekli, U. and Richard, G. (2020) ‘Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data’, IEEE/ACM Transactions on Audio Speech and Language Processing, 28, pp. 2638–2650. doi: 10.1109/TASLP.2020.3019642.                                                                                               |
| Onset-duration        | Cifka, O., Simsekli, U. and Richard, G. (2020) ‘Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data’, IEEE/ACM Transactions on Audio Speech and Language Processing, 28, pp. 2638–2650. doi: 10.1109/TASLP.2020.3019642.                                                                                               |
