# Lifelong Event Detection with Knowledge Transfer
Repo for the EMNLP2021 paper: Lifelong Event Detection with Knowledge Transfer


## Data Preprocessing

We are unable to include entire datasets of MAVEN and ACE due to large size of data. We show examples of processed data in the data folded. `prepare_inputs.py` preprocesses the original MAVEN data. For ACE 2005 the data needs to be processed similarly.

For the subsets that forms incremental tasks, please refer to `data/*/streams.json` for partitions (integers to label names is in `data/*/label2id.json`), and `run_train.py: PERM` for order permutations. Since splits collection includes manual check to make sure all types are covered in development/test data, we don't include scripts for this process. For Silver Negative settings, instances are collected with `prepare_stream_instances.py`.

## Training and Testing

Run the cmd `python run_train.py` with proper arguments in order to train/test. Valid arguments can be found in `utils/options.py`. We will release more details in later updates of this repo. By default the script runs evaluation periodically during training. But it is also possible to run testing only using `--test-only`.


## Requirements:
- python == 3.7.3
- pytorch == 1.6.0
- transformers == 3.1.0
- torchmeta == 1.5.3
- numpy == 1.16.5
- tqdm == 4.48.1