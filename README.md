# Program overview

## Setup

Recommended python version: 3.8

```
# install requirements
pip install -r requirements.txt
```

## Train model

### Prepare data
First download dataset from: https://www.kaggle.com/datasets/jangedoo/utkface-new
Extract `UTKFace` to the `./dataset` directory. You can generate train and test datasets by running `./dataset/split.ipynb` file.
`./dataset` directory should have the following structure
```
dataset
├── faces_train
├── faces_test
├── UTKFace
```
Once train and test datasets are generated you can remove `UTKFace`.

### Run training
To run the training of the model run:
```
python -m src.train.main -f <folds_no, eg. 10> -e <epochs_no, eg. 10> -l <learning_rate, eg. 0.001>
```

To get more information regarding arguments run:
```
python -m src.train.main -h
```

## Test model
To test the model run the command:
```
python -m src.test.main -c "./log/checkpoints"
```
To get more information regarding arguments run:
```
python -m test.main -h
```

## Run realtime face age prediction app
Run this command to start capturing camera frames and predict faces age based on the saved model in the `<path>` path. 
```
python -m src.app.main -c <path>
```
To quit the app press `q`.