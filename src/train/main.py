import argparse

import pyrootutils
import pytorch_lightning as pl

from src.train.model_trainer import ModelTrainer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", "--epochs", type=int, help="Epochs count")
    arg_parser.add_argument("-f", "--folds", type=int, help="Folds count")
    arg_parser.add_argument("-l", "--learning_rate", type=float, help="Learning rate value of the Adam optimizer", )
    arg_parser.add_argument("-d", "--debug", action="store_true",
                            help='Flag to run in debug mode. Debug mode reads data from the "data/faces_train_debug"', )

    args = arg_parser.parse_args()

    epochs = args.epochs
    folds = args.folds
    learning_rate = args.learning_rate
    debug = args.debug

    return epochs, folds, learning_rate, debug


def main():
    (epochs, folds, learning_rate, debug) = parse_args()

    pl.seed_everything(42)

    fitter = ModelTrainer(folds=folds, epoch=epochs, learning_rate=learning_rate, debug=debug)

    fitter.fit()


if __name__ == "__main__":
    main()
