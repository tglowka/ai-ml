import argparse
import os

import pyrootutils
import pytorch_lightning as pl

from src.test.model_tester import ModelTester

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--checkpoints_dir", help="Directory of checkpoints to load")
    arg_parser.add_argument("-d", "--debug", action="store_true",
                            help='Flag to run in debug mode, either "True" or "False". Debug mode reads data from the '
                                 '"data/face_age_dataset_debug"', )

    args = arg_parser.parse_args()

    debug = args.debug
    dir = args.checkpoints_dir
    ckpt_paths = [os.path.join(dir, x) for x in os.listdir(dir)]

    return ckpt_paths, debug


def main():
    (ckpt_paths, debug) = parse_args()

    pl.seed_everything(42)

    result = ModelTester(ckpt_paths=ckpt_paths, debug=debug).test()
    print("Result: ", result)


if __name__ == "__main__":
    main()
