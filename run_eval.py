from libs.eval import Evaluator
import os
import datetime
import random
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        '--image_resolution',
        type=int,
        default=1024,
        choices=(
            256,
            1024),
        help="pre-trained GAN modeln")
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=4,
        help="set test batch size")
    parser.add_argument('--workers', type=int, default=1, help="set workers")
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help="set dataset path for evaluation")
    parser.add_argument(
        '--eval_type',
        type=str,
        default='cross')
    parser.add_argument(
        '--model_type',
        type=str,
        default='Hyper')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None)
    args = parser.parse_args()
    args = vars(args)  # convert to dictionary
    evaluator = Evaluator(args)
    evaluator.load_models()
    evaluator.configure_dataset()
    evaluator.evaluate_model_reenactment()


if __name__ == '__main__':
    main()
