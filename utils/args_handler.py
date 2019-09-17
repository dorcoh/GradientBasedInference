import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Gradient based inference on SRL task using OntoNotes v5.0 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', action='store', default='test', type=str,
                        choices=['train', 'test', 'development', 'selected', 'fixed', 'gzero', 'failed'],
                        help="load samples from data: train/test/development, or from pickle files: fixed/failed/gzero")
    parser.add_argument('--store', action='store_true',
                        help="store on disk samples groups in pickle files, according to their classified group: "
                             "fixed/failed/gzero")
    parser.add_argument('-l', action='store', default=1, type=float, help="learning rate")
    parser.add_argument('-i', action='store', default=15, type=int, help="number of gradient inference iterations")
    parser.add_argument('-a', action='store', default=0, type=float, help="regularization parameter")
    args = parser.parse_args()

    return args