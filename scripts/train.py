import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scripts_dir = os.path.join(parent_dir, 'scripts')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

device = "cuda"

from scripts.train_approx import train_GED
from scripts.train_EC import train_EC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script training GEDAN")

    parser.add_argument("--type", type=str, help="Training Type: GED or EC")
    parser.add_argument("--dataset", type=str, help="Dataset type")
    parser.add_argument("--model", type=str, help="UGEDAN or SGEDAN")
    parser.add_argument("--idx_costs", type=str, help="Cost Index", default=0, required=False)

    args = parser.parse_args()

    type_training = args.type
    dataset_name = args.dataset

    supervised = True
    if args.model == "UGEDAN":
        supervised = False

    if type_training == "GED":
        idx_costs = args.idx_costs
        train_GED(dataset_name, supervised, idx_costs)

    if type_training == "EC":

        if supervised:
            train_EC(dataset_name)
        else:
            print("U-GEDAN cannot be trained due to edit costs")

