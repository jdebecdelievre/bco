from argparse import ArgumentParser
import shutil
from torch.utils.tensorboard import get_inspection_units, print_dict, get_dict_to_print

"""
Deletes all folders with small tensorboard run files
"""

parser = ArgumentParser('delete small runs')
parser.add_argument('--logdir', type=str, default='.')
parser.add_argument('--delete_smaller_than', type=int, default=18000)
parser.add_argument('--list', action='store_true')
args = parser.parse_args()

run_len = {}
inspect_units = get_inspection_units(logdir=args.logdir)


for run in inspect_units:
    path = run[0]
    max_length = 0
    for key, value in get_dict_to_print(run.field_to_obs).items():
        if value is not None:
            length = value['max_step']
            if max_length < length:
                max_length = length
    run_len[path] = max_length

for run, length in run_len.items():
    if length < args.min_run_len:
        if args.list:
            print(f'{run} is {length} steps long and so will be deleted')
        else:
            try:
                print(f'{run} is {length} and was deleted')
                shutil.rmtree(run)
            except:
                print(f"OS didn't let us delete {run}")
    else:
        print(f'{run} is {length} and is good')