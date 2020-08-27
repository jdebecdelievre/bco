import json
import os.path as op

with open(op.join(op.dirname(__file__),'default_params.json'), 'r') as f:
    default_params = json.load(f)