#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
ROOTDIR=$SCRIPTDIR/../../
PYTHONPATH=$PYTHONPATH:$ROOTDIR
export PYTHONPATH

python $SCRIPTDIR/e2e_builder.py --name 37-slices-temp/train --path ~/workspace/OCT-DL/Data/train/control ~/workspace/OCT-DL/Data/train/study --label HEALTHY SICK
python $SCRIPTDIR/e2e_builder.py --name 37-slices-temp/eval --path ~/workspace/OCT-DL/Data/validation/control ~/workspace/OCT-DL/Data/validation/study --label HEALTHY SICK
python $SCRIPTDIR/e2e_builder.py --name 37-slices-temp/test --path ~/workspace/OCT-DL/Data/test/control ~/workspace/OCT-DL/Data/test/study --label HEALTHY SICK
