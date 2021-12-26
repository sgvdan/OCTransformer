#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
ROOTDIR=$SCRIPTDIR/../../
PYTHONPATH=$PYTHONPATH:$ROOTDIR
export PYTHONPATH

python $SCRIPTDIR/kermany_builder.py --name kermany/train --path ~/workspace/datasets/kermany/train/CNV ~/workspace/datasets/kermany/train/DME ~/workspace/datasets/kermany/train/DRUSEN ~/workspace/datasets/kermany/train/NORMAL --label CNV DME DRUSEN NORMAL

python $SCRIPTDIR/kermany_builder.py --name kermany/eval --path ~/workspace/datasets/kermany/val/CNV ~/workspace/datasets/kermany/val/DME ~/workspace/datasets/kermany/val/DRUSEN ~/workspace/datasets/kermany/val/NORMAL --label CNV DME DRUSEN NORMAL

python $SCRIPTDIR/kermany_builder.py --name kermany/test --path ~/workspace/datasets/kermany/test/CNV ~/workspace/datasets/kermany/test/DME ~/workspace/datasets/kermany/test/DRUSEN ~/workspace/datasets/kermany/test/NORMAL --label CNV DME DRUSEN NORMAL
