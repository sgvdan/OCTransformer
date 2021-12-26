python e2e_builder.py --name 37-slices/train --path ~/workspace/OCT-DL/Data/train/control ~/workspace/OCT-DL/Data/train/study --label HEALTHY SICK
python e2e_builder.py --name 37-slices/eval --path ~/workspace/OCT-DL/Data/validation/control ~/workspace/OCT-DL/Data/validation/study --label HEALTHY SICK
python e2e_builder.py --name 37-slices/test --path ~/workspace/OCT-DL/Data/test/control ~/workspace/OCT-DL/Data/test/study --label HEALTHY SICK
