import os
import re
from pathlib import Path

dir = Path('/home/nih/nih-dannyh/data/oct/DR_TIFF')

count = 0
regex = r'\d*_\d*_(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)_.*_(?P<eye>[R|L])'
for patient in dir.glob('DR_*'):
    for sample in patient.glob('*'):
        if ('bscan_36.tiff' in os.listdir(sample)) and ('bscan_37.tiff' not in os.listdir(sample)):
            matches = re.match(regex, sample.name)
            if matches is None:
                continue

            matches = matches.groupdict()

            print('{pid},{eye},{day}/{month}/{year},{filename},{labels}'
                  .format(pid=patient.name, eye='OD' if matches['eye'] == 'R' else 'OS',
                          day=matches['day'], month=matches['month'], year=matches['year'],
                          filename=sample.name, labels=','.join(['0'] * 16)))
            count += 1

print('DONE - ', count)
