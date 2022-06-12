import argparse
import os
import re
import pandas

COLUMNS_COUNT = 20


def standardize_annotation(annotations):
    assert annotations.iloc[0, COLUMNS_COUNT + 1] == 'LAST VISIT'

    annotations = annotations.rename(columns={'P.I.D': 'DR_CODE'})
    for row_idx in range(1, 44):
        start = annotations.iloc[row_idx, :COLUMNS_COUNT]
        end = annotations.iloc[row_idx, COLUMNS_COUNT + 1: 2 * COLUMNS_COUNT - 1]

        patient_code = start['DR_CODE']
        eye = start['EYE']

        annotations.loc[row_idx+.5, :COLUMNS_COUNT] = [patient_code, eye, *[value for value in end]]

    annotations = annotations.iloc[:, :COLUMNS_COUNT]
    annotations.insert(0, 'GROUP', None)
    annotations.iloc[0, 0] = 'TRAIN/EVAL/TEST'

    return annotations.sort_index().reset_index(drop=True)


def resolve_filenames(annotations, root_path):
    for row_idx in range(1, len(annotations)):
        patient_code = annotations.iloc[row_idx]['DR_CODE']
        day = '{:02d}'.format(annotations.iloc[row_idx]['DATE'].day)
        month = '{:02d}'.format(annotations.iloc[row_idx]['DATE'].month)
        year = '{:04d}'.format(annotations.iloc[row_idx]['DATE'].year)
        eye = 'L' if annotations.iloc[row_idx]['EYE'] == 'OS' else 'R'

        regex = r'\d*_\d*_{year}-{month}-{day}_.*_{eye}'.format(year=year, month=month, day=day, eye=eye)
        r = re.compile(regex)
        path = os.path.join(root_path, patient_code)
        if not os.path.exists(path):
            print('Patient {patient_code} does not exist in the dataset'.format(patient_code=patient_code))
            continue
        matches = list(filter(r.match, os.listdir(path)))

        if len(matches) == 0:
            print('Match not found for patient:{patient_code}, date:{year}-{month}-{day}, and eye:{eye}'
                  .format(patient_code=patient_code, year=year, month=month, day=day, eye=eye))
        elif len(matches) == 1:
            temp = annotations.iloc[row_idx].copy()
            temp['FileName'] = matches[0]
            annotations.iloc[row_idx] = temp
        else:
            valid_matches = []
            for match in matches:
                match_path = os.path.join(path, match)
                # Valid volume should have at least 7 scans in it
                if all([bscan in os.listdir(match_path) for bscan in ['bscan_0.tiff', 'bscan_3.tiff', 'bscan_6.tiff']]):
                    valid_matches.append(match)

            if len(valid_matches) == 1:
                temp = annotations.iloc[row_idx].copy()
                temp['FileName'] = valid_matches[0]
                annotations.iloc[row_idx] = temp
            else:
                print('Several valid occurrences in patient:{patient_code}, '
                      'date:{year}-{month}-{day}, and eye:{eye}: {matches}'
                      .format(patient_code=patient_code, year=year, month=month, day=day, eye=eye, matches=matches))

    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardize Hadassah Annotations')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input annotations path (.xlsx)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output (standardized) annotations path (.xlsx)')

    args = parser.parse_args()

    annotations = pandas.read_excel(args.input)
    annotations = standardize_annotation(annotations)
    annotations = resolve_filenames(annotations, root_path='/home/nih/nih-dannyh/data/oct/DR_TIFF')
    annotations.to_excel(args.output)
