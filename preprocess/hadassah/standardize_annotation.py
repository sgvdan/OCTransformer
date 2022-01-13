from zipfile import Path
import pandas

LAST_COLUMN = 19


def standardize_annotation(annotations):
    assert annotations.iloc[0, LAST_COLUMN] == 'LAST VISIT'

    annotations = annotations.rename(columns={'P.I.D': 'DR_CODE'})
    for row_idx in range(1, len(annotations)):
        start = annotations.iloc[row_idx, :LAST_COLUMN]
        end = annotations.iloc[row_idx, LAST_COLUMN:]

        patient_code = start['DR_CODE']
        eye = start['EYE']

        annotations.loc[row_idx+.5, :LAST_COLUMN] = [patient_code, eye, *[value for value in end]]

    # Add important specifiers
    annotations.insert(1, 'P.I.D', None)
    annotations.insert(2, 'S.I.D', None)
    annotations.insert(3, 'E.I.D', None)

    annotations = annotations.iloc[:, :LAST_COLUMN]
    
    return annotations.sort_index().reset_index(drop=True)


if __name__ == '__main__':
    annotations_path = '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/original/annotations.xlsx'
    dest_path = '/home/projects/ronen/sgvdan/workspace/datasets/hadassah/original/std_annotations.xlsx'

    annotations = pandas.read_excel(annotations_path)
    standardize_annotation(annotations).to_excel(dest_path)
