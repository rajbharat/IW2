import pandas as pd
import numpy as np
import matplotlib.pyplot
from pathlib import Path

images_train = pd.read_csv('/content/drive/My Drive/ProstrateX2/Dataset/Training/ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv')
ktrans_train = pd.read_csv('/content/drive/My Drive/ProstrateX2/Dataset/Training/ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv')
findings_train = pd.read_csv('/content/drive/My Drive/ProstrateX2/Dataset/Training/ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv')

# The usual check for successful conversions
dicom2nifti_success = Path.cwd().parent  /'/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/logs/dicom2nifti_successful.txt'
successful_conv = dicom2nifti_success.read_text()
successful_conv = successful_conv.split('\n')
successful_conv = list(filter(None, successful_conv)) # For sanity - remove any empty string(s)
import os
successful_conv=[i for i in os.listdir("/content/drive/My Drive/ProstrateX2/Dataset/Training/KtransTrain")]

root_dir = Path.cwd().parent
nifti = root_dir.joinpath('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/nifti')


def generate_df_for_sequence(sequence_type):
    """
    This function generates a dataframe for all patients in the dataset. Each row contains
    a string that is analgous to the DCMSerDescr label in the other train files. This string
    is generated from the original filename. The second column contains a path object for the
    resampled nifti file. This table can be joined to the other training files to create one
    large table with the appropriate sampling information.
    """
    patient_data = {}
    patient_folders = [x for x in nifti.iterdir() if x.is_dir()]
    for patient in patient_folders:
        if patient.stem in successful_conv:
            sequences = [x for x in patient.iterdir() if x.is_dir()]
            for sequence in sequences:
                if sequence.parts[-1] == sequence_type:
                    for item in sequence.rglob('*.*'):

                        def generate_DCMSerDescr_from_filename():
                            # remove extension from path
                            full_name = item.parts[-1]
                            split = full_name.split('.')
                            name_without_extension = split[0]

                            # remove first num and underscore from path
                            first_underscore = name_without_extension.find('_') + 1
                            value = name_without_extension[first_underscore:]
                            return value

                        def get_path_to_resampled(sequence_type):
                            nifti_resampled = root_dir.joinpath(
                                '/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/nifti_resampled')
                            sequence_types = [x for x in nifti_resampled.iterdir() if x.is_dir()]

                            for sequence in sequence_types:
                                # check if directory name contains sequence type

                                # if sequence_type in str(sequence):

                                # then get all files in subdirectory
                                files = sequence.rglob('*.*')
                                for file in files:

                                    # then check if filename contains patient_id
                                    if patient.parts[-1] in str(file):
                                        path_to_resampled = file

                            return path_to_resampled

                        DCMSerDescr_fn = generate_DCMSerDescr_from_filename()
                        path_to_resampled = get_path_to_resampled(sequence_type)

                        key = patient.parts[-1]  # patient_ID
                        value = [DCMSerDescr_fn, path_to_resampled]
                        patient_data[key] = value

    data_frame = pd.DataFrame.from_dict(patient_data, orient='index')
    data_frame = data_frame.reset_index()
    data_frame.columns = ['ProxID', 'DCMSerDescr', 'path_to_resampled_file']  # renaming columns
    return data_frame

print('Images:', images_train.columns.values)
print('Ktrans:', ktrans_train.columns.values)
print('Findings:', findings_train.columns.values)


def join_dataframes(sequence_df, images_train_df, findings_train_df):
    sequence_df.loc[:, 'DCMSerDescr'] = sequence_df.loc[:, 'DCMSerDescr'].apply(lambda x: x.lower())

    if 'DCMSerDescr' in list(images_train_df.columns.values):
        # Subset to desired columns only and lowercase
        images_train_df.loc[:, 'DCMSerDescr'] = images_train_df.loc[:, 'DCMSerDescr'].apply(lambda x: x.lower())
        images_train_df = images_train_df[['ProxID', 'DCMSerDescr', 'fid', 'pos', 'WorldMatrix', 'ijk']]

        first_merge = pd.merge(sequence_df, images_train_df, how='left', on=['ProxID', 'DCMSerDescr'])
        final_merge = pd.merge(first_merge, findings_train_df, how='left', on=['ProxID', 'fid', 'pos'])
    else:
        first_merge = pd.merge(sequence_df, images_train_df, how='left', on=['ProxID'])
        final_merge = pd.merge(first_merge, findings_train_df, how='left', on=['ProxID', 'fid', 'pos'])

    return final_merge
t2_df = generate_df_for_sequence('t2')
adc_df = generate_df_for_sequence('adc')
bval_df = generate_df_for_sequence('bval')
ktrans_df = generate_df_for_sequence('ktrans')

t2_df_pe = join_dataframes(t2_df, images_train, findings_train)
adc_df_pe = join_dataframes(adc_df, images_train, findings_train)
bval_df_pe = join_dataframes(bval_df, images_train, findings_train)
bval_df_pe = join_dataframes(ktrans_df, ktrans_train, findings_train)

t2_df_pe.to_pickle("/content/drive/My Drive/ProstrateX2/Dataset/Training/tables/t2_train.pkl")
adc_df_pe.to_pickle("/content/drive/My Drive/ProstrateX2/Dataset/Training/tables/adc_train.pkl")
bval_df_pe.to_pickle("/content/drive/My Drive/ProstrateX2/Dataset/Training/tables/bval_train.pkl")
bval_df_pe.to_pickle("/content/drive/My Drive/ProstrateX2/Dataset/Training/tables/ktrans_train.pkl")
print("Done")