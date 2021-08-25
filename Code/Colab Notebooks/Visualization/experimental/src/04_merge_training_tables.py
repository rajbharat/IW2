"""
Description: A script to generate dataframes containing all the
necessary information to complete path extraction and for later 
generation of training and test datasets.
"""
#%% DEPENDENCIES
import pandas as pd
import pickle
from pathlib import Path

#%% SETTING ROOT PATH TO DATA FOLDER
path_data = Path('/content/gdrive/My Drive/ProstrateX2-Challenge/ProstateX_temp/ProstateX-master/experimental/src')
path_data2 = Path('/content/gdrive/My Drive/ProstrateX2-Challenge/ProstateX_temp/ProstateX-master/src/data/train')
path_string = '/content/gdrive/My Drive/ProstrateX2-Challenge/ProstateX_temp/ProstateX-master/experimental/src'

#%% GENERATING FILE STRUCTURE TO STORE TABLES
tables_path = Path('/content/gdrive/My Drive/ProstrateX2-Challenge/ProstateX_temp/ProstateX-master/experimental/src/generated/tables/')
tables_path.mkdir(exist_ok = True)

#%% MERGING TRAINING TABLES
def generate_df_for_sequence(sequence_type, successful_conv):
    """
    This function generates a data frame for all patients in the data set. Each
    row contains a string that is analogous to the DCMSerDescr label in the
    provided train files. This string is generated from the original filename.
    The second column contains a path object for the re-sampled nifti file. This
    table can be joined to the other training files to create one large table
    with the appropriate sampling information.
    """

    nifti = path_data2 / 'generated/nifti'
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
                            nifti_resampled = path_data.joinpath('generated/nifti_resampled')
                            sequence_types = [x for x in nifti_resampled.iterdir() if x.is_dir()]
                            for sequence in sequence_types:
                                # check if directory name contains sequence type
                                if sequence_type in str(sequence):
                                    # then get all files in subdirectory
                                    files = sequence.rglob('*.*')          
                                    for file in files:
                                        # then check if filename contains patient_id
                                        if patient.parts[-1] in str(file): 
                                            path_to_resampled = file
                                            
                            return path_to_resampled
                        
                        DCMSerDescr_fn = generate_DCMSerDescr_from_filename()
                        path_to_resampled = get_path_to_resampled(sequence_type)
                        
                        key = patient.parts[-1] # patient_ID
                        value = [DCMSerDescr_fn, path_to_resampled]
                        patient_data[key] = value 
    
    data_frame = pd.DataFrame.from_dict(patient_data, orient = 'index')
    data_frame = data_frame.reset_index()
    data_frame.columns = ['ProxID','DCMSerDescr', 'path_to_resampled_file'] # renaming columns
    return data_frame

def join_dataframes (sequence_df, images_train_df, findings_train_df):
    """
    This function accepts a sequence dataframe (containing the the patient id
    and path to resampled data for that particular sequence) along with the
    images and findings train datasets from the ProstateX project. It returns a
    dataframe that combines information from each of these data sources to
    provide information about each sample in the dataset for later processing. 
    """

    sequence_df.loc[:,'DCMSerDescr'] = sequence_df.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
    
    if 'DCMSerDescr' in list(images_train_df.columns.values):
        # Subset to desired columns only and lowercase
        images_train_df.loc[:,'DCMSerDescr'] = images_train_df.loc[:,'DCMSerDescr'].apply(lambda x: x.lower())
        images_train_df = images_train_df[['ProxID', 'DCMSerDescr', 'fid', 'pos', 'WorldMatrix', 'ijk']]
        
        # the join was originally 'left' --> changed to 'inner' to retain values from both dataframes only because NaN gen in pos column
        first_merge = pd.merge(sequence_df, images_train_df, how = 'inner', on = ['ProxID', 'DCMSerDescr'])    
        final_merge = pd.merge(first_merge, findings_train_df, how = 'inner', on = ['ProxID', 'fid','pos'])
    else:
        first_merge = pd.merge(sequence_df, images_train_df, how = 'inner', on = ['ProxID'])
        final_merge = pd.merge(first_merge, findings_train_df, how = 'inner', on = ['ProxID', 'fid', 'pos'])

    return final_merge

def repair_entries(dataframe):
    """
    This function accepts a dataframe and reformats entries in select columns to
    make them more ammenable to use in the next phase of the analysis. 
    """

    def convert_to_tuple(dataframe, column):
        """
        A function to convert row values (represented as string of floats
        delimited by spaces) to a tuple of floats. Accepts the original data
        frame and a string for the specified column that needs to be converted.
        """  
        pd_series_containing_lists_of_strings = dataframe[column].str.split() 
        list_for_new_series = []
        for list_of_strings in pd_series_containing_lists_of_strings:
            container_list = []
            for item in list_of_strings:
                if column == 'pos':
                    container_list.append(float(item))
                else:
                    container_list.append(int(item))
            list_for_new_series.append(tuple(container_list))
        
        return pd.Series(list_for_new_series)    

    # Call function to convert select columns
    dataframe = dataframe.assign(pos_tuple = convert_to_tuple(dataframe, 'pos'))
    dataframe = dataframe.assign(ijk_tuple = convert_to_tuple(dataframe, 'ijk'))
    
    # Drop old columns, rename new ones, and reorder...
    dataframe = dataframe.drop(columns = ['pos','ijk', 'WorldMatrix'])
    dataframe = dataframe.rename(columns = {'pos_tuple':'pos', 'ijk_tuple':'ijk'})
    column_titles = ['ProxID', 'DCMSerDescr', 'path_to_resampled_file', 'fid', 'pos', 'ijk', 'zone', 'ClinSig']
    dataframe = dataframe.reindex(columns = column_titles)
    return dataframe

def main():
    # Load the ProstateX datasets
    images_train = pd.read_csv(str(path_data) + '/ProstateX-Images-Train.csv')
    ktrans_train = pd.read_csv(str(path_data) + '/ProstateX-Images-KTrans-Train.csv')
    findings_train = pd.read_csv(str(path_data) + '/ProstateX-Findings-Train.csv')

    # Check for successful dicom conversions
    dicom2nifti_success = Path('/content/gdrive/My Drive/ProstrateX2-Challenge/ProstateX_temp/ProstateX-master/logs/dicom2nifti_successful.txt')
    successful_conv = dicom2nifti_success.read_text()
    successful_conv = successful_conv.split('\n')
    successful_conv = list(filter(None, successful_conv)) # For sanity - remove any empty string(s)

    # Generating dataframe of information for specified sequence
    t2_df = generate_df_for_sequence('t2', successful_conv)
    adc_df = generate_df_for_sequence('adc', successful_conv)
    bval_df = generate_df_for_sequence('bval', successful_conv)
    ktrans_df = generate_df_for_sequence('ktrans', successful_conv)

    # t2 findings
    t2_findings = join_dataframes(t2_df, images_train, findings_train)
    t2_repaired = repair_entries(t2_findings)
    t2_repaired.to_csv(str(tables_path) + '/t2_train.csv')
    t2_repaired.to_pickle(str(tables_path) + '/t2_train.pkl')

    # adc_findings
    adc_findings = join_dataframes(adc_df, images_train, findings_train)
    adc_repaired = repair_entries(adc_findings)
    adc_repaired.to_csv(str(tables_path) + '/adc_train.csv')
    adc_repaired.to_pickle(str(tables_path) + '/adc_train.pkl')
    
    # bval_findings = join_dataframes(bval_df, images_train, findings_train)
    bval_findings = join_dataframes(bval_df, images_train, findings_train)
    bval_repaired = repair_entries(bval_findings)
    bval_repaired.to_csv(str(tables_path) + '/bval_train.csv')
    bval_repaired.to_pickle(str(tables_path) + '/bval_train.pkl')

    # ktrans_findings = join_dataframes(ktrans_df, ktrans_train, findings_train)
    ktrans_findings = join_dataframes(ktrans_df, ktrans_train, findings_train)
    ktrans_repaired = repair_entries(ktrans_findings)
    ktrans_repaired.to_csv(str(tables_path) + '/ktrans_train.csv')
    ktrans_repaired.to_pickle(str(tables_path) + '/ktrans_train.pkl')

main()