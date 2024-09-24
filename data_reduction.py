# %%
"""

AIM: to extract the atmospheric spectra from each observation, with an estimate of its level of uncertainty.
EVALUATION: Gaussian Log-likelihood (GLL) function

Files:
axis_info.parquet - constant time step information for each image
AIRS-CH0_signal.parquet - signal data for each image
[train/test]_adc_info.csv - instrument gain, electrons count to instrumental fluxes.

# Data reduction pipeline

AIRS (instrument):
- Unflatten signal data( np.reshape(11250,32,356))
- Multiple signal data by gain from train/test_adc_info.csv
- Add offest from same file

FSG1 (instrument):
- Unflattened signal data (np.reshape(135000, 32, 32)  (with constant 0.1s timesteps)
- Multiple signal data by gain from train/test_adc_info.csv
- Add offset from same file

RAW SPECTRUM GENERATED! NEXT
Instrumental fluxes to real fluxes


Calibration: 
$image = \\frac{exposure - bias - dark}{flat - bias}$

•⁠  ⁠exposure: 
•⁠  ⁠dark: [train/test]/[planet_id]/[AIRS-CHO/FSG1]_calibration/dark.parquet
•⁠  ⁠flat: [train/test]/[planet_id]/[AIRS-CHO/FSG1]_calibration/flat.parquet
•⁠  ⁠bias: [train/test]/[planet_id]/[AIRS-CHO/FSG1]_calibration/read.parquet

- dead: [train/test]/[planet_id]/[AIRS-CHO/FSG1]_calibration/dead.parquet - mask, continumm fit to smooth out the data
- linear_corr: [train/test]/[planet_id]/[AIRS-CHO/FSG1]_calibration/linear_corr.parquet - mask, linear correction to the data


"""
# %%
############################################
print("Importing libraries")
############################################
import numpy as np
import pandas as pd
import tqdm as tqdm
import os
import matplotlib.pyplot as plt
import optuna
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../../../../Volumes/Siroccogrid/ariel-data-challenge-2024')))

# %%

# Import the data from USB
path_to_data = '../../../../../../../../../Volumes/Siroccogrid/ariel-data-challenge-2024'
print(os.listdir(path_to_data))

# Metadata Files
train_adc_info = pd.read_csv(os.path.join(path_to_data, 'train_adc_info.csv')) # ADC gain and offset and star column
test_adc_info = pd.read_csv(os.path.join(path_to_data, 'test_adc_info.csv'))
train_labels = pd.read_csv(os.path.join(path_to_data, 'train_labels.csv'))
axis_info = pd.read_parquet(os.path.join(path_to_data, 'axis_info.parquet')) # time step information for each image
wavelengths = pd.read_csv(os.path.join(path_to_data, 'wavelengths.csv'))

# Signal Files
path_to_train = os.path.join(path_to_data, 'train')
path_to_test = os.path.join(path_to_data, 'test')
train_planet_ids = os.listdir(path_to_train)
test_planet_ids = os.listdir(path_to_test)
train_planet_data = {}
test_planet_data = {}
for planet_id in train_planet_ids:
    train_planet_data[planet_id] = {}
    train_planet_data[planet_id]['AIRS-CH0_signal'] = pd.read_parquet(os.path.join(path_to_train, planet_id, 'AIRS-CH0_signal.parquet'))
    train_planet_data[planet_id]['FGS1_signal'] = pd.read_parquet(os.path.join(path_to_train, planet_id, 'FGS1_signal.parquet'))
    train_planet_data[planet_id]['AIRS-CH0_calibration'] = {}
    train_planet_data[planet_id]['FGS1_calibration'] = {}
    for instrument in ['AIRS-CH0', 'FGS1']:
        train_planet_data[planet_id][f'{instrument}_calibration']['dark'] = pd.read_parquet(os.path.join(path_to_train, planet_id, f'{instrument}_calibration', 'dark.parquet'))
        train_planet_data[planet_id][f'{instrument}_calibration']['flat'] = pd.read_parquet(os.path.join(path_to_train, planet_id, f'{instrument}_calibration', 'flat.parquet'))
        train_planet_data[planet_id][f'{instrument}_calibration']['read'] = pd.read_parquet(os.path.join(path_to_train, planet_id, f'{instrument}_calibration', 'read.parquet'))
        train_planet_data[planet_id][f'{instrument}_calibration']['dead'] = pd.read_parquet(os.path.join(path_to_train, planet_id, f'{instrument}_calibration', 'dead.parquet'))
        train_planet_data[planet_id][f'{instrument}_calibration']['linear_corr'] = pd.read_parquet(os.path.join(path_to_train, planet_id, f'{instrument}_calibration', 'linear_corr.parquet'))
for planet_id in test_planet_ids:
    test_planet_data[planet_id] = {}
    test_planet_data[planet_id]['AIRS-CH0_signal'] = pd.read_parquet(os.path.join(path_to_test, planet_id, 'AIRS-CH0_signal.parquet'))
    test_planet_data[planet_id]['FGS1_signal'] = pd.read_parquet(os.path.join(path_to_test, planet_id, 'FGS1_signal.parquet'))
    test_planet_data[planet_id]['AIRS-CH0_calibration'] = {}
    test_planet_data[planet_id]['FGS1_calibration'] = {}
    for instrument in ['AIRS-CH0', 'FGS1']:
        test_planet_data[planet_id][f'{instrument}_calibration']['dark'] = pd.read_parquet(os.path.join(path_to_test, planet_id, f'{instrument}_calibration', 'dark.parquet'))
        test_planet_data[planet_id][f'{instrument}_calibration']['flat'] = pd.read_parquet(os.path.join(path_to_test, planet_id, f'{instrument}_calibration', 'flat.parquet'))
        test_planet_data[planet_id][f'{instrument}_calibration']['read'] = pd.read_parquet(os.path.join(path_to_test, planet_id, f'{instrument}_calibration', 'read.parquet'))
        test_planet_data[planet_id][f'{instrument}_calibration']['dead'] = pd.read_parquet(os.path.join(path_to_test, planet_id, f'{instrument}_calibration', 'dead.parquet'))
        test_planet_data[planet_id][f'{instrument}_calibration']['linear_corr'] = pd.read_parquet(os.path.join(path_to_test, planet_id, f'{instrument}_calibration', 'linear_corr.parquet'))

# Sample Submission, formatting much be same
sample_submission = pd.read_csv(os.path.join(path_to_data, 'sample_submission.csv'))
print('Information Loaded')

# %%
############################################
print("Reducing the data")
############################################

print('axis_info', type(axis_info), np.shape(axis_info))
print('wavelengths', type(wavelengths), np.shape(wavelengths))
print('sample_submission', type(sample_submission), np.shape(sample_submission))
print('train_adc_info', type(train_adc_info), np.shape(train_adc_info))
print('test_adc_info', type(test_adc_info), np.shape(test_adc_info))
print('train_labels', type(train_labels), np.shape(train_labels))

# print the headers
print('axis_info', axis_info.head())
print('wavelengths', wavelengths.head())
print('sample_submission', sample_submission.head())
print('train_adc_info', train_adc_info.head())
print('test_adc_info', test_adc_info.head())
print('train_labels', train_labels.head())



# %%
############################################
print("Submission File")
############################################
"""You must predict a mean and uncertainty for each planet_id. An example submission file is included in the Data Files. Each submission row must include 567 columns, so we will not attempt to provide an example here. The leftmost column must be the planet_id, the next 283 columns must be the spectra, and the remaining columns the uncertainties."""
submission = 0
submission.to_csv('submission.csv', index=False)