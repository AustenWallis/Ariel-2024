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
# %load_ext cudf.pandas
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import optuna
import sys
import pyinstrument
import gc


# %%
############################################
print("Data Reduction to Science Images")
############################################
# Import the data
print(os.listdir())

# Metadata Files
train_adc_info = pd.read_csv('train_adc_info.csv') # ADC gain and offset and star column

test_adc_info = pd.read_csv('test_adc_info.csv')
adc_info = {'train': train_adc_info, 'test': test_adc_info}
#train_labels = pd.read_csv('train_labels.csv').values # planet spectra?
#don't include the first column
train_labels = pd.read_csv('train_labels.csv').iloc[:, 1:].values
train_labels_airs = train_labels[:, :-1] # all wavelengths for AIRS except the last one Å
train_labels_fgs1 = train_labels[:, -1] # 283rd wavelength for FGS1 Å
axis_info = pd.read_parquet('axis_info.parquet') # time step information for each image

wavelengths = pd.read_csv('wavelengths.csv').values
wavelengths = wavelengths.reshape(283,)
wavelengths_airs = wavelengths[:282]*1e4 # all wavelengths for AIRS except the last one Å
wavelength_fgs1 = wavelengths[282]*1e4 # 283rd wavelength for FGS1 Å

# Signal Files
path_to_train = 'train'
path_to_test = 'test'

train_planet_ids = os.listdir(path_to_train)
train_planet_ids.sort(key=int)
test_planet_ids = os.listdir(path_to_test)
test_planet_ids.sort(key=int)

def correct_gain_offset(signal, gain, offset):
    """To restore full dynamic range of the signal, the signal is divided by the
    gain and an offset is added.

    Args:
        signal (array): The uncorrected signal
        gain (float): The instrument gain for each planet
        offset (float): The instrument offset for each planet

    Returns:
        array: The corrected signal
    """
    return signal / gain + offset

def correct_calibration(signal, dark, flat, read, delta_T=None):
    """The signal is corrected to a science image by subtracting the dark current
    and read noise and dividing by the flat field - bias.
    Dark may need to be multiplied by the instrument exposure time.

    Args:
        signal (array): Image signal
        dark (array): Dark current
        flat (array): Flat field
        read (array): Biases

    Returns:
        array: science image
    """
    print(type(signal), type(dark), type(flat), type(read), type(delta_T))
    print(np.shape(signal), np.shape(dark), np.shape(flat), np.shape(read), np.shape(delta_T))
    if delta_T is None:
        processed = (signal - read - dark) / (flat - read)
    else:
        processed = (signal - read - (dark*delta_T[:, np.newaxis, np.newaxis])) / (flat - read)
    return processed

def correct_dead(signal, dead):
    """TODO

    Args:
        signal (_type_): _description_
        dead (_type_): _description_
    """
    return signal * dead

def correct_linear_corr(signal, linear_corr):
    """TODO

    Args:
        signal (_type_): _description_
        linear_corr (_type_): _description_
    """
    return signal * linear_corr

def correlate_double_sampling(signal):
    """Difference between the end and the start of a single exposure. Each 
    recorded sample is the start of exposure, next end of exposure, then start
    of next exposure, then end of next exposure. The difference between the
    start and end of the exposure is the correlated double sampling.

    Args:
        signal (array): science image
    """
    signal = np.diff(signal, axis=1)[:,::2, :] # selecting 1-0, 3-2, 5-4 .. so on
    return signal

def spectra_to_1D(signal):
    """Sum the spectra along the y-axis to get a 1D spectrum.

    Args:
        signal (array): 2D image

    Returns:
        array: 1D spectrum
    """
    return np.sum(signal, axis=1)

def plot_signal(signal):
    """Plot the signal

    Args:
        signal (array): 2D image
    """
    plt.plot(signal)
    plt.show()

def reduce_data(planet_index, planet_id, path_to_data, adc_info, instrument='AIRS-CH0'):
    """FGS1 primarly should be focused on photometery,
    AIRS-CH0 should be focused on spectrometry.
    Data pipeline to impliment reducing the data to science images.

    Args:
        planet_index (_type_): _description_
        planet_id (_type_): _description_
        path_to_data (_type_): _description_
        adc_info (_type_): _description_
        instrument (str, optional): _description_. Defaults to 'AIRS-CH0'.

    Returns:
        _type_: _description_
    """
    
    # profiler = pyinstrument.Profiler()
    # profiler.start()
    print(f'Reducing data for planet {planet_id}, index {planet_index}')
    gc.collect()
    
    # User Switches
    gain = True
    calibration = True
    dead = False
    linear_corr = False
    corr_double_sampling = False
    spectra_sum = True
    
    planet_dict = {}
    clean_planet = {}
    
    if instrument == 'AIRS-CH0':
        signal = pd.read_parquet(
            os.path.join(path_to_data, planet_id, 'AIRS-CH0_signal.parquet')).values.reshape(11250, 32, 356)
        
        if calibration:
            # dark np.shape(32,356)
            dark = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'dark.parquet')).values
            # flat np.shape(32, 356)
            flat = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'flat.parquet')).values
            # read np.shape(32, 356)
            read = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'read.parquet')).values
            delta_t = axis_info['AIRS-CH0-integration_time'].dropna().values
            
            signal = correct_calibration(signal, dark, flat, read, delta_t)
        
        plot_signal(signal[0][15])
        
        if gain:
            signal = correct_gain_offset(signal,
                                         adc_info['AIRS-CH0_adc_gain'][planet_index],
                                         adc_info['AIRS-CH0_adc_offset'][planet_index]
                                         )
        plot_signal(signal[0][15])
        
        if dead:
            dead = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'dead.parquet')).values
            signal = correct_dead(signal, dead)
        
        if linear_corr:
            linear_corr = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'linear_corr.parquet')).values
            signal = correct_linear_corr(signal, linear_corr)

        if corr_double_sampling:
            signal = correlate_double_sampling(signal)
        plot_signal(signal[0][15])
        # cut wavelengths down to 282 (clean wavelengths)
        signal = signal[:, :, 39:321]
        
        if spectra_sum:
            signal = spectra_to_1D(signal)
        plot_signal(signal[0][15])
    elif instrument == 'FGS1':
        signal = pd.read_parquet(
            os.path.join(path_to_data, planet_id, 'FGS1_signal.parquet')).values.reshape(135000, 32, 32)
        
        if gain:
            signal = correct_gain_offset(signal,
                                         adc_info['FGS1_adc_gain'][planet_index],
                                         adc_info['FGS1_adc_offset'][planet_index]
                                         )
        
        if calibration:
            # dark np.shape(32,356)
            dark = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'FGS1_calibration', 'dark.parquet')).values
            # flat np.shape(32, 356)
            flat = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'FGS1_calibration', 'flat.parquet')).values
            # read np.shape(32, 356)
            read = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'FGS1_calibration', 'read.parquet')).values
            
            signal = correct_calibration(signal, dark, flat, read)
            
        if dead:
            dead = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'FGS1_calibration', 'dead.parquet')).values
            signal = correct_dead(signal, dead)
        
        if linear_corr:
            linear_corr = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'FGS1_calibration', 'linear_corr.parquet')).values
            signal = correct_linear_corr(signal, linear_corr)

        if corr_double_sampling:
            signal = correlate_double_sampling(signal)

    print(f'Planet {planet_id} reduced')
    # profiler.stop()
    # profiler.print()
    return signal

# import shelve
# with shelve.open('clean_planet_data') as clean_planet_data:
#     for planet_id in train_planet_ids:
#         clean_planet_data[planet_id] = reduce_data(planet_id, path_to_train, adc_info['train'])


# for planet_id in train_planet_ids:
#     #train_planet_data[planet_id] = reduce_data(planet_id, path_to_train, adc_info['train'])
#     reduce_data(planet_id, path_to_train, adc_info['train'])

train_planet_data = {}
test_planet_data_airs = {}
test_planet_data_fgs1 = {}
train_planet_data_airs = {}
train_planet_data_fgs1 = {}

for planet_index, planet_id in enumerate(test_planet_ids):
    test_planet_data_airs[planet_id] = reduce_data(planet_index, planet_id, path_to_test, adc_info['test'], instrument='AIRS-CH0')
    test_planet_data_fgs1[planet_id] = reduce_data(planet_index, planet_id, path_to_test, adc_info['test'], instrument='FGS1')
    
for planet_index, planet_id in enumerate(train_planet_ids[0:1]):
    train_planet_data_airs[planet_id] = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='AIRS-CH0')
    train_planet_data_fgs1[planet_id] = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='FGS1')
    

print('Information Loaded')
# %%
############################################
print("Example plot")
############################################
plt.figure(figsize=(20, 20))
plt.imshow(test_planet_data_airs['499191466'])
# change x values to wavelengths_airs values
plt.xticks(ticks=np.arange(0, 282, 20), labels=wavelengths_airs[::20], rotation=45)
plt.xlabel('Wavelength')
plt.title('2D Spectra where the colour represents flux at a particular wavelength')
# add colour bar
plt.colorbar()

plt.show()
# plt.scatter(wavelengths_airs, test_planet_data_airs['499191466'][6700])
# plt.xlabel('Wavelength')
# plt.ylabel('Flux')
# plt.title('1D Spectra')
# plt.show()


plt.figure(figsize=(20, 20))
plt.imshow(test_planet_data_fgs1['499191466'][0])
plt.title('Image with colour representing flux of a particular wavelength')
plt.colorbar()
plt.show()

plt.figure(figsize=(20, 20))
plt.scatter(wavelengths_airs, train_planet_data_airs['785834'][0])
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('1D Spectra')
plt.show()

plt.scatter(wavelengths_airs, train_labels[0][:-1])
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('1D Spectra')
plt.show()

# TODO LIST
# 1. Add dead pixels
# 2. Add linear correction
# 3. SUM spectra along the y axis


# %%
############################################
print("Reducing the data")
############################################

print('axis_info', type(axis_info), np.shape(axis_info))
print('wavelengths_airs', type(wavelengths_airs), np.shape(wavelengths_airs))
print('wavelength_fgs1', type(wavelength_fgs1), np.shape(wavelength_fgs1))
print('sample_submission', type(sample_submission), np.shape(sample_submission))
print('train_adc_info', type(train_adc_info), np.shape(train_adc_info))
print('test_adc_info', type(test_adc_info), np.shape(test_adc_info))
print('train_labels', type(train_labels), np.shape(train_labels))

# print the headers
print('axis_info', axis_info.head())
print('wavelengths_airs', wavelengths_airs)
print('wavelength_fgs1', wavelength_fgs1)
print('sample_submission', sample_submission.head())
print('train_adc_info', train_adc_info.head())
print('test_adc_info', test_adc_info.head())
print('train_labels', train_labels.head())



# %%
############################################
print("Submission File")
############################################
"""You must predict a mean and uncertainty for each planet_id. An example submission file is included in the Data Files. Each submission row must include 567 columns, so we will not attempt to provide an example here. The leftmost column must be the planet_id, the next 283 columns must be the spectra, and the remaining columns the uncertainties."""

# Sample Submission, formatting much be same
sample_submission = pd.read_csv('sample_submission.csv').iloc[:, 1:].values


submission = 0
submission.to_csv('submission.csv', index=False)








































# %%
#def reduce(planet_id, path_to_data, adc_info):
    # # Unflatten the data, multiple by gain and add offset
    # airs_signal = pd.read_parquet(
    #     os.path.join(path_to_data, planet_id, 'AIRS-CH0_signal.parquet')).values.reshape(11250, 32, 356)
    # planet_dict['AIRS-CH0_signal'] = airs_signal / adc_info['AIRS-CH0_adc_gain'][adc_planet_index] + adc_info['AIRS-CH0_adc_offset'][adc_planet_index]

    # # Unflatten the data, multiple by gain and add offset
    # fgs1_signal = pd.read_parquet(
    #     os.path.join(path_to_data, planet_id, 'FGS1_signal.parquet')).values.reshape(135000, 32, 32)
    # planet_dict['FGS1_signal'] = fgs1_signal / adc_info['FGS1_adc_gain'][adc_planet_index] + adc_info['FGS1_adc_offset'][adc_planet_index]

    # # temp_list = list(adc_info['planet_id'])
    # # adc_planet_index = temp_list.index(int(planet_id))
    # # Calibration
    # planet_dict['AIRS-CH0_calibration'] = {}
    # planet_dict['FGS1_calibration'] = {}

    # #load the calibration data
    # for instrument in ['AIRS-CH0', 'FGS1']:
    #     planet_dict[f'{instrument}_calibration']['dark'] = pd.read_parquet(
    #         os.path.join(path_to_data, planet_id, f'{instrument}_calibration', 'dark.parquet'))
    #     planet_dict[f'{instrument}_calibration']['flat'] = pd.read_parquet(
    #         os.path.join(path_to_data, planet_id, f'{instrument}_calibration', 'flat.parquet'))
    #     planet_dict[f'{instrument}_calibration']['read'] = pd.read_parquet(
    #         os.path.join(path_to_data, planet_id, f'{instrument}_calibration', 'read.parquet'))
    #     planet_dict[f'{instrument}_calibration']['dead'] = pd.read_parquet(
    #         os.path.join(path_to_data, planet_id, f'{instrument}_calibration', 'dead.parquet'))
    #     planet_dict[f'{instrument}_calibration']['linear_corr'] = pd.read_parquet(
    #         os.path.join(path_to_data, planet_id, f'{instrument}_calibration', 'linear_corr.parquet'))

    # # for each 11250 time steps, 32x356 pixels, we have 32x356 dark, flat, read values
    # # math = signal - read - (TODO dark *delta_T) / flat - read
    
    # clean_planet['AIRS-CH0'] = (planet_dict['AIRS-CH0_signal'] - planet_dict['AIRS-CH0_calibration']['read'].values - planet_dict['AIRS-CH0_calibration']['dark'].values) / (planet_dict['AIRS-CH0_calibration']['flat'].values - planet_dict['AIRS-CH0_calibration']['read'].values)
    # # cut 356 down to [39,321]
    # clean_planet['AIRS-CH0'] = clean_planet['AIRS-CH0'][:, :, 39:321] # cut the 356 pixels down to 282 pixels for clean wavelengths
    # # cds
    # clean_planet['AIRS-CH0'] = np.diff(clean_planet['AIRS-CH0'], axis=1) # correlated double sampling

    
    # clean_planet['FGS1'] = (planet_dict['FGS1_signal'] - planet_dict['FGS1_calibration']['read'].values - planet_dict['FGS1_calibration']['dark'].values) / (planet_dict['FGS1_calibration']['flat'].values - planet_dict['FGS1_calibration']['read'].values)
    # clean_planet['FGS1'] = np.diff(clean_planet['FGS1'], axis=1) # correlated double sampling

    # clean_planet['star'] = adc_info['star'].values # the star simulate for a particular planet
    
    # # you can add dead pixels in here : 
    
    # # you can add linear correction in here :
    