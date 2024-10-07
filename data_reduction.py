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
import h5py
from numba import njit

# %%
############################################
print("Data Reduction to Science Images")
############################################
# Import the data
# print(os.listdir())

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
    #print(type(signal), type(dark), type(flat), type(read), type(delta_T))
    #print(np.shape(signal), np.shape(dark), np.shape(flat), np.shape(read), np.shape(delta_T))
#     plt.imshow(signal[4550])
# # horizontal colourbar
#     plt.colorbar(orientation='horizontal')
#     plt.title('Signal')
#     plt.show()
#     plt.imshow(read)
#     plt.colorbar(orientation='horizontal')
#     plt.title('Bias')
#     plt.show()
    # plt.imshow(dark)
    # plt.colorbar(orientation='horizontal')
    # plt.title('Dark')
    # plt.show()


    
    if delta_T is None:
        # numerator = signal
        # numerator -= read
        # numerator -= dark
        # numerator /= flat
        # #denominator = flat #- read
        # processed = numerator
        processed = (signal - read - dark) / (flat )# - read)
    else:
        #print(np.max(delta_T), np.min(delta_T), 'delta_T')
        # numerator = signal 
        # dark *= delta_T[:, np.newaxis, np.newaxis]
        # numerator -= read
        # numerator -= dark
        # numerator /= flat
        processed = signal - read - (dark*delta_T[:, np.newaxis, np.newaxis]) /flat
        #- read - (dark*delta_T[:, np.newaxis, np.newaxis])
        # plt.imshow(dark*delta_T[:, np.newaxis, np.newaxis][4550])
        # plt.colorbar(orientation='horizontal')
        # plt.title('Dark*delta_T')
        # plt.show()
        
        # plt.imshow(numerator[4550])
        # plt.colorbar(orientation='horizontal')
        # plt.title('Numerator')
        # plt.show()
        # denominator = flat#-read
        
        # plt.imshow(flat)
        # plt.colorbar(orientation='horizontal')
        # plt.title('Flat')
        # plt.show()
        # plt.imshow(read)
        # plt.colorbar(orientation='horizontal')
        # plt.title('Read')
        # plt.show()
        #print(np.max(numerator), np.min(numerator), 'numerator')  
        # plt.imshow(denominator)
        # plt.colorbar(orientation='horizontal')
        # plt.title('Denominator')
        # plt.show()
  
        #print(np.max(denominator), np.min(denominator), 'denominator')
        #processed = processed_signal
        # processed = (signal - read - (dark*delta_T[:, np.newaxis, np.newaxis])) / (flat - read)
        
        # plt.imshow(processed[4550])
        # plt.colorbar(orientation='horizontal')
        # plt.title('Processed')
        # plt.show()
        #print(np.max(processed), np.min(processed), 'processed')
        
    return processed

def correct_dead(signal, dead):
    """Convert the signal to a masked_array to mask out the dead pixels.
    
    Args:
        signal (_type_): _description_
        dead (_type_): _description_
    """
    #print(np.shape(signal), np.shape(dead), 'signal and dead')
    dead_expanded = np.tile(dead, (signal.shape[0], 1, 1))
    #print(np.shape(dead_expanded), 'dead expanded') 
    return np.ma.masked_array(signal, mask=dead_expanded)

def correct_linear_corr(signal, linear_corr):
    # Reshape linear_corr to (degree + 1, 32, 282)
    degree_plus_one = linear_corr.shape[0] // signal.shape[1]  # 192 // 32 = 6
    linear_corr = linear_corr.reshape((degree_plus_one, signal.shape[1], signal.shape[2]))  # (6, 32, 282)
    
    # Flip coefficients
    linear_corr = np.flip(linear_corr, axis=0)  # Shape: (6, 32, 282)

    # Reshape coefficients for vectorized np.polyval
    coeffs = linear_corr.reshape(degree_plus_one, -1)  # Shape: (6, 32*282)

    # Reshape signal for vectorized evaluation
    signal_reshaped = signal.reshape(signal.shape[0], -1)  # Shape: (11250, 32*282)

    # Evaluate polynomial for each pixel over all samples
    corrected_signal_flat = np.polyval(coeffs, signal_reshaped)  # Shape: (11250, 32*282)

    # Reshape back to original dimensions
    corrected_signal = corrected_signal_flat.reshape(signal.shape)
    return corrected_signal

def correlate_double_sampling(signal):
    """Difference between the end and the start of a single exposure. Each 
    recorded sample is the start of exposure, next end of exposure, then start
    of next exposure, then end of next exposure. The difference between the
    start and end of the exposure is the correlated double sampling.

    Args:
        signal (array): science image
    """
    #print(np.shape(signal), 'signal before ')
    # signal = signal[:,1::2,:,:] - signal[:,::2,:,:]
    signal = np.diff(signal, axis=1)[:,::2, :] # selecting 1-0, 3-2, 5-4 .. so on
    #print(np.shape(signal), 'signal after')
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
    #print(f'Reducing data for planet {planet_id}, index {planet_index}')
    gc.collect()
    
    # User Switches
    gain = True
    calibration = True
    dead = True
    linear_corr = False
    corr_double_sampling = True
    spectra_sum = False
    
    planet_dict = {}
    clean_planet = {}
    
    if instrument == 'AIRS-CH0':
        signal = pd.read_parquet(
            os.path.join(path_to_data, planet_id, 'AIRS-CH0_signal.parquet')).values.reshape(11250, 32, 356)
        # cut wavelengths down to 282 (clean wavelengths)
        signal = signal[:, :, 39:321]
        #print(np.max(signal), np.min(signal), 'signal') 
        
        if calibration:
            # dark np.shape(32,356)
            dark = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'dark.parquet')).to_numpy(dtype=np.float32)
            dark = dark[:, 39:321]
            #print(np.max(dark), np.min(dark), 'dark')
            # flat np.shape(32, 356)
            flat = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'flat.parquet')).to_numpy(dtype=np.float32)
            flat = flat[:, 39:321]
            #print(np.max(flat), np.min(flat), 'flat')
            # read np.shape(32, 356)
            read = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'read.parquet')).to_numpy(dtype=np.float32)
            read = read[:, 39:321]
            #print(np.max(read), np.min(read), 'read')
            delta_t = axis_info['AIRS-CH0-integration_time'].dropna().to_numpy(dtype=np.float32)
            
            
            
            signal = correct_calibration(signal, dark, flat, read, delta_t)
        
        #plot_signal(signal[0][15])
        
        if gain:
            #print(adc_info['AIRS-CH0_adc_gain'][planet_index], adc_info['AIRS-CH0_adc_offset'][planet_index], 'gain and offset')
            signal = correct_gain_offset(signal,
                                         adc_info['AIRS-CH0_adc_gain'][planet_index],
                                         adc_info['AIRS-CH0_adc_offset'][planet_index]
                                         )
        #plot_signal(signal[0][15])
        
        if dead:
            dead = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'dead.parquet')).values
            dead = dead[:, 39:321]
            signal = correct_dead(signal, dead)
        
        if linear_corr:
            linear_corr = pd.read_parquet(
                os.path.join(path_to_data, planet_id, 'AIRS-CH0_calibration', 'linear_corr.parquet')).values
            linear_corr = linear_corr[:, 39:321]
            signal = correct_linear_corr(signal, linear_corr)

        if corr_double_sampling:
            signal = correlate_double_sampling(signal)
        #plot_signal(signal[0][15])
        
        # cut wavelengths down to 282 (clean wavelengths)
        #signal = signal[:, :, 39:321]
        
        if spectra_sum:
            signal = spectra_to_1D(signal)
        #plot_signal(signal[0][15])
        
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
    
# for planet_index, planet_id in enumerate(train_planet_ids[:5]):
#     train_planet_data_airs[planet_id] = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='AIRS-CH0')
#     train_planet_data_fgs1[planet_id] = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='FGS1')
    

        

# %%
save_dir_airs = os.path.join('processed_data', 'AIRS-CH0')
save_dir_fgs1 = os.path.join('processed_data', 'FGS1')
os.makedirs(save_dir_airs, exist_ok=True)
os.makedirs(save_dir_fgs1, exist_ok=True)

for planet_index, planet_id in tqdm(enumerate(train_planet_ids[:10])):
    train_planet_data_airs = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='AIRS-CH0')
    save_path_airs = os.path.join(save_dir_airs, f'{planet_id}.h5')
    with h5py.File(save_path_airs, 'w') as f:
        f.create_dataset('data', data=train_planet_data_airs)
    del train_planet_data_airs
    
    train_planet_data_fgs1 = reduce_data(planet_index, planet_id, path_to_train, adc_info['train'], instrument='FGS1')
    save_path_fgs1 = os.path.join(save_dir_fgs1, f'{planet_id}.h5')
    with h5py.File(save_path_fgs1, 'w') as f:
        f.create_dataset('data', data=train_planet_data_fgs1)
    del train_planet_data_fgs1
    gc.collect()
    

print('Information Loaded')

############################################
print('Methodology')
############################################
"""I am going to train an autoencoder to encode and decode true spectral data.
We will then take the decoder part of the autoencoder. Perform ICA on the 
observed spectra where the Components match the number of neurons in the 
code layer of the autoencoder. Have a deep neural network between the ICA 
components and the code layer of the autoencoder. The deep neural network
will learn the relationship between the ICA components and the code layer."""

# %%
############################################
print('Independent Components Analysis')
############################################

# ICA is a dimensionality reduction technique that is used to separate independent sources from a mixed signal.

from sklearn.decomposition import FastICA
# Truth planet spectra
#train_labels_airs
#train_labels_fgs1 

############################################
print('Ensamble Autoencoder training')
############################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import math
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): If True, prints messages when validation loss improves.
            delta (float): Minimum change in monitored quantity to qualify as improvement.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Number of epochs with no improvement
        self.best_score = None  # Best score seen so far
        self.early_stop = False  # Flag to indicate early stopping
        self.val_loss_min = np.Inf  # Minimum validation loss
        self.delta = delta  # Minimum change to count as improvement
        self.path = path  # Path to save the checkpoint

    def __call__(self, val_loss, model):
        score = -val_loss  # We use negative val_loss because we want to maximize the score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.path)  # Save the model
        self.val_loss_min = val_loss
        
class Autoencoder(nn.Module):
    def __init__(self, input_dim, code_dim, layer_num):
        super(Autoencoder, self).__init__()
    #     self.encoder = nn.Sequential(
    #         nn.Linear(input_dim, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, code_dim),
    #         nn.ReLU()
    #     )
    #     self.decoder = nn.Sequential(
    #         nn.Linear(code_dim, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, input_dim),
    #         nn.ReLU()
    #     )
        
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder(x)
    #     return x

            # Compute the sizes of the layers for the encoder
        encoder_layer_sizes = self.compute_layer_sizes(input_dim, code_dim, layer_num)
        
        # Build the encoder
        encoder_layers = []
        for i in range(len(encoder_layer_sizes) - 1):
            in_features = encoder_layer_sizes[i]
            out_features = encoder_layer_sizes[i + 1]
            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.LeakyReLU())
        # Remove the last ReLU activation
        encoder_layers = encoder_layers[:-1]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build the decoder (reverse of encoder)
        decoder_layer_sizes = encoder_layer_sizes[::-1]
        decoder_layers = []
        for i in range(len(decoder_layer_sizes) - 1):
            in_features = decoder_layer_sizes[i]
            out_features = decoder_layer_sizes[i + 1]
            decoder_layers.append(nn.Linear(in_features, out_features))
            decoder_layers.append(nn.LeakyReLU())
        # Remove the last ReLU activation
        decoder_layers = decoder_layers[:-1]
        self.decoder = nn.Sequential(*decoder_layers)
        
    def compute_layer_sizes(self, input_dim, code_dim, layer_num):
        """
        Compute layer sizes that decrease linearly in log space from input_dim to code_dim.
        """
        # Calculate logarithms of input_dim and code_dim
        log_input_dim = math.log(input_dim)
        log_code_dim = math.log(code_dim)
        
        # Generate linearly spaced values in log space
        layer_sizes_log = [
            log_input_dim + (log_code_dim - log_input_dim) * i / layer_num
            for i in range(layer_num + 1)
        ]
        
        # Exponentiate to get the actual sizes and round to nearest integer
        layer_sizes = [int(round(math.exp(s))) for s in layer_sizes_log]
        
        # Ensure sizes are unique and sorted
        layer_sizes = sorted(list(set(layer_sizes)), reverse=True)
        
        return layer_sizes
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
data = train_labels_airs
train_scaler = StandardScaler()

X_nn, X_test = train_test_split(data, test_size=0.2, random_state=63436)
X_train, X_val = train_test_split(X_nn, test_size=0.1, random_state=63436)
X_train = train_scaler.fit_transform(X_train)

X_val = train_scaler.transform(X_val)  # don't fit the validation data
X_test = train_scaler.transform(X_test) # don't fit the test data

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)



# TODO Add validation set and Early stopping
def objective(trial):
    # Suggest hyperparameters
    code_dim = trial.suggest_int('code_dim', low=5, high=200, step=1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-8, 1e-1)
    number_of_layers = trial.suggest_int('number_of_layers', low=2, high=10, step=1)
    
    # Create DataLoaders with the suggested batch_size
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(X_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    
    # Initialize the model with the suggested code_dim
    model = Autoencoder(input_dim=X_train.shape[1], code_dim=code_dim, layer_num=number_of_layers)
    
    # Move model to device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = EarlyStopping(patience=20, verbose=False)
    
    # Training loop
    n_epochs = 400  # Reduce epochs during optimization to save time
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
    
        # Evaluation on val data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in validation_loader:
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                val_loss += loss.item()
        val_loss /= len(validation_loader)
        
        # Report intermediate results to Optuna
        trial.report(val_loss, epoch)
        
        # Handle pruning (optional)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss
    
    return val_loss  # Optuna will minimize this value

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)  # Adjust n_trials as needed

print("Best hyperparameters:")
print(study.best_params)
print(f"Best test loss: {study.best_value}")

# retrain with best model 
best_code_dim = study.best_params['code_dim']
best_batch_size = study.best_params['batch_size']
best_learning_rate = study.best_params['learning_rate']
best_number_of_layers = study.best_params['number_of_layers']

# Create DataLoaders with the best batch_size
train_loader = DataLoader(X_train, batch_size=best_batch_size, shuffle=True)
validation_loader = DataLoader(X_val, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(X_test, batch_size=best_batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model with the best code_dim
model = Autoencoder(input_dim=X_train.shape[1], code_dim=best_code_dim, layer_num=best_number_of_layers)
model.to(device)

# Define loss function and optimizer with the best learning_rate
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

early_stopping = EarlyStopping(patience=20, verbose=True, path='best_model.pth')
# Train the model with more epochs
n_epochs = 1000  # Increase epochs for final training
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_data in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data in validation_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            val_loss += loss.item()
    val_loss /= len(validation_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.5f}')
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Final Test Loss: {test_loss}')
    

# save the best model for use
torch.save(model.state_dict(), 'autoencoder.pth')

# %%

# reload the saved model from autoencoder.pth
model = Autoencoder(input_dim=X_train.shape[1], code_dim=best_code_dim, layer_num=best_number_of_layers)
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# %%
# use the autoencode to see how well it can reconstruct the data
from scipy.ndimage import gaussian_filter1d
for planet in train_labels_airs[:50]:
    #standardize the data
    planet_copy = planet.copy()
    planet = train_scaler.transform(planet.reshape(1, -1))
    planet = torch.tensor(planet, dtype=torch.float32)
    reconstructed = model(planet).detach().numpy()
    reconstructed = train_scaler.inverse_transform(reconstructed)
    reconstructed = reconstructed.reshape(282)
    # gaussian smoothing reconstruction
    reconstructed = gaussian_filter1d(reconstructed, 1)
    # plot 10ppm confidence interval on planet_copy
    
    max_planet = planet_copy + 1e-5
    min_planet = planet_copy - 1e-5
    plt.fill_between(np.arange(282), max_planet, min_planet, alpha=0.5)
    plt.plot(planet_copy, label='Original')
    plt.plot(reconstructed, label='Reconstructed')
    plt.legend()
    plt.show()


# %%
############################################
print("Example plot")
############################################
plt.figure(figsize=(20, 20))
plt.imshow(test_planet_data_airs['499191466'][4550])
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
plt.imshow(test_planet_data_fgs1['499191466'][47568])
plt.title('Image with colour representing flux of a particular wavelength')
plt.colorbar()
plt.show()

plt.figure(figsize=(20, 20))
plt.scatter(wavelengths_airs, test_planet_data_airs['499191466'][4623])
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
    