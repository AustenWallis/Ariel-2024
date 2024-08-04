# %%
"""
Files:
axis_info.parquet - constant time step information for each image
ARIS-CH0_signal.parquet - signal data for each image
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
•⁠  ⁠dark: [train/test]/[planet_id]/[ARIS-CHO/FSG1]_calibration/dark.parquet
•⁠  ⁠flat: [train/test]/[planet_id]/[ARIS-CHO/FSG1]_calibration/flat.parquet
•⁠  ⁠bias: [train/test]/[planet_id]/[ARIS-CHO/FSG1]_calibration/read.parquet

- dead: [train/test]/[planet_id]/[ARIS-CHO/FSG1]_calibration/dead.parquet - mask, continumm fit to smooth out the data
- linear_corr: [train/test]/[planet_id]/[ARIS-CHO/FSG1]_calibration/linear_corr.parquet - mask, linear correction to the data


bahahababahababah
"""