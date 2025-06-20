import numpy as np
import pandas as pd
import os
from io import StringIO


#data importing function
def load_data(filepath, key=None):
    # open the file and determine where the data starts using the #DATA string in the file
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('#DATA'):
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError("No #DATA section found in file.")

    # Replace comma with period for decimal separators
    data_lines = [line.replace(',', '.') for line in lines[data_start:]]

    # Read the data into a DataFrame
    data_str = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_str), delimiter='\t', engine='python', header=None)

    # Optionally, set column names if known, e.g., ['Wavenumber', 'Intensity']
    df.columns = ['Wavenumber', 'Intensity']

    # Use the filename (without extension) as the key
    if key is None:
        key = os.path.splitext(os.path.basename(filepath))[0]
    data_dict = {key: df}

    return data_dict

def find_troughs_simple(spectrum_dict, key, trough_dict, window=10):
    """
    For each expected trough in troughs_df, find the minimum intensity in spectrum_df
    within ±window of the expected wavenumber, and set it as the actual trough.
    Modifies troughs_df in place and also returns it.
    """
    spectrum_df = spectrum_dict[key]
    troughs_df = trough_dict[key]
    for i, expected in enumerate(troughs_df['Expected Trough']):
        # Define window
        mask = (spectrum_df['Wavenumber'] >= expected - window) & (spectrum_df['Wavenumber'] <= expected + window)
        window_df = spectrum_df[mask]
        if not window_df.empty:
            min_idx = window_df['Intensity'].idxmin()
            actual_trough = spectrum_df.loc[min_idx, 'Wavenumber']
            troughs_df.at[i, 'Actual Trough'] = actual_trough
            actual_intensity = spectrum_df.loc[min_idx, 'Intensity']
            troughs_df.at[i, 'Intensity'] = actual_intensity   
        else:
            troughs_df.at[i, 'Actual Trough'] = None  # or np.nan if you prefer
    return troughs_df

#Normalizing function
def normalize(data_dict):
    normalized_data_dict = {}
    
    #Normalize the data for each set
    for key, data in data_dict.items():
        #Get the maximum value of the data
        max_value = data['Intensity'].max()
        
        #Normalize the data
        normalized_data = data.copy()
        normalized_data['Normalized Intensity'] = normalized_data['Intensity'] / max_value
        
        #Add the normalized data to the dictionary
        normalized_data_dict[key] = normalized_data
                
        print(f"Data for {key} successfully normalized\n")
    return normalized_data_dict


def noise_slice(df, cutoff=0.005, intensity_col='Normalized Intensity'):
    # Reverse the DataFrame (so highest index is first)
    df_rev = df.iloc[::-1].reset_index(drop=True)
    # Find the first index where intensity is above the cutoff
    above_cutoff = df_rev[df_rev[intensity_col] > cutoff]
    if above_cutoff.empty:
        # No value above cutoff, return empty DataFrame
        return df
    first_idx = above_cutoff.index[0]
    # Slice the original DataFrame to keep only values after this point
    # (since we reversed, need to convert index back)
    keep_idx = len(df) - first_idx
    return df.iloc[:keep_idx]

def noise_cutoff_snr(df, intensity_col='NormDivWaterVapor', window=10, noise_std_thresh=0.001):
    """
    Slices the DataFrame from the point where signal falls into noise based on std dev in a sliding window.
    """
    # Reverse so we go from high to low wavenumbers
    df_rev = df.iloc[::-1].reset_index(drop=True)

    intensities = df_rev[intensity_col].values

    for i in range(len(intensities) - window):
        window_std = np.std(intensities[i:i+window])

        if window_std < noise_std_thresh:
            cutoff_idx = len(df) - i
            return df.iloc[:cutoff_idx]

    # If no cutoff found, return full DF
    return df

def match_experimental_peaks_to_reference(experimental_peaks, reference_df, tolerance=1.0):
    results = {}

    for sample_name, peaks in experimental_peaks.items():
        sample_matches = []
        for peak in peaks:
            # Filter reference table within tolerance range
            matches = reference_df[
                (reference_df["Energy_keV"] >= peak - tolerance) &
                (reference_df["Energy_keV"] <= peak + tolerance)
            ]
            if not matches.empty:
                for _, row in matches.iterrows():
                    sample_matches.append({
                        "Experimental_Peak": peak,
                        "Matched_Energy": row["Energy_keV"],
                        "Isotope": row["Nuclide"],
                        "Intensity": row["Intensity_percent"]
                    })
        results[sample_name] = sample_matches
    return results



#Simple peak finding function
# def find_troughs_simple(data, start_channel, channel_step, cutoff_value, threshold_factor=2):

#     ### Debugging
#     print(f"Data type: {type(data)}")
#     print(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")

#     peaks_dict = {}  # Dictionary to store peaks for each element

#     for key, data in data.items():
#         peaks = []  # List to store peaks for the current element
#         saved_indices = set()  # Set to keep track of saved indices

#         #Set the current start and end channels
#         current_start_channel = start_channel
#         current_end_channel = start_channel + channel_step

#         while current_end_channel <= len(data):
#             #Slice the data
#             #print(f"Trying slice: {current_start_channel} to {current_end_channel}")       #Debug line for loop checking

#             sliced_data = data.iloc[current_start_channel:current_end_channel]
#             min_peak_value = sliced_data['Counts per Second'].mean()

#             # Check if the slice is empty
#             if sliced_data.empty:
#                 print(f"Warning: Empty slice for {key} from {current_start_channel} to {current_end_channel}.")
#                 break
            
#             #Calculate the max and average
#             max_value = sliced_data['Counts per Second'].max()
#             avg_value = sliced_data['Counts per Second'].mean()
#             peak_index = sliced_data['Counts per Second'].idxmax()
            

#             #Compare the max value to the threshold
#             if max_value > threshold_factor * avg_value and max_value > max(min_peak_value, cutoff_value):               
#                 if peak_index not in saved_indices:
#                     # Save the peak if it hasn't been saved before
#                     peaks.append(data.iloc[peak_index])
#                     saved_indices.add(peak_index)  # Add the index to the set of saved indices
#                     print(f"Peak found for {key} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
#                     if peak_index - int(channel_step / 2) < 0:
#                         current_start_channel = 0
#                         current_end_channel += channel_step
#                     else:
#                         current_start_channel = peak_index - int(channel_step / 2)
#                         current_end_channel = min(current_start_channel + channel_step, len(data) + 1)
#                     print(f"Updated to: {current_start_channel} → {current_end_channel}\n")
#                 else:
#                     #Skip to the next window if the peak has already been saved
#                     print(f"Duplicate peak found for {key} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
#                     current_start_channel += channel_step 
#                     current_end_channel += channel_step
#                     print(f"Updated to: {current_start_channel} → {current_end_channel}\n")

#             else:
#                 print(f"No peak found for {key} in channel range {current_start_channel} to {current_end_channel} (Max: {max_value}, Avg: ", avg_value * threshold_factor, ")")       #debug for loop checking
#                 current_start_channel += channel_step
#                 current_end_channel += channel_step
#                 print(f"Updated to: {current_start_channel} → {current_end_channel}")          #debug for loop checking

#         if peaks:
#             peaks_dict[key] = pd.DataFrame(peaks)
#         else:
#             print(f"No peaks found for {key}.")
#             peaks_dict[key] = pd.DataFrame(columns=data.columns)  # Create an empty DataFrame with the same columns
        
#         # # Debugging: Print detected peaks (for loop checking)
#         print(f"Peaks for {key}:")
#         print(peaks_dict[key][['Channel', 'Counts per Second']])
#         print("-" * 50)
        
#     return peaks_dict