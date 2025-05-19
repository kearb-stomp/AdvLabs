#Utility functions created for Lab 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import json

print("Matplotlib imported as plt:", plt)

#Channel to Energy function
def channel_to_energy(channel, slope, intercept):
    return slope * channel + intercept

#To superscript function
def to_superscript(number):
    superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(number).translate(superscript_map)

#Linear function
def linear(x, m, b):
    return m * x + b

#Gaussian function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

#Truncating Function
def truncate(data_dict):
    truncated_data_dict = {}
    
    #Truncate the data for each set
    for key, data in data_dict.items():
        threshold = 20                          #in total counts
        
        #Truncate the data
        for i in range(len(data) -1, -1, -1):
            if data['Counts'].iloc[i] > threshold:
                #Get the number of extra windows to add
                extra_windows = 0.3 * i
                
                #Find the truncation
                truncation_point = round(min(int(i + extra_windows), len(data))/1000) * 1000

                #Truncate the data
                if truncation_point + extra_windows < len(data):                          
                    print(f"Truncating {key} at channel {truncation_point} (original length: {len(data)}) \n")               

                else:
                    print(f"Truncation point is greater than data length for {key} (length: {len(data)})")
                    truncation_point = len(data)
                break
   
        #Truncate the data to the truncation point
        truncated_data = data.iloc[:truncation_point].reset_index(drop=True)
        
        truncated_data['Display Element Name'] = data['Display Element Name'].iloc[0]
        #truncated_data['Element Name'] = data['Element Name'].iloc[0]
        

              
        #Add the truncated data to the dictionary
        truncated_data_dict[key] = truncated_data
        
    return truncated_data_dict

#Normalizing function
def normalize(data_dict):
    normalized_data_dict = {}
    
    #Normalize the data for each set
    for key, data in data_dict.items():
        #Get the maximum value of the data
        max_value = data['Counts per Second'].max()
        
        #Normalize the data
        normalized_data = data.copy()
        normalized_data['Counts per Second'] = normalized_data['Counts per Second'] / max_value
        
        #Add the normalized data to the dictionary
        normalized_data_dict[key] = normalized_data

        #Add the display element names
        normalized_data['Display Element Name'] = data['Display Element Name'].iloc[0]
        
        print(f"Data for {key} successfully normalized\n")
    return normalized_data_dict


#Simple peak finding function
def find_peaks_simple(data, start_channel, channel_step, cutoff_value, threshold_factor=2):

    ### Debugging
    print(f"Data type: {type(data)}")
    print(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")

    peaks_dict = {}  # Dictionary to store peaks for each element

    for key, data in data.items():
        peaks = []  # List to store peaks for the current element
        saved_indices = set()  # Set to keep track of saved indices

        #Set the current start and end channels
        current_start_channel = start_channel
        current_end_channel = start_channel + channel_step

        while current_end_channel <= len(data):
            #Slice the data
            #print(f"Trying slice: {current_start_channel} to {current_end_channel}")       #Debug line for loop checking

            sliced_data = data.iloc[current_start_channel:current_end_channel]
            min_peak_value = sliced_data['Counts per Second'].mean()

            # Check if the slice is empty
            if sliced_data.empty:
                print(f"Warning: Empty slice for {key} from {current_start_channel} to {current_end_channel}.")
                break
            
            #Calculate the max and average
            max_value = sliced_data['Counts per Second'].max()
            avg_value = sliced_data['Counts per Second'].mean()
            peak_index = sliced_data['Counts per Second'].idxmax()
            

            #Compare the max value to the threshold
            if max_value > threshold_factor * avg_value and max_value > max(min_peak_value, cutoff_value):               
                if peak_index not in saved_indices:
                    # Save the peak if it hasn't been saved before
                    peaks.append(data.iloc[peak_index])
                    saved_indices.add(peak_index)  # Add the index to the set of saved indices
                    print(f"Peak found for {key} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
                    if peak_index - int(channel_step / 2) < 0:
                        current_start_channel = 0
                        current_end_channel += channel_step
                    else:
                        current_start_channel = peak_index - int(channel_step / 2)
                        current_end_channel = min(current_start_channel + channel_step, len(data) + 1)
                    print(f"Updated to: {current_start_channel} → {current_end_channel}")
                else:
                    #Skip to the next window if the peak has already been saved
                    print(f"Duplicate peak found for {key} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
                    current_start_channel += channel_step 
                    current_end_channel += channel_step
                    print(f"Updated to: {current_start_channel} → {current_end_channel}")

            else:
                print(f"No peak found for {key} in channel range {current_start_channel} to {current_end_channel} (Max: {max_value}, Avg: ", avg_value * threshold_factor, ")")       #debug for loop checking
                current_start_channel += channel_step
                current_end_channel += channel_step
                print(f"Updated to: {current_start_channel} → {current_end_channel}")          #debug for loop checking

        if peaks:
            peaks_dict[key] = pd.DataFrame(peaks)
        else:
            print(f"No peaks found for {key}.")
            peaks_dict[key] = pd.DataFrame(columns=data.columns)  # Create an empty DataFrame with the same columns
        
        # # Debugging: Print detected peaks (for loop checking)
        print(f"Peaks for {key}:")
        print(peaks_dict[key][['Channel', 'Counts per Second']])
        print("-" * 50)
        
    return peaks_dict

#Peak-to-Compton function
def peak_to_compton(compton_data, gaussian_data, element, compton_range=(1040, 1096)):
    element_data = compton_data[element]
    element_data['Energy (keV)'] = pd.to_numeric(element_data['Energy (keV)'], errors='coerce')
    peak_count_rate=gaussian_data['Co']['Counts per Second'].max()
    #Slice the compton data from 1040 keV to 1096 keV
    compton_data_slice = element_data[
        (element_data['Energy (keV)'] >= compton_range[0]) & 
        (element_data['Energy (keV)'] <= compton_range[1])
    ]
    #compton_data['Counts per Second'] = compton_data['Counts'] / 180
    #Get the mean of the compton data
    compton_mean = compton_data_slice['Counts per Second'].mean()
    peak_counts = gaussian_data['Co']['Counts'].max()
    #print(compton_data_slice['Counts'], compton_data_slice['Counts per Second'])   #for debugging
    #Divide the mean by the peak energy for Co at 1332 keV
    ptc_ratio = peak_count_rate / compton_mean

    return ptc_ratio, compton_mean, peak_count_rate, peak_counts, compton_data_slice

def load_data(file_path):
    data = pd.read_csv(file_path, header=None, sep=r'\s+', names=["Counts"])

    if data.empty:
        print(f"Warning: {file_path} is empty.")
        return None, None
    else:
        #Extract the total measuring time
        total_time = data.iloc[0, 0]

        #Remove the first two rows from the data
        data = data.iloc[2:].reset_index(drop=True)

        #Add a "Channels"" column to the data
        data['Channel'] = np.arange(1, len(data) + 1)

        #Divide the data values by the total measuring time to get counts per second
        data['Counts per Second'] = data['Counts'] / total_time

        #Reorder the data columns to have the "Row Number" first, then "Counts per Second"
        data = data[['Channel', 'Counts per Second', 'Counts']]

    return data, total_time

