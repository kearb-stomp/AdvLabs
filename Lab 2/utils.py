#Utility functions created for Lab 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import json

print("Matplotlib imported as plt:", plt)

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
def find_peaks_simple(data, start_channel, channel_step, threshold_factor=2):
    peaks_dict = {}  # Dictionary to store peaks for each element

    for element_name, data in data.items():
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
                print(f"Warning: Empty slice for {element_name} from {current_start_channel} to {current_end_channel}.")
                break
            
            #Calculate the max and average
            max_value = sliced_data['Counts per Second'].max()
            avg_value = sliced_data['Counts per Second'].mean()
            peak_index = sliced_data['Counts per Second'].idxmax()
            

            #Compare the max value to the threshold
            if max_value > threshold_factor * avg_value and max_value > max(min_peak_value, 0.10):               
                if peak_index not in saved_indices:
                    # Save the peak if it hasn't been saved before
                    peaks.append(data.iloc[peak_index])
                    saved_indices.add(peak_index)  # Add the index to the set of saved indices
                    print(f"Peak found for {element_name} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
                    current_start_channel = peak_index - 25
                    current_end_channel = min(current_start_channel + channel_step, len(data) + 1)
                    #print(f"Updated to: {current_start_channel} → {current_end_channel}")
                else:
                    #Skip to the next window if the peak has already been saved
                    print(f"Duplicate peak found for {element_name} at channel {data['Channel'].iloc[peak_index]} with value {data['Counts per Second'].iloc[peak_index]} (Max: {max_value}, Avg: {avg_value})")
                    current_start_channel += channel_step 
                    current_end_channel += channel_step
                    #print(f"Updated to: {current_start_channel} → {current_end_channel}")

            else:
                #print(f"No peak found for {element_name} in channel range {current_start_channel} to {current_end_channel} (Max: {max_value}, Avg: ", avg_value * threshold_factor, ")")       #debug for loop checking
                current_start_channel = current_start_channel + 50
                current_end_channel += channel_step
                #print(f"Updated to: {current_start_channel} → {current_end_channel}")          #debug for loop checking

        if peaks:
            peaks_dict[element_name] = pd.DataFrame(peaks)
        else:
            print(f"No peaks found for {element_name}.")
            peaks_dict[element_name] = pd.DataFrame(columns=data.columns)  # Create an empty DataFrame with the same columns
        
        # # Debugging: Print detected peaks (for loop checking)
        # print(f"Peaks for {element_name}:")
        # print(peaks_dict[element_name][['Channel', 'Counts per Second']])
        # print("-" * 50)
        
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