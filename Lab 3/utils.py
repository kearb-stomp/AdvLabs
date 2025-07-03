import pandas as pd

# Function to find the maximum intensity for each integer interval
# def find_peaks(df, cutoff):
#     peaks = []
#     peaks_mass = []
    
#     # Sort the DataFrame by Mass
#     df_sorted = df.sort_values(by='Mass')
    
#     max_intensities = []
#     current_interval = None
    
#     # Iterate through the sorted DataFrame
#     for _, row in df_sorted.iterrows():
#         mass = row['Mass']
#         intensity = row['Intensity']
        
#         if current_interval is None or mass >= current_interval + 1:
#             # Start a new interval
#             current_interval = int(mass)
#             max_intensities.append((mass, intensity))
#         else:
#             # Update the maximum value in the current interval
#             last_mass, last_max_intensity = max_intensities[-1]
#             if intensity > last_max_intensity:
#                 max_intensities[-1] = (last_mass, intensity)
    
#     # Filter out intervals with values less than the cutoff
#     for mass, intensity in max_intensities:
#         if intensity >= cutoff:
#             peaks.append(intensity)
#             peaks_mass.append(mass)
    
#     return peaks, peaks_mass

# def find_peaks(df, cutoff, interval):
#     peaks = []
#     peaks_mass = []

#     df_sorted = df.sort_values(by='Mass')

#     max_intensities = []          # list of (mass, intensity)
#     current_interval = None

#     for _, row in df_sorted.iterrows():
#         mass = row['Mass']
#         intensity = row['Intensity']

#         if current_interval is None or mass >= current_interval + interval:
#             # start new interval
#             current_interval = int(mass)
#             max_intensities.append((mass, intensity))
#         else:
#             # update max in current interval *and keep its mass*
#             last_mass, last_max_int = max_intensities[-1]
#             if intensity > last_max_int:
#                 max_intensities[-1] = (mass, intensity)   # <- mass not last_mass

#     # keep only peaks ≥ cutoff
#     for mass, intensity in max_intensities:
#         if intensity >= cutoff:
#             peaks.append(intensity)
#             peaks_mass.append(mass)

#     return peaks, peaks_mass

def find_peaks(df, cutoff, interval=1.0):
    """
    Pick the highest Intensity in each `interval`-wide mass bin and
    keep peaks whose height ≥ cutoff.

    Parameters
    ----------
    df : DataFrame with columns 'Mass' and 'Intensity'
    cutoff : float
        Minimum Intensity to accept.
    interval : float
        Bin width in amu (e.g. 1.5).  Bin start is always the integer floor
        of the first mass that lands in it: [floor(m), floor(m)+interval).

    Returns
    -------
    peaks : list[float]      # intensities
    peaks_mass : list[float] # corresponding masses
    """
    peaks, peaks_mass = [], []

    df_sorted = df.sort_values("Mass")

    max_intensities = []      # stores (mass, intensity) per bin
    current_bin_start = None  # left edge of the active bin

    for _, row in df_sorted.iterrows():
        m = row["Mass"]
        I = row["Intensity"]

        # initialise or open a new bin when m is outside the current one
        if current_bin_start is None or m >= current_bin_start + interval:
            current_bin_start = int(m)          # floor to nearest integer
            max_intensities.append((m, I))
        else:
            # update max within the bin
            best_m, best_I = max_intensities[-1]
            if I > best_I:
                max_intensities[-1] = (m, I)    # keep the better mass & intensity

    # keep only those above cutoff
    for m, I in max_intensities:
        if I >= cutoff:
            peaks.append(I)
            peaks_mass.append(m)

    return peaks, peaks_mass

# Data load

def load_scan(path):
    return (
        pd.read_csv(
            path,
            skiprows=22,              # header lines
            sep=",",                 # comma-delimited (trailing comma OK)
            skipinitialspace=True,
            names=["Mass", "Pressure (uncorrected)", "trash"],
            usecols=[0, 1],          # keep just the first two columns
            dtype=float
        )
        .set_index("Mass")           # Mass as the index → easy alignment
    )
