#!/usr/bin/env python3
import numpy as np
import glob

def main():
    # Find all files matching "output_*.dat" but exclude "output_average.dat"
    files = sorted(glob.glob("output_*.dat"))
    files = [f for f in files if f != "output_average.dat"]
    
    if not files:
        print("No output files found!")
        return

    ref_freq = None
    normalized_amp_list = []

    # Process each file individually
    for f in files:
        try:
            data = np.loadtxt(f, comments='#')
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

        # Ensure data is 2D
        if data.ndim == 1:
            data = data[np.newaxis, :]

        freq = data[:, 0]
        amp = data[:, 1]

        # Select frequency range from 400 to 5000 cm^-1
        mask = (freq >= 400) & (freq <= 5000)
        if not np.any(mask):
            print(f"No frequency points in the range 400 to 5000 cm^-1 in file {f}. Skipping.")
            continue

        # Compute normalization factor as the maximum amplitude in the desired range
        norm_factor = np.max(amp[mask])
        if norm_factor == 0:
            print(f"Normalization factor is zero in file {f}. Skipping.")
            continue

        # Normalize all amplitudes to 100 using the factor computed in the range
        normalized_amp = amp / norm_factor * 100.0

        # Restrict to the selected frequency range
        freq_range = freq[mask]
        normalized_amp_range = normalized_amp[mask]

        # Set the reference frequency grid from the first file
        if ref_freq is None:
            ref_freq = freq_range
        else:
            if not np.allclose(ref_freq, freq_range, rtol=1e-5, atol=1e-8):
                print(f"Frequency grid in file {f} does not match the reference. Exiting.")
                return

        normalized_amp_list.append(normalized_amp_range)

    if not normalized_amp_list:
        print("No valid data to process.")
        return

    # Convert list to array: shape = (number of files, number of frequency points)
    amp_array = np.array(normalized_amp_list)
    avg_amp = np.mean(amp_array, axis=0)

    # Write the averaged output to output_average.dat
    output_filename = "output_average.dat"
    with open(output_filename, "w") as fout:
        fout.write("# Frequency (cm^-1)    Average_Normalized_FFT_Amplitude\n")
        for f_val, avg_val in zip(ref_freq, avg_amp):
            fout.write(f"{f_val:.6e} {avg_val:.6e}\n")
    print(f"Averaged FFT output written to {output_filename}")

if __name__ == "__main__":
    main()
