#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified simulation processing script.
Now, instead of prompting the user, it takes three command-line arguments:
   1) The configuration index (line number in config-out.dat, 1-indexed)
   2) The start frame (first simulation step in XDATCAR)
   3) The finish frame (last simulation step in XDATCAR)
It reads the specified configuration from config-out.dat and then processes
the simulation steps from XDATCAR accordingly. Output files are written as:
   output_<config_index>.dat and geo_<config_index>.dat.
Note: Non-ASCII strings (degree symbol and Angstrom) have been replaced by " deg" and " A".
"""

import numpy as np
import sys
import math
import time

def compute_distance(f1, f2, lattice):
    diff = f1 - f2
    diff = diff - np.round(diff)
    cart_diff = np.dot(diff, lattice)
    return np.linalg.norm(cart_diff)

def angle_between_points_pbc(f1, f2, f3, lattice):
    diff1 = f1 - f2
    diff1 = diff1 - np.round(diff1)
    diff2 = f3 - f2
    diff2 = diff2 - np.round(diff2)
    cart1 = np.dot(diff1, lattice)
    cart2 = np.dot(diff2, lattice)
    dot = np.dot(cart1, cart2)
    norm1 = np.linalg.norm(cart1)
    norm2 = np.linalg.norm(cart2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def read_xdatcar(filename="XDATCAR"):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    scaling_factor = float(lines[1].strip())
    a_vector = np.array(list(map(float, lines[2].split()))) * scaling_factor
    b_vector = np.array(list(map(float, lines[3].split()))) * scaling_factor
    c_vector = np.array(list(map(float, lines[4].split()))) * scaling_factor
    lattice = np.array([a_vector, b_vector, c_vector])
    
    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)
    
    start = None
    for i, line in enumerate(lines):
        if "configuration" in line.lower():
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not find the beginning of a configuration in XDATCAR.")
    
    structure = []
    for i in range(total_atoms):
        pos_line = lines[start + i].strip().split()
        if pos_line:
            pos = [float(x) for x in pos_line[:3]]
            structure.append(np.array(pos))
    return structure, lattice

def apply_pbc(diff, lattice):
    diff = diff - np.round(diff)
    return np.dot(diff, lattice)

def find_closest_neighbor(target_index, positions, lattice, exclude=None):
    target = positions[target_index]
    min_dist = float('inf')
    closest_index = None
    for i, pos in enumerate(positions):
        if exclude is not None and i in exclude:
            continue
        diff = pos - target
        cart_diff = apply_pbc(diff, lattice)
        dist = np.linalg.norm(cart_diff)
        if dist < 1e-8:
            continue
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    return closest_index, min_dist

def process_simulation():
    # Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python process_simulation.py <config_index> <start_frame> <finish_frame>")
        sys.exit(1)
    config_index = int(sys.argv[1])
    start_step = int(sys.argv[2])
    finish_step = int(sys.argv[3])
    
    # Read the configuration line from config-out.dat
    try:
        with open("config-out.dat", "r") as f:
            config_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print("Error reading config-out.dat:", e)
        sys.exit(1)
    
    if config_index < 1 or config_index > len(config_lines):
        print("Invalid configuration index.")
        sys.exit(1)
    
    # Use the specified configuration line (assumed 1-indexed)
    config_line = config_lines[config_index - 1]
    tokens = config_line.split()
    try:
        num_points = int(tokens[0])
        indices = list(map(int, tokens[1:]))
        # Convert indices from 1-indexed to 0-indexed
        indices = [i - 1 for i in indices]
    except Exception as e:
        print(f"Error parsing configuration on line {config_index}:", e)
        sys.exit(1)
    
    mode = "angle" if num_points == 3 else "bond"
    print(f"Processing configuration {config_index} with mode: {mode} and atom indices: {[i+1 for i in indices]}")
    
    positions, lattice = read_xdatcar("XDATCAR")
    
    total_steps = finish_step - start_step
    if total_steps <= 0:
        print("Invalid simulation step range.")
        sys.exit(1)
    
    measurements = []
    dt = 0.5e-15  # time step in seconds
    # Compute the starting line in XDATCAR for the simulation steps.
    current_line = 8 + start_step * (len(positions) + 1)  # Adjust if needed based on XDATCAR format.
    
    # Process each simulation step in the range.
    with open("XDATCAR", "r") as f:
        xdatcar_lines = f.readlines()
    for step in range(total_steps):
        frac_coords = []
        for i in range(len(positions)):
            line = xdatcar_lines[current_line + i].strip()
            try:
                frac = np.array(list(map(float, line.split())))
            except Exception as e:
                print(f"Error parsing coordinates on step {start_step + step}, line {current_line+i}: {e}")
                sys.exit(1)
            frac_coords.append(frac)
        current_line += len(positions) + 1
        
        if mode == "angle":
            f1 = frac_coords[indices[0]]
            f2 = frac_coords[indices[1]]
            f3 = frac_coords[indices[2]]
            value = angle_between_points_pbc(f1, f2, f3, lattice)
        else:
            f1 = frac_coords[indices[0]]
            f2 = frac_coords[indices[1]]
            value = compute_distance(f1, f2, lattice)
        measurements.append(value)
    
    # Report the first measurement using plain ASCII strings.
    if measurements:
        if mode == "angle":
            print(f"Angle at first step for configuration {config_index}: {measurements[0]:.2f} deg")
        else:
            print(f"Bond length at first step for configuration {config_index}: {measurements[0]:.2f} A")
    
    # Write geometry data.
    geo_filename = f"geo_{config_index}.dat"
    try:
        with open(geo_filename, "w") as geo_f:
            geo_f.write("# Step\tMeasurement\n")
            for i, meas in enumerate(measurements):
                geo_f.write(f"{start_step + i}\t{meas:.6e}\n")
        print(f"Geometry data for configuration {config_index} written to {geo_filename}")
    except Exception as e:
        print(f"Error writing {geo_filename}:", e)
    
    # Perform FFT on the measurements.
    N_points = len(measurements)
    f = np.fft.fftfreq(N_points, dt)
    # Convert frequencies to cm^-1 (using provided conversion factor)
    f_converted = f * 3.33e-11
    FFT_result = np.fft.fft(measurements)
    
    output_filename = f"output_{config_index}.dat"
    try:
        with open(output_filename, "w") as out_f:
            out_f.write("# Fourier Transform results\n")
            out_f.write("# Frequency (cm^-1)    FFT_Amplitude\n")
            for i in range(N_points):
                out_f.write(f"{f_converted[i]:.6e} {abs(FFT_result[i]):.6e}\n")
        print(f"Fourier Transform data for configuration {config_index} written to {output_filename}")
    except Exception as e:
        print(f"Error writing {output_filename}:", e)

if __name__ == "__main__":
    start_time = time.time()
    process_simulation()
    end_time = time.time()
    print("Total processing time:", end_time - start_time)
