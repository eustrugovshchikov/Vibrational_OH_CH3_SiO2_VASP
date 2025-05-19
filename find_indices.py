#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified find_indices.py script for functionalized SiO2 surfaces.
It uses H atoms (from the XDATCAR header) as starting points.
Based on the command-line argument (1,2,3,4), it outputs:
  1) Mode 1 (only OH): Bonds: H–O and O–Si, Angle: H–O–Si.
  2) Mode 2 (only CH3): Bonds: C–H (for each H in CH3) and C–Si;
       Angles: all H–C–H combinations and H–C–Si angles.
  3) Mode 3 (only surface angles): Angles at the Si center between any two functional groups.
  4) Mode 4 (all): Both groups’ bonds/angles plus the surface angles.
All indices are output in 1-indexed format.
"""

import numpy as np
import sys

def get_element_list(xdatcar_lines):
    """
    Constructs a list of element symbols for each atom based on the header.
    Assumes line 6 holds element symbols (e.g., "Si O C H") and line 7 holds counts.
    """
    elements = xdatcar_lines[5].split()
    counts = list(map(int, xdatcar_lines[6].split()))
    elem_list = []
    for e, count in zip(elements, counts):
        elem_list.extend([e] * count)
    return elem_list

def get_H_indices(xdatcar_lines, element_list):
    """
    Returns the indices (0-indexed) of all atoms that are hydrogen.
    """
    return [i for i, e in enumerate(element_list) if e.upper() == 'H']

def read_xdatcar(filename="XDATCAR"):
    """
    Reads the first configuration from XDATCAR.
    Assumes:
      - Line 1: comment
      - Line 2: scaling factor
      - Lines 3-5: lattice vectors
      - Line 6: element symbols
      - Line 7: atom counts
      - Line 8: coordinate type
    Then a configuration block starts (identified by a line containing "configuration").
    Returns a list of fractional coordinate numpy arrays (one per atom) and the lattice (3x3 array).
    """
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
        raise ValueError("Could not find configuration block in XDATCAR.")
    positions = []
    for i in range(total_atoms):
        pos_line = lines[start + i].strip().split()
        if pos_line:
            pos = [float(x) for x in pos_line[:3]]
            positions.append(np.array(pos))
    return positions, lattice

def apply_pbc(diff, lattice):
    """Applies the minimum image convention to a fractional difference and converts to Cartesian."""
    diff = diff - np.round(diff)
    return np.dot(diff, lattice)

def distance(i, j, positions, lattice):
    """Returns the Cartesian distance between atoms i and j."""
    diff = positions[j] - positions[i]
    cart_diff = apply_pbc(diff, lattice)
    return np.linalg.norm(cart_diff)

def find_closest_neighbor(target_index, positions, lattice, exclude=None):
    """
    Finds the closest neighbor of atom at target_index among positions,
    excluding any indices in the optional list 'exclude'.
    Returns (closest_index, distance).
    """
    min_dist = float('inf')
    closest_index = None
    for i in range(len(positions)):
        if exclude is not None and i in exclude:
            continue
        d = distance(target_index, i, positions, lattice)
        if d < 1e-8:
            continue
        if d < min_dist:
            min_dist = d
            closest_index = i
    return closest_index, min_dist

def find_closest_neighbor_of_type(target_index, positions, lattice, element_list, target_element, exclude=None):
    """
    Finds the closest neighbor of atom at target_index that has element matching target_element.
    Returns (closest_index, distance) or (None, None) if not found.
    """
    min_dist = float('inf')
    closest_index = None
    for i in range(len(positions)):
        if exclude is not None and i in exclude:
            continue
        if element_list[i].upper() != target_element.upper():
            continue
        d = distance(target_index, i, positions, lattice)
        if d < 1e-8:
            continue
        if d < min_dist:
            min_dist = d
            closest_index = i
    return closest_index, min_dist

def main():
    # Get group selection from command-line argument.
    if len(sys.argv) < 2:
        print("Usage: python find_indices.py <group_mode>")
        print("   group_mode: 1=OH only, 2=CH3 only, 3=Surface angles only, 4=All")
        sys.exit(1)
    try:
        group_mode = int(sys.argv[1])
    except ValueError:
        print("Invalid group_mode. Must be an integer (1,2,3,4).")
        sys.exit(1)
    if group_mode not in (1,2,3,4):
        print("group_mode must be 1, 2, 3, or 4.")
        sys.exit(1)

    positions, lattice = read_xdatcar("XDATCAR")
    with open("XDATCAR", "r") as f:
        lines = f.readlines()
    element_list = get_element_list(lines)
    h_indices = get_H_indices(lines, element_list)
    
    output_lines = []
    si_attachments = {}  # key: Si index, value: list of (group_type, group_atom)
    ch3_groups = {}      # key: C index, value: list of H indices (for CH3)
    oh_groups = []       # list of tuples (H, O, Si) for OH groups

    # Process each H atom as starting point.
    for H in h_indices:
        neighbor, d = find_closest_neighbor(H, positions, lattice, exclude=[H])
        if neighbor is None:
            continue
        neighbor_el = element_list[neighbor].upper()
        if neighbor_el == 'O':
            # Assume OH group.
            Si, d2 = find_closest_neighbor_of_type(neighbor, positions, lattice, element_list, 'Si', exclude=[H, neighbor])
            if Si is None:
                continue
            oh_groups.append((H, neighbor, Si))
            si_attachments.setdefault(Si, []).append(('O', neighbor))
        elif neighbor_el == 'C':
            # Assume CH3 group.
            C = neighbor
            ch3_groups.setdefault(C, []).append(H)
        else:
            continue

    # Process CH3 groups: only keep groups with at least 3 H atoms.
    ch3_groups_processed = []
    for C, H_list in ch3_groups.items():
        if len(H_list) < 3:
            print(f"Warning: CH3 group with C index {C+1} has only {len(H_list)} H atoms. Skipping.")
            continue
        if len(H_list) > 3:
            H_list = sorted(H_list, key=lambda H: distance(C, H, positions, lattice))[:3]
        exclude_list = H_list + [C]
        Si, d3 = find_closest_neighbor_of_type(C, positions, lattice, element_list, 'Si', exclude=exclude_list)
        if Si is None:
            continue
        ch3_groups_processed.append((C, H_list, Si))
        si_attachments.setdefault(Si, []).append(('C', C))
    
    # Depending on group_mode, output different sets of bonds/angles.
    # Mode 1: Only OH groups.
    if group_mode in (1,4):
        for (H, O, Si) in oh_groups:
            output_lines.append(f"2 {H+1} {O+1}")      # Bond: H-O
            output_lines.append(f"2 {O+1} {Si+1}")      # Bond: O-Si
            output_lines.append(f"3 {H+1} {O+1} {Si+1}")  # Angle: H-O-Si

    # Mode 2: Only CH3 groups.
    if group_mode in (2,4):
        for (C, H_list, Si) in ch3_groups_processed:
            for H in H_list:
                output_lines.append(f"2 {C+1} {H+1}")    # Bond: C-H
            output_lines.append(f"2 {C+1} {Si+1}")        # Bond: C-Si
            # H-C-H angles (all combinations)
            if len(H_list) == 3:
                output_lines.append(f"3 {H_list[0]+1} {C+1} {H_list[1]+1}")
                output_lines.append(f"3 {H_list[0]+1} {C+1} {H_list[2]+1}")
                output_lines.append(f"3 {H_list[1]+1} {C+1} {H_list[2]+1}")
            # H-C-Si angles for each H.
            for H in H_list:
                output_lines.append(f"3 {H+1} {C+1} {Si+1}")
    
    # Mode 3: Only surface angles (Si at the center).
    if group_mode in (3,4):
        for Si, attachments in si_attachments.items():
            if len(attachments) < 2:
                continue
            n = len(attachments)
            for i in range(n):
                for j in range(i+1, n):
                    # attachments[i] and attachments[j] are tuples (group_type, atom_index).
                    output_lines.append(f"3 {attachments[i][1]+1} {Si+1} {attachments[j][1]+1}")
    
    with open("config-out.dat", "w") as f:
        for line in output_lines:
            f.write(line + "\n")
    print("Processing complete. Output written to config-out.dat.")

if __name__ == "__main__":
    main()
