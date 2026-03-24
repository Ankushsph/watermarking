# Step 3: SS-MOD - Spread Spectrum Modulation

import numpy as np
from .config import *

def generate_pn_sequence(length, seed=42):
    """
    Generate Pseudo-Noise sequence for spreading
    Uses Linear Feedback Shift Register (LFSR)
    """
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=length)

def spread_spectrum_modulate(encoded_bits, spreading_factor=SPREADING_FACTOR):
    """
    Step 3: SS-MOD - Spread encoded bits across full audio
    
    Each bit is spread using a PN sequence, making it more robust to attacks
    
    Args:
        encoded_bits: BCH-encoded watermark bits
        spreading_factor: Number of chips per bit
    
    Returns:
        spread_bits: Spread spectrum modulated bits
        pn_sequences: PN sequences used for spreading (needed for extraction)
    """
    spread_bits = []
    pn_sequences = []
    
    for bit in encoded_bits:
        # Generate PN sequence for this bit
        pn_seq = generate_pn_sequence(spreading_factor)
        pn_sequences.append(pn_seq)
        
        # BPSK modulation: 0 -> -1, 1 -> +1
        bit_value = 1 if bit == 1 else -1
        
        # Spread the bit
        spread = bit_value * pn_seq
        spread_bits.extend(spread)
    
    print(f"SS-MOD Spread Spectrum:")
    print(f"  Input bits: {len(encoded_bits)}")
    print(f"  Spreading factor: {spreading_factor}")
    print(f"  Output chips: {len(spread_bits)}")
    
    return np.array(spread_bits), pn_sequences

def spread_spectrum_demodulate(received_chips, pn_sequences, spreading_factor=SPREADING_FACTOR):
    """
    Step 3: SS-MOD - Despread received chips to recover bits
    
    Args:
        received_chips: Received spread spectrum signal
        pn_sequences: PN sequences used during spreading
        spreading_factor: Number of chips per bit
    
    Returns:
        recovered_bits: Despread bits
    """
    num_bits = len(pn_sequences)
    recovered_bits = []
    
    for i in range(num_bits):
        start_idx = i * spreading_factor
        end_idx = start_idx + spreading_factor
        
        if end_idx > len(received_chips):
            # Pad if necessary
            chip_segment = np.pad(received_chips[start_idx:], 
                                 (0, end_idx - len(received_chips)), 
                                 mode='constant')
        else:
            chip_segment = received_chips[start_idx:end_idx]
        
        # Correlate with PN sequence
        pn_seq = pn_sequences[i]
        correlation = np.sum(chip_segment * pn_seq)
        
        # Decision: positive correlation -> bit 1, negative -> bit 0
        bit = 1 if correlation > 0 else 0
        recovered_bits.append(bit)
    
    print(f"SS-MOD Despreading:")
    print(f"  Input chips: {len(received_chips)}")
    print(f"  Recovered bits: {len(recovered_bits)}")
    
    return np.array(recovered_bits, dtype=int)

def map_spread_bits_to_frames(spread_bits, num_frames):
    """
    Map spread spectrum bits to audio frames
    Distributes bits evenly across all frames
    
    Args:
        spread_bits: Spread spectrum modulated bits
        num_frames: Total number of frames available
    
    Returns:
        frame_bit_map: Dictionary mapping frame_idx -> bit_value
    """
    frame_bit_map = {}
    
    # Distribute bits evenly across frames
    bits_per_frame = len(spread_bits) / num_frames
    
    for i, bit in enumerate(spread_bits):
        frame_idx = int(i / bits_per_frame)
        if frame_idx >= num_frames:
            frame_idx = num_frames - 1
        
        if frame_idx not in frame_bit_map:
            frame_bit_map[frame_idx] = []
        frame_bit_map[frame_idx].append(bit)
    
    # Average multiple bits per frame
    for frame_idx in frame_bit_map:
        frame_bit_map[frame_idx] = np.mean(frame_bit_map[frame_idx])
    
    return frame_bit_map
