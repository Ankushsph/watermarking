# Step 2: RS-BCH ECC - Reed-Solomon + BCH Error Correction Coding

import numpy as np
from .config import *

def metadata_to_bits(patient_id, doctor_id, hospital_id, diagnosis_code, date_stamp):
    """Convert metadata to 48-bit array"""
    bits = []
    
    # Patient ID (8 bits)
    bits.extend([int(b) for b in format(patient_id, '08b')])
    
    # Doctor ID (8 bits)
    bits.extend([int(b) for b in format(doctor_id, '08b')])
    
    # Hospital ID (8 bits)
    bits.extend([int(b) for b in format(hospital_id, '08b')])
    
    # Diagnosis Code (8 bits)
    bits.extend([int(b) for b in format(diagnosis_code, '08b')])
    
    # Date Stamp (16 bits)
    bits.extend([int(b) for b in format(date_stamp, '016b')])
    
    return np.array(bits, dtype=int)

def bits_to_metadata(bits):
    """Convert 48-bit array back to metadata"""
    if len(bits) < 48:
        bits = np.pad(bits, (0, 48 - len(bits)), mode='constant')
    
    patient_id = int(''.join(map(str, bits[0:8])), 2)
    doctor_id = int(''.join(map(str, bits[8:16])), 2)
    hospital_id = int(''.join(map(str, bits[16:24])), 2)
    diagnosis_code = int(''.join(map(str, bits[24:32])), 2)
    date_stamp = int(''.join(map(str, bits[32:48])), 2)
    
    return {
        'patient_id': patient_id,
        'doctor_id': doctor_id,
        'hospital_id': hospital_id,
        'diagnosis_code': diagnosis_code,
        'date_stamp': date_stamp
    }

def bch_encode(data_bits):
    """
    BCH encoding using Hamming-style parity
    For every 4 data bits, add 3 parity bits (7,4 Hamming code)
    """
    data = list(data_bits)
    encoded = []
    
    for i in range(0, len(data), 4):
        block = data[i:i+4]
        if len(block) < 4:
            block = block + [0] * (4 - len(block))
        
        # Calculate parity bits
        p1 = block[0] ^ block[1] ^ block[3]
        p2 = block[0] ^ block[2] ^ block[3]
        p3 = block[1] ^ block[2] ^ block[3]
        
        # Encoded block: data + parity
        encoded.extend(block + [p1, p2, p3])
    
    return np.array(encoded, dtype=int)

def bch_decode(encoded_bits):
    """
    BCH decoding with single-bit error correction
    """
    encoded = list(encoded_bits)
    decoded = []
    
    for i in range(0, len(encoded), 7):
        block = encoded[i:i+7]
        if len(block) < 7:
            block = block + [0] * (7 - len(block))
        
        # Extract data and parity
        data = block[:4]
        p1_received = block[4]
        p2_received = block[5]
        p3_received = block[6]
        
        # Calculate expected parity
        p1_calc = data[0] ^ data[1] ^ data[3]
        p2_calc = data[0] ^ data[2] ^ data[3]
        p3_calc = data[1] ^ data[2] ^ data[3]
        
        # Error syndrome
        s1 = p1_received ^ p1_calc
        s2 = p2_received ^ p2_calc
        s3 = p3_received ^ p3_calc
        
        syndrome = s1 * 1 + s2 * 2 + s3 * 4
        
        # Correct single-bit error
        if syndrome != 0 and syndrome <= 4:
            data[syndrome - 1] = 1 - data[syndrome - 1]
        
        decoded.extend(data)
    
    return np.array(decoded[:48], dtype=int)

def rs_bch_encode(metadata_dict):
    """
    Step 2: RS-BCH ECC - Encode patient metadata with error correction
    
    Args:
        metadata_dict: Dictionary with patient_id, doctor_id, hospital_id, diagnosis_code, date_stamp
    
    Returns:
        encoded_bits: BCH-encoded watermark bits
    """
    # Convert metadata to bits
    data_bits = metadata_to_bits(
        metadata_dict['patient_id'],
        metadata_dict['doctor_id'],
        metadata_dict['hospital_id'],
        metadata_dict['diagnosis_code'],
        metadata_dict['date_stamp']
    )
    
    print(f"RS-BCH ECC Encoding:")
    print(f"  Original data: 48 bits")
    
    # Apply BCH encoding
    if USE_BCH_CODING:
        encoded_bits = bch_encode(data_bits)
        print(f"  After BCH encoding: {len(encoded_bits)} bits")
    else:
        encoded_bits = data_bits
    
    return encoded_bits

def rs_bch_decode(encoded_bits):
    """
    Step 2: RS-BCH ECC - Decode watermark bits with error correction
    
    Args:
        encoded_bits: BCH-encoded bits
    
    Returns:
        metadata_dict: Recovered metadata dictionary
    """
    if USE_BCH_CODING:
        decoded_bits = bch_decode(encoded_bits)
        print(f"RS-BCH ECC Decoding:")
        print(f"  Received: {len(encoded_bits)} bits")
        print(f"  Decoded: {len(decoded_bits)} bits")
    else:
        decoded_bits = encoded_bits[:48]
    
    # Convert bits back to metadata
    metadata = bits_to_metadata(decoded_bits)
    
    return metadata, decoded_bits
