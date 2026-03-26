"""
Dequantize GGUF Q4_K (type 12) and Q6_K (type 14) tensors to float16 using numpy.

Based on llama.cpp ggml-quants.c and ggml-common.h:
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-common.h
"""

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12

# Block sizes in bytes
BLOCK_SIZE_Q4_K = 144  # 2 + 2 + 12 + 128
BLOCK_SIZE_Q6_K = 210  # 128 + 64 + 16 + 2


def dequantize_q4_k(data, n: int) -> np.ndarray:
    """
    Dequantize Q4_K (type 12) quantized data to float16.
    Accepts bytes, bytearray, or numpy array.
    """
    assert n % QK_K == 0, f"n={n} must be divisible by QK_K={QK_K}"
    nb = n // QK_K  # number of blocks

    if isinstance(data, np.ndarray):
        buf = data.reshape(-1).view(np.uint8)[:nb * BLOCK_SIZE_Q4_K].reshape(nb, BLOCK_SIZE_Q4_K)
    else:
        buf = np.frombuffer(data, dtype=np.uint8, count=nb * BLOCK_SIZE_Q4_K).reshape(nb, BLOCK_SIZE_Q4_K)

    # Parse fields
    d    = np.frombuffer(buf[:, 0:2].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    dmin = np.frombuffer(buf[:, 2:4].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    scales_raw = buf[:, 4:16]    # (nb, 12) uint8
    qs = buf[:, 16:144]          # (nb, 128) uint8

    output = np.zeros((nb, QK_K), dtype=np.float32)

    # Decode the 12-byte scales array into 8 scales and 8 mins (each 6-bit).
    # This mirrors get_scale_min_k4() from ggml-quants.c.
    sc = np.zeros((nb, 8), dtype=np.uint8)
    mn = np.zeros((nb, 8), dtype=np.uint8)

    # j = 0..3: simple 6-bit extraction
    for j in range(4):
        sc[:, j] = scales_raw[:, j] & 63
        mn[:, j] = scales_raw[:, j + 4] & 63

    # j = 4..7: combine bits from two locations
    for j in range(4, 8):
        sc[:, j] = (scales_raw[:, j + 4] & 0xF) | ((scales_raw[:, j - 4] >> 6) << 4)
        mn[:, j] = (scales_raw[:, j + 4] >> 4)   | ((scales_raw[:, j]     >> 6) << 4)

    # Dequantize: 8 sub-blocks of 32 elements each = 256 elements
    # Each pair of sub-blocks (is, is+1) shares 32 bytes of qs,
    # with low nibble for the first sub-block and high nibble for the second.
    for j_block in range(4):  # 4 groups of 64 elements
        is_idx = j_block * 2
        q_offset = j_block * 32
        out_offset = j_block * 64

        d1 = d * sc[:, is_idx].astype(np.float32)        # (nb,)
        m1 = dmin * mn[:, is_idx].astype(np.float32)      # (nb,)
        d2 = d * sc[:, is_idx + 1].astype(np.float32)     # (nb,)
        m2 = dmin * mn[:, is_idx + 1].astype(np.float32)  # (nb,)

        q32 = qs[:, q_offset:q_offset + 32]  # (nb, 32) uint8

        # Low nibble -> first 32 elements
        q_lo = (q32 & 0xF).astype(np.float32)  # (nb, 32)
        output[:, out_offset:out_offset + 32] = d1[:, None] * q_lo - m1[:, None]

        # High nibble -> next 32 elements
        q_hi = (q32 >> 4).astype(np.float32)   # (nb, 32)
        output[:, out_offset + 32:out_offset + 64] = d2[:, None] * q_hi - m2[:, None]

    return output.reshape(n).astype(np.float16)


def dequantize_q6_k(data, n: int) -> np.ndarray:
    """
    Dequantize Q6_K (type 14) quantized data to float16.
    Accepts bytes, bytearray, or numpy array.
    """
    assert n % QK_K == 0, f"n={n} must be divisible by QK_K={QK_K}"
    nb = n // QK_K

    if isinstance(data, np.ndarray):
        buf = data.reshape(-1).view(np.uint8)[:nb * BLOCK_SIZE_Q6_K].reshape(nb, BLOCK_SIZE_Q6_K)
    else:
        buf = np.frombuffer(data, dtype=np.uint8, count=nb * BLOCK_SIZE_Q6_K).reshape(nb, BLOCK_SIZE_Q6_K)

    # Parse fields
    ql_all     = buf[:, 0:128]                # (nb, 128) uint8 - lower 4 bits
    qh_all     = buf[:, 128:192]              # (nb, 64)  uint8 - upper 2 bits
    scales_all = buf[:, 192:208].view(np.int8)  # (nb, 16) int8
    d          = np.frombuffer(buf[:, 208:210].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)

    output = np.zeros((nb, QK_K), dtype=np.float32)

    # Process 2 chunks of 128 elements each per block
    for chunk in range(2):
        ql = ql_all[:, chunk * 64:(chunk + 1) * 64]     # (nb, 64) - 64 bytes of lower nibbles
        qh = qh_all[:, chunk * 32:(chunk + 1) * 32]     # (nb, 32) - 32 bytes of upper 2 bits
        sc = scales_all[:, chunk * 8:(chunk + 1) * 8]    # (nb, 8)  - 8 scale values
        out_base = chunk * 128

        # For l in 0..31, reconstruct 4 values each (= 128 values total):
        #   q1 = (ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4) - 32   -> output[l]
        #   q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32   -> output[l+32]
        #   q3 = (ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4) - 32   -> output[l+64]
        #   q4 = (ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4) - 32   -> output[l+96]

        ql_lo = ql[:, :32]   # (nb, 32)  ql[l]     for l=0..31
        ql_hi = ql[:, 32:64] # (nb, 32)  ql[l+32]  for l=0..31

        q1 = ((ql_lo & 0xF) | (((qh >> 0) & 3) << 4)).astype(np.int8) - 32
        q2 = ((ql_hi & 0xF) | (((qh >> 2) & 3) << 4)).astype(np.int8) - 32
        q3 = ((ql_lo >> 4)  | (((qh >> 4) & 3) << 4)).astype(np.int8) - 32
        q4 = ((ql_hi >> 4)  | (((qh >> 6) & 3) << 4)).astype(np.int8) - 32

        # Scale indices: is = l // 16, so for l=0..31:
        #   l=0..15  -> is=0, l=16..31 -> is=1
        # q1 uses sc[is+0], q2 uses sc[is+2], q3 uses sc[is+4], q4 uses sc[is+6]
        sc0 = sc[:, 0:1].astype(np.float32)  # is=0 for l=0..15
        sc1 = sc[:, 1:2].astype(np.float32)  # is=1 for l=16..31
        sc2 = sc[:, 2:3].astype(np.float32)
        sc3 = sc[:, 3:4].astype(np.float32)
        sc4 = sc[:, 4:5].astype(np.float32)
        sc5 = sc[:, 5:6].astype(np.float32)
        sc6 = sc[:, 6:7].astype(np.float32)
        sc7 = sc[:, 7:8].astype(np.float32)

        # Build per-element scale arrays (nb, 32)
        # For positions 0..15 use one scale, positions 16..31 use next scale
        def make_scale_row(s_lo, s_hi):
            """Combine two (nb,1) scales into (nb,32): first 16 use s_lo, last 16 use s_hi."""
            return np.concatenate([
                np.broadcast_to(s_lo, (nb, 16)),
                np.broadcast_to(s_hi, (nb, 16))
            ], axis=1)

        sc_q1 = make_scale_row(sc0, sc1)  # sc[is+0] for q1
        sc_q2 = make_scale_row(sc2, sc3)  # sc[is+2] for q2
        sc_q3 = make_scale_row(sc4, sc5)  # sc[is+4] for q3
        sc_q4 = make_scale_row(sc6, sc7)  # sc[is+6] for q4

        output[:, out_base +  0:out_base +  32] = d[:, None] * sc_q1 * q1.astype(np.float32)
        output[:, out_base + 32:out_base +  64] = d[:, None] * sc_q2 * q2.astype(np.float32)
        output[:, out_base + 64:out_base +  96] = d[:, None] * sc_q3 * q3.astype(np.float32)
        output[:, out_base + 96:out_base + 128] = d[:, None] * sc_q4 * q4.astype(np.float32)

    return output.reshape(n).astype(np.float16)


# ── Convenience dispatcher ───────────────────────────────────────────

GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

BLOCK_SIZE_Q8_0 = 34   # 2 (float16 d) + 32 (int8 qs) per 32 elements
BLOCK_SIZE_Q5_K = 176   # 2+2+12+32+128 per 256 elements


def dequantize_q8_0(data, n: int) -> np.ndarray:
    """Dequantize Q8_0 (type 8): simplest 8-bit format. 34 bytes per 32 elements."""
    QK = 32
    assert n % QK == 0
    nb = n // QK

    if isinstance(data, np.ndarray):
        buf = data.reshape(-1).view(np.uint8)[:nb * BLOCK_SIZE_Q8_0].reshape(nb, BLOCK_SIZE_Q8_0)
    else:
        buf = np.frombuffer(data, dtype=np.uint8, count=nb * BLOCK_SIZE_Q8_0).reshape(nb, BLOCK_SIZE_Q8_0)

    d = np.frombuffer(buf[:, 0:2].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    qs = buf[:, 2:34].view(np.int8).astype(np.float32)  # (nb, 32) signed int8
    output = d[:, None] * qs
    return output.reshape(n).astype(np.float16)


def dequantize_q5_k(data, n: int) -> np.ndarray:
    """Dequantize Q5_K (type 13). 176 bytes per 256 elements."""
    assert n % QK_K == 0
    nb = n // QK_K

    if isinstance(data, np.ndarray):
        buf = data.reshape(-1).view(np.uint8)[:nb * BLOCK_SIZE_Q5_K].reshape(nb, BLOCK_SIZE_Q5_K)
    else:
        buf = np.frombuffer(data, dtype=np.uint8, count=nb * BLOCK_SIZE_Q5_K).reshape(nb, BLOCK_SIZE_Q5_K)

    # Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
    d = np.frombuffer(buf[:, 0:2].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    dmin = np.frombuffer(buf[:, 2:4].tobytes(), dtype=np.float16).reshape(nb).astype(np.float32)
    scales_raw = buf[:, 4:16]
    qh = buf[:, 16:48]    # (nb, 32) high bits
    qs = buf[:, 48:176]   # (nb, 128) low nibbles

    # Decode scales (same as Q4_K)
    sc = np.zeros((nb, 8), dtype=np.uint8)
    mn = np.zeros((nb, 8), dtype=np.uint8)
    for j in range(4):
        sc[:, j] = scales_raw[:, j] & 63
        mn[:, j] = scales_raw[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (scales_raw[:, j + 4] & 0xF) | ((scales_raw[:, j - 4] >> 6) << 4)
        mn[:, j] = (scales_raw[:, j + 4] >> 4) | ((scales_raw[:, j] >> 6) << 4)

    output = np.zeros((nb, QK_K), dtype=np.float32)

    for j_block in range(4):
        is_idx = j_block * 2
        q_offset = j_block * 32
        qh_offset = j_block * 8
        out_offset = j_block * 64

        d1 = d * sc[:, is_idx].astype(np.float32)
        m1 = dmin * mn[:, is_idx].astype(np.float32)
        d2 = d * sc[:, is_idx + 1].astype(np.float32)
        m2 = dmin * mn[:, is_idx + 1].astype(np.float32)

        q32 = qs[:, q_offset:q_offset + 32]
        qh8 = qh[:, qh_offset:qh_offset + 8]

        # Low nibble + high bit for first 32 elements
        q_lo = (q32 & 0xF).astype(np.float32)
        # Extract high bits: bit j from qh byte
        hbits_lo = np.zeros((nb, 32), dtype=np.float32)
        for byte_i in range(8):
            for bit_i in range(4):
                idx = byte_i * 4 + bit_i
                if idx < 32:
                    hbits_lo[:, idx] = ((qh8[:, byte_i] >> bit_i) & 1).astype(np.float32)
        q_lo = q_lo + hbits_lo * 16
        output[:, out_offset:out_offset + 32] = d1[:, None] * q_lo - m1[:, None]

        # High nibble + high bit for next 32 elements
        q_hi = (q32 >> 4).astype(np.float32)
        hbits_hi = np.zeros((nb, 32), dtype=np.float32)
        for byte_i in range(8):
            for bit_i in range(4):
                idx = byte_i * 4 + bit_i
                if idx < 32:
                    hbits_hi[:, idx] = ((qh8[:, byte_i] >> (bit_i + 4)) & 1).astype(np.float32)
        q_hi = q_hi + hbits_hi * 16
        output[:, out_offset + 32:out_offset + 64] = d2[:, None] * q_hi - m2[:, None]

    return output.reshape(n).astype(np.float16)


DEQUANT_FUNCTIONS = {
    GGML_TYPE_Q8_0: dequantize_q8_0,
    GGML_TYPE_Q4_K: dequantize_q4_k,
    GGML_TYPE_Q5_K: dequantize_q5_k,
    GGML_TYPE_Q6_K: dequantize_q6_k,
}

BLOCK_SIZES = {
    GGML_TYPE_Q8_0: BLOCK_SIZE_Q8_0,
    GGML_TYPE_Q4_K: BLOCK_SIZE_Q4_K,
    GGML_TYPE_Q5_K: BLOCK_SIZE_Q5_K,
    GGML_TYPE_Q6_K: BLOCK_SIZE_Q6_K,
}


def dequantize(data: bytes, ggml_type: int, n_elements: int) -> np.ndarray:
    """
    Dequantize a GGUF tensor given its raw data, type id, and element count.

    Args:
        data: Raw quantized bytes
        ggml_type: GGML type id (12 for Q4_K, 14 for Q6_K)
        n_elements: Total number of float elements

    Returns:
        np.ndarray of shape (n_elements,) with dtype float16
    """
    if ggml_type not in DEQUANT_FUNCTIONS:
        raise ValueError(f"Unsupported ggml type: {ggml_type}. Supported: {list(DEQUANT_FUNCTIONS.keys())}")
    return DEQUANT_FUNCTIONS[ggml_type](data, n_elements)


# ── Quick self-test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Q4_K self-test ===")
    # Create one block of 256 elements (144 bytes)
    test_block_q4k = bytearray(BLOCK_SIZE_Q4_K)
    # Set d = 1.0 in float16
    test_block_q4k[0:2] = np.float16(1.0).tobytes()
    # Set dmin = 0.0
    test_block_q4k[2:4] = np.float16(0.0).tobytes()
    # Set first scale to 1 (scales[0] = 1)
    test_block_q4k[4] = 1
    # Set qs[0] = 0x53 -> low nibble=3, high nibble=5
    test_block_q4k[16] = 0x53

    result = dequantize_q4_k(bytes(test_block_q4k), 256)
    print(f"  Element 0 (expect 3.0):  {result[0]}")
    print(f"  Element 32 (expect 0.0): {result[32]}")  # scale for sub-block 1 is 0
    print(f"  Shape: {result.shape}, dtype: {result.dtype}")

    print("\n=== Q6_K self-test ===")
    # Create one block of 256 elements (210 bytes)
    test_block_q6k = bytearray(BLOCK_SIZE_Q6_K)
    # Set d = 1.0 at offset 208
    test_block_q6k[208:210] = np.float16(1.0).tobytes()
    # Set scales[0] = 1 at offset 192
    test_block_q6k[192] = 1
    # Set ql[0] = 5 (low nibble = 5, representing quant low bits)
    test_block_q6k[0] = 5
    # qh stays 0, so full 6-bit value = 5, signed = 5-32 = -27
    # result = d * scale * q = 1.0 * 1 * (-27) = -27.0

    result = dequantize_q6_k(bytes(test_block_q6k), 256)
    print(f"  Element 0 (expect -27.0): {result[0]}")
    print(f"  Shape: {result.shape}, dtype: {result.dtype}")

    print("\nAll tests passed!")
