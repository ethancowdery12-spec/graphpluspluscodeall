"""
patch_gguf.py — Fix qwen35 GGUF metadata so llama.cpp can load it.

The model was quantized with a version of the GGUF tooling that stored
hyperparameter keys under the "qwen35." namespace.  Modern llama.cpp
still registers "qwen35" as an architecture alias for qwen2, but looks
for the keys under the "qwen2." prefix.  This script renames every
"qwen35." KV key to "qwen2." while leaving all tensor bytes untouched.

Usage:  python patch_gguf.py
Output: graphrag-plus-plus-qwen35-4b-q3_k_m-patched.gguf
"""
import struct, os, sys

INPUT  = "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"
OUTPUT = "graphrag-plus-plus-qwen35-4b-q3_k_m-patched.gguf"

GGUF_MAGIC   = 0x46554747   # "GGUF"
GGUF_VERSION = 3

# GGUF KV value types
GGUF_TYPE = {
    0: "uint8",  1: "int8",   2: "uint16", 3: "int16",
    4: "uint32", 5: "int32",  6: "float32",7: "bool",
    8: "string", 9: "array",
    10: "uint64",11: "int64", 12: "float64",
}
GGUF_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h",
    4: "I", 5: "i", 6: "f", 7: "?",
    10: "Q", 11: "q", 12: "d",
}


def read_str(data, pos):
    length = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    s = data[pos:pos+length].decode("utf-8")
    pos += length
    return s, pos


def write_str(s):
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def read_value(data, pos, vtype):
    if vtype == 8:  # string
        return read_str(data, pos)
    elif vtype == 9:  # array
        elem_type = struct.unpack_from("<I", data, pos)[0]; pos += 4
        count     = struct.unpack_from("<Q", data, pos)[0]; pos += 8
        items = []
        for _ in range(count):
            item, pos = read_value(data, pos, elem_type)
            items.append(item)
        return (elem_type, items), pos
    elif vtype in GGUF_FMT:
        fmt  = GGUF_FMT[vtype]
        size = struct.calcsize(fmt)
        val  = struct.unpack_from("<" + fmt, data, pos)[0]
        return val, pos + size
    else:
        raise ValueError(f"Unknown type {vtype}")


def write_value(vtype, val):
    if vtype == 8:
        return write_str(val)
    elif vtype == 9:
        elem_type, items = val
        out = struct.pack("<IQ", elem_type, len(items))
        for item in items:
            out += write_value(elem_type, item)
        return out
    elif vtype in GGUF_FMT:
        return struct.pack("<" + GGUF_FMT[vtype], val)
    else:
        raise ValueError(f"Unknown type {vtype}")


def patch():
    print(f"Reading {INPUT} ...")
    with open(INPUT, "rb") as f:
        data = bytearray(f.read())

    pos = 0

    # ── header ────────────────────────────────────────────────────────────────
    magic, version = struct.unpack_from("<II", data, pos); pos += 8
    assert magic == GGUF_MAGIC, "Not a GGUF file"
    assert version == GGUF_VERSION, f"Expected v3, got v{version}"

    n_tensors = struct.unpack_from("<Q", data, pos)[0]; pos += 8
    n_kv      = struct.unpack_from("<Q", data, pos)[0]; pos += 8

    print(f"  v{version}, {n_tensors} tensors, {n_kv} KV pairs")

    # ── collect KV pairs, renaming qwen35.* → qwen2.* ─────────────────────────
    kv_start = pos
    kv_data  = []
    renamed  = 0

    for _ in range(n_kv):
        key, pos      = read_str(data, pos)
        vtype         = struct.unpack_from("<I", data, pos)[0]; pos += 4
        val, pos      = read_value(data, pos, vtype)

        new_key = key.replace("qwen35.", "qwen2.")
        if new_key != key:
            renamed += 1
            print(f"  renaming: {key!r}  ->  {new_key!r}")
        kv_data.append((new_key, vtype, val))

    tensor_data_start = pos  # everything after KV is tensor info + data
    rest = data[tensor_data_start:]

    print(f"Renamed {renamed} KV keys.")

    # ── rebuild header + KV, keep rest untouched ───────────────────────────────
    new_header  = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, n_tensors, n_kv)
    new_kv      = bytearray()
    for (k, vtype, val) in kv_data:
        new_kv += write_str(k)
        new_kv += struct.pack("<I", vtype)
        new_kv += write_value(vtype, val)

    # Tensor-info alignment: the original format pads the tensor-info block to
    # GGUF_DEFAULT_ALIGNMENT (32 bytes) *starting from file byte 0*.
    # We must match the original offset so tensor data offsets (stored in the
    # tensor-info section) remain valid.
    old_kv_end  = tensor_data_start
    new_kv_end  = len(new_header) + len(new_kv)
    pad_needed  = old_kv_end - new_kv_end

    if pad_needed < 0:
        print(f"ERROR: new KV is {-pad_needed} bytes longer than original — "
              "cannot preserve tensor data offsets without full rebuild.")
        sys.exit(1)

    print(f"  padding {pad_needed} bytes to keep tensor offsets stable")
    new_kv += b"\x00" * pad_needed

    # Stream-write: header + KV, then copy the rest of the file in chunks
    # so we never need the whole 2 GB in memory at once.
    out_size = len(new_header) + len(new_kv) + len(rest)
    print(f"Writing {OUTPUT} ({out_size/1e9:.2f} GB) ...")
    with open(INPUT, "rb") as src, open(OUTPUT, "wb") as dst:
        dst.write(new_header)
        dst.write(new_kv)
        # Skip the original header + KV in src and copy everything after that
        src.seek(tensor_data_start)
        chunk = 1 << 22  # 4 MB chunks
        while True:
            block = src.read(chunk)
            if not block:
                break
            dst.write(block)
    print("Done.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    patch()
