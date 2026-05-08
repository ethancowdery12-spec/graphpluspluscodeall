"""
fix_gguf_v2.py — Minimal fix: rewrite GGUF with general.architecture="qwen35".

The original file has general.architecture="qwen2" (wrong) but all KV keys use
the correct "qwen35.*" prefix and all tensor names match what llama.cpp's
qwen35 handler (b9066+) expects.  Only fix needed is the architecture string.
"""
import os
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

INPUT  = "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"
OUTPUT = "graphrag-plus-plus-qwen35-4b-q3_k_m-fixed.gguf"


def main():
    print(f"Reading {INPUT} ...")
    reader = GGUFReader(INPUT)
    print(f"  {len(reader.fields)} KV fields, {len(reader.tensors)} tensors")

    writer = GGUFWriter(OUTPUT, "qwen35")  # writer adds general.architecture for us

    for orig_key, field in reader.fields.items():
        if orig_key == "general.architecture":
            continue  # handled by GGUFWriter constructor

        ftypes = field.types
        if not ftypes:
            continue
        outer = ftypes[0]

        if outer == GGUFValueType.STRING:
            writer.add_string(orig_key, bytes(field.parts[field.data[0]]).decode("utf-8"))
        elif outer == GGUFValueType.UINT8:
            writer.add_uint8(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.INT8:
            writer.add_int8(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.UINT16:
            writer.add_uint16(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.INT16:
            writer.add_int16(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.UINT32:
            writer.add_uint32(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.INT32:
            writer.add_int32(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.UINT64:
            writer.add_uint64(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.INT64:
            writer.add_int64(orig_key, int(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.FLOAT32:
            writer.add_float32(orig_key, float(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.FLOAT64:
            writer.add_float64(orig_key, float(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.BOOL:
            writer.add_bool(orig_key, bool(field.parts[field.data[0]][0]))
        elif outer == GGUFValueType.ARRAY:
            elem_type = field.types[1]
            items = []
            for idx in field.data:
                v = field.parts[idx]
                if elem_type == GGUFValueType.STRING:
                    items.append(bytes(v).decode("utf-8"))
                else:
                    raw = v[0] if hasattr(v, "__len__") and len(v) == 1 else v
                    if hasattr(raw, "item"):
                        raw = raw.item()
                    elif hasattr(raw, "tolist"):
                        raw = raw.tolist()
                    items.append(raw)
            writer.add_array(orig_key, items)
        else:
            print(f"  skipping unhandled type {outer.name} for {orig_key}")

    print(f"  copying {len(reader.tensors)} tensors verbatim ...")
    for t in reader.tensors:
        writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)

    print(f"Writing {OUTPUT} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
