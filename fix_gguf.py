import sys
from gguf import GGUFReader, GGUFWriter
from gguf.gguf_writer import GGUFWriter, TensorInfo

# Monkey-patch GGUFWriter to bypass the hard-coded unquantized-only check
def patched_add_tensor_info(self, name, shape, data_type, data_nbytes, raw_dtype=None):
    if not self.tensors:
        self.tensors.append({})
    self.tensors[-1][name] = TensorInfo(
        shape=shape,
        dtype=data_type,
        nbytes=data_nbytes
    )
GGUFWriter.add_tensor_info = patched_add_tensor_info

input_file = "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"
output_file = "graphrag-plus-plus-qwen2-4b-q3_k_m.gguf"

print(f"Reading {input_file}...")
reader = GGUFReader(input_file)

writer = GGUFWriter(output_file, "qwen2")

# Copy all KVs, replacing 'qwen35' with 'qwen2'
for key, field in reader.fields.items():
    if key == "general.architecture":
        continue  # GGUFWriter adds this automatically based on arch argument
        
    new_key = key.replace("qwen35", "qwen2")
    
    val = field.parts[field.data[-1]]
    
    # gguf library handles different types:
    if field.types[0].name == 'UINT32':
        writer.add_uint32(new_key, val[0])
    elif field.types[0].name == 'FLOAT32':
        writer.add_float32(new_key, val[0])
    elif field.types[0].name == 'STRING':
        # Need to convert memmap to bytes, then decode
        s_val = val.tobytes().decode('utf-8') if hasattr(val, 'tobytes') else str(val)
        writer.add_string(new_key, s_val)
    elif field.types[0].name == 'ARRAY':
        # val is already a list or numpy array, add array
        writer.add_array(new_key, val)
    else:
        # Fallback for other standard types
        print(f"Skipping unhandled type {field.types[0].name} for {key}")

# Copy all tensors
print("Copying tensors...")
for tensor in reader.tensors:
    # tensor.tensor_type is an Enum/int representing GGML_TYPE (e.g. 11 for Q3_K)
    # The original copy routine would crash here by default without our monkey patch
    writer.add_tensor_info(tensor.name, tensor.shape, tensor.tensor_type, len(tensor.data))
    # add_tensor_info registers it, now we just append to the raw data buffer
    # GGUFWriter accumulates tensor data in self.tensor_data
    # Actually wait, let's just use add_tensor() if Data isn't exposed...
    # I'll just write directly:
    writer.tensors[-1][tensor.name].data = tensor.data


print(f"Writing to {output_file}...")
writer.write_header_to_file()
writer.write_kv_data_to_file()

# write_tensors_to_file() relies on raw tensor arrays which we skirted, 
# so we manually write the tensor data block by block, paying attention to padding!
writer.fout.write(b"\x00" * writer._get_padding_size(writer.fout.tell(), writer.alignment))
for tensor_dict in writer.tensors:
    for name, t in tensor_dict.items():
        writer.fout.write(b"\x00" * writer._get_padding_size(writer.fout.tell(), writer.alignment))
        t.offset = writer.fout.tell() - writer.data_offset
        writer.fout.write(t.data)

writer.close()

print("Done! Replaced qwen35 with qwen2 in GGUF.")
