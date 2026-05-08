import os
import struct

input_file = "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"
output_file = "graphrag-plus-plus-qwen2-4b-q3_k_m.gguf"

print(f"Reading {input_file}...")
with open(input_file, 'rb') as f:
    data = f.read()

# We need to find the KV length fields and the strings.
# GGUF KVs are sequential. Let's just find `qwen35.X` and replace.
# But wait, looking for `qwen35` preceded by its length as uint64 is easy!
# The string "qwen35." is 7 bytes.
# "tokenizer.ggml.pre" = "qwen35" (6 bytes)

new_data = bytearray()
idx = 0

def find_all(sub):
    res = []
    i = data.find(sub)
    while i != -1:
        res.append(i)
        i = data.find(sub, i+1)
    return res

import re
# We just do a smart search and replace using regex on the bytes? No, we have to adjust the uint64 lengths!

print("Parsing structure to safely replace qwen35 with qwen2...")

from gguf import GGUFReader

reader = GGUFReader(input_file)
# We know the keys that need changing. We can read their exact byte offsets!
# Actually, GGUFReader gives us `field.offset` which is the start of the key structure!
# `offset` points to: length of key (8b) + key bytes + ...
offsets_to_modify_key = []
offsets_to_modify_val = []

for key, field in reader.fields.items():
    if "qwen35" in key:
        offsets_to_modify_key.append((field.offset, key))
    
    # Check if the string VALUE is qwen35
    if field.types[0].name == 'STRING':
        val = field.parts[field.data[-1]].tobytes()
        if b"qwen35" in val:
            # We need the offset of the string length for the value.
            # Parts[-2] is the string length uint64.
            val_len_offset = field.offset # We'd have to calculate exactly where parts[-2] is
            # Luckily part objects don't store their file offset in python, but we can search for it!

print("Patching headers...")

# Alternative approach: use GGUFReader to find where `qwen35` is, but it's simpler to just do:
# 1. Identify all places where `qwen35` occurs.
# 2. Check if the 8 bytes prior equal the string length.
# If so, it's a GGUF string! Reduce length by 1, and strip the '5'.

patched_data = bytearray(data)
replacements_made = 0

# Find all "qwen35"
pos = 0
while True:
    pos = patched_data.find(b"qwen35", pos)
    if pos == -1:
        break
        
    # Check the 8 bytes before
    # For a key, it might be exactly 8 bytes before, or if it's "qwen35.block", the length is 18
    # Oh wait! In GGUF, string length is uint64 right before the string!
    # Let's read 8 bytes before pos:
    str_len = struct.unpack('<Q', patched_data[pos-8:pos])[0]
    
    if str_len < 1000: # Safe sanity check for string length
        # Read the actual string
        s = patched_data[pos:pos+str_len]
        if s.startswith(b"qwen35"):
            # It's a match!
            print(f"Found string at {pos}: {s}")
            
            # Reduce length by 1
            new_len = str_len - 1
            patched_data[pos-8:pos] = struct.pack('<Q', new_len)
            
            # Change '3' to '2' and delete the '5'
            patched_data[pos+4] = ord('2')
            del patched_data[pos+5:pos+6]
            
            replacements_made += 1
            # We don't advance pos because we shifted data
            continue
            
    pos += 1

print(f"Made {replacements_made} replacements.")

# Because we shifted the data, the tensor data alignment is now broken.
# GGUF tensor data starts after the KV store, aligned to 32 bytes.
# We removed `replacements_made` bytes. We need to add them back as padding before the first tensor.
# Or just pad the ALIGNMENT padding area?
# Yes, there's padding between the last KV and the first tensor.
# But wait, we also have to update the tensor offsets so they don't break?!
# NO! Tensor offsets in GGUF are relative to the *end* of the KV section (tensor_data_offset).
# If we shift the entire file by deleting bytes in the KV section, the `tensor_data_offset` shifts automatically if we just pad the gap!!
# Actually! `data_offset` is recorded implicitly. No, `tensor_data` just starts at `alignment * ceil(header_size / alignment)`.
# If our header size shrank by 28 bytes, but remains in the same alignment block, the padding just expands by 28 bytes!
# So all tensor bytes seamlessly fall into the exact same absolute file offset as before, NO tensor offsets need to be changed!

# Let's find the original `data_offset`:
alignment = reader.alignment
# The original data_offset was:
orig_data_offset = reader.data_offset

new_data_offset = orig_data_offset # We want it to be exactly the same absolute file position

# Let's see where the new KV section ends:
# We just write the patched_data up to the end of the KV store (which we need to find).
# But wait, `patched_data` still has ALL the tensor data attached!
# If we just deleted bytes, the tensor data SHIFTED LEFT by 28 bytes.
# To put it back to its original absolute file position, we must INSERT 28 bytes of padding right before the tensor data starts!

print(f"Original data offset: {orig_data_offset}")
# The new end of the header (which is where padding starts) is just its original end minus replacements_made.
# Actually, the original data_offset minus replacements_made is where the original padding ended?!
# Wait, the tensor data started exactly at `orig_data_offset`.
# In `patched_data`, the tensor data now starts at `orig_data_offset - replacements_made`.
# We need to insert `replacements_made` bytes of \x00 at `orig_data_offset - replacements_made`.

insert_pos = orig_data_offset - replacements_made
patched_data[insert_pos:insert_pos] = b'\x00' * replacements_made

print(f"Inserted {replacements_made} bytes of padding at {insert_pos}.")

# Verify that the tensor data starts at the exactly same offset
# (The size of patched_data should now match original data)

assert len(patched_data) == len(data), "Length mismatch!"

with open(output_file, 'wb') as f:
    f.write(patched_data)
    
print("Successfully wrote binary patched GGUF without touching unquantized tensors!")
