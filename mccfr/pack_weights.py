import json
import zlib
import base64
import os

def pack_weights():
    input_file = 'mccfr/bot_weights.json'
    output_file = 'packed_weights.py'
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'. Ensure your training script has finished and exported the weights.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        json_data = f.read()
        
    original_size_mb = len(json_data) / (1024 * 1024)
    print(f"Original JSON size: {original_size_mb:.2f} MB")
    
    # Compress the raw JSON string using zlib at maximum compression level (9)
    print("Compressing...")
    compressed_data = zlib.compress(json_data.encode('utf-8'), level=9)
    
    # Encode the binary data to a safe Base64 ascii string
    b64_string = base64.b64encode(compressed_data).decode('ascii')
    
    compressed_size_mb = len(b64_string) / (1024 * 1024)
    print(f"Compressed Base64 size: {compressed_size_mb:.2f} MB")
    print(f"Compression ratio: {(compressed_size_mb / original_size_mb) * 100:.1f}%")
    
    # Write the string into a Python variable format
    with open(output_file, 'w') as f:
        f.write('COMPRESSED_WEIGHTS = \\\n')
        f.write(f'"{b64_string}"\n')
        
    print(f"\nSuccess! Open '{output_file}' and copy the COMPRESSED_WEIGHTS variable to the top of your bot.py file.")

if __name__ == '__main__':
    pack_weights()