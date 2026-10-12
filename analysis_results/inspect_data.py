
import json
import os

files = [
    r"d:\My\University\NCKH\Code\ViVQAv2\data\vivqa_v2\vivqa_v2_test.json",
    r"d:\My\University\NCKH\Code\ViVQAv2\saved_models\saaa_region_x152++_faster_rcnn_vivqav2\test_results.json"
]

output_file = r"d:\My\University\NCKH\Code\ViVQAv2\analysis_results\structure.txt"

with open(output_file, 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f"--- Inspecting {f} ---\n")
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    out.write(f"Type: dict, Keys: {list(data.keys())}\n")
                    for k, v in data.items():
                        if isinstance(v, list):
                            out.write(f"Key '{k}' is list of length {len(v)}\n")
                            if len(v) > 0:
                                out.write(f"Sample item from '{k}': {v[0]}\n")
                        else:
                            out.write(f"Key '{k}' value type: {type(v)}\n")
                elif isinstance(data, list):
                    out.write(f"Type: list, Length: {len(data)}\n")
                    if len(data) > 0:
                        out.write(f"Sample item: {data[0]}\n")
        except Exception as e:
            out.write(f"Error reading {f}: {e}\n")
        out.write("\n")
