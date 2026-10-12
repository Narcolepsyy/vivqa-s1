
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# # Search for all test_results.json files in saved_models
# SEARCH_DIR = r"d:\My\University\NCKH\Code\ViVQAv2\saved_models"
OUTPUT_DIR = r"d:\My\University\NCKH\Code\ViVQAv2\analysis_results"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# result_files = glob.glob(os.path.join(SEARCH_DIR, "**", "test_results.json"), recursive=True)

# print(f"Found {len(result_files)} result files.")

# data = []

# for f in result_files:
#     model_name = os.path.basename(os.path.dirname(f))
#     try:
#         with open(f, 'r', encoding='utf-8') as file:
#             res = json.load(file)
            
#             # Extract scalar metrics
#             # Some files might have metrics in a dict or top level
#             # Based on previous inspection: top level keys 'Accuracy', 'F1', 'BLEU', etc.
            
#             # Helper to get float value safely
#             def get_val(key):
#                 val = res.get(key, None)
#                 if isinstance(val, (int, float)):
#                     return val
#                 if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
#                     # Sometimes BLEU might be a list of 4 values, take the last/best or average? 
#                     # Usually BLEU-4 is the standard. If list, assume BLEU-1 to 4. 
#                     # Let's take the last one (BLEU-4) if it's BLEU.
#                     if key == 'BLEU': 
#                          return val[-1] 
#                     return val[0]
#                 return None

#             entry = {
#                 'Model': model_name,
#                 'Accuracy': get_val('Accuracy'),
#                 'F1': get_val('F1'),
#                 'Precision': get_val('Precision'),
#                 'Recall': get_val('Recall'),
#                 'CIDEr': get_val('CIDEr'),
#                 'BLEU-4': get_val('BLEU'),
#                 'ROUGE': get_val('ROUGE'),
#                 'Path': f
#             }
#             data.append(entry)
#     except Exception as e:
#         print(f"Error reading {model_name}: {e}")


df = pd.read_csv(r"d:\My\University\NCKH\Code\ViVQAv2\analysis_results\model_comparison.csv")

# Filter out rows with no Accuracy (failed runs or different format)
df = df.dropna(subset=['Accuracy']).sort_values('Accuracy', ascending=False)

print("\n--- Model Comparison (Sorted by Accuracy) ---")
print(df[['Model', 'Accuracy', 'F1', 'CIDEr', 'BLEU-4']].to_string(index=False))

# Save to CSV
csv_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved comparison to {csv_path}")

# Visualization
if not df.empty:
    plt.figure(figsize=(12, 8))
    # Top 10 models
    top_df = df.head(10)
    sns.barplot(x='Accuracy', y='Model', data=top_df, palette='viridis')
    plt.title('Models by Accuracy')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    
    # Add text labels
    for i, v in enumerate(top_df['Accuracy']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")
else:
    print("No valid result data found to visualize.")
