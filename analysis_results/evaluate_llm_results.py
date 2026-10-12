
import json
import os
import sys
import numpy as np

# Add project root to path to import evaluation module
project_root = r"d:\My\University\NCKH\Code\ViVQAv2"
if project_root not in sys.path:
    sys.path.append(project_root)

from evaluation import compute_scores

GT_FILE = os.path.join(project_root, "data", "vivqa_v2", "vivqa_v2_test.json")
PRED_DIR = os.path.join(project_root, "LLMs")
FILES_TO_EVAL = [
    "lavys_predictions.json",
    "lavys_predictions_1760187507.json",
    "results_llava_zero_shot_2025-09-22_23-07-14.json"
]

def load_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # In ViVQA v2 test set, annotations are usually under a key or at the end
    # Based on inspection, it's a heterogenous file or has specific structure
    # Let's try to find 'id' and 'answers' in any list or top level
    
    gts = {}
    
    # If it's the COCO-like structure we saw:
    if isinstance(data, dict):
        # We saw 'images' array. Let's look for others.
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if 'id' in item and 'answers' in item:
                        qid = item['id']
                        ans = item['answers']
                        if isinstance(ans, str):
                            ans = [ans.strip()]
                        gts[qid] = ans
    elif isinstance(data, list):
        for item in data:
            if 'id' in item and 'answers' in item:
                qid = item['id']
                ans = item['answers']
                if isinstance(ans, str):
                    ans = [ans.strip()]
                gts[qid] = ans
                
    return gts

def load_pred(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preds = {}
    for item in data:
        qid = item['id']
        # Handle different key names
        if 'predicted_answer' in item:
            ans = item['predicted_answer']
        elif 'pred_answer' in item:
            ans = item['pred_answer']
        else:
            ans = ""
            
        if ans is None:
            ans = ""
            
        # Basic cleanup: remove prefix like ": " often seen in LLM outputs
        ans = ans.strip()
        if ans.startswith(": "):
            ans = ans[2:].strip()
            
        preds[qid] = [ans]
    return preds

def main():
    print("Loading ground truth...")
    gts = load_gt(GT_FILE)
    print(f"Loaded {len(gts)} ground truth annotations.")
    
    results = []
    
    for filename in FILES_TO_EVAL:
        path = os.path.join(PRED_DIR, filename)
        print(f"\nEvaluating {filename}...")
        
        preds = load_pred(path)
        print(f"Loaded {len(preds)} predictions.")
        
        # Syncing IDs
        common_ids = set(gts.keys()) & set(preds.keys())
        print(f"Common IDs: {len(common_ids)}")
        
        sync_gts = {qid: gts[qid] for qid in common_ids}
        sync_preds = {qid: preds[qid] for qid in common_ids}
        
        if not common_ids:
            print("No common IDs found. Skipping.")
            continue
            
        scores, _ = compute_scores(sync_gts, sync_preds)
        
        # Display scores
        print(f"Scores for {filename}:")
        for metric, score in scores.items():
            if isinstance(score, list):
                # BLEU returns a list of 4 scores
                print(f"  {metric}: {score}")
            else:
                print(f"  {metric}: {score:.4f}")
        
        results.append({
            'file': filename,
            'scores': scores
        })

    # Final Summary Table
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'CIDEr', 'BLEU', 'ROUGE']
    header = "File | " + " | ".join(metrics)
    print(header)
    print("-" * len(header))
    
    for res in results:
        s = res['scores']
        row = f"{res['file']} | "
        row += f"{s.get('Accuracy', 0):.4f} | "
        row += f"{s.get('F1', 0):.4f} | "
        row += f"{s.get('Precision', 0):.4f} | "
        row += f"{s.get('Recall', 0):.4f} | "
        row += f"{s.get('CIDEr', 0):.4f} | "
        bl = s.get('BLEU', [0,0,0,0])
        if isinstance(bl, list): bl = bl[-1] # Use BLEU-4
        row += f"{bl:.4f} | "
        row += f"{s.get('ROUGE', 0):.4f}"
        print(row)

if __name__ == "__main__":
    main()
