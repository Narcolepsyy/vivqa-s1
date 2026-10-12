
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
# Use the best model path identified from model_comparison.csv
TEST_FILE = r"d:\My\University\NCKH\Code\ViVQAv2\data\vivqa_v2\vivqa_v2_test.json"
RESULT_FILE = r"d:\My\University\NCKH\Code\ViVQAv2\saved_models\mcan_region_x152++_faster_rcnn_vivqav2\test_results.json"
OUTPUT_DIR = r"d:\My\University\NCKH\Code\ViVQAv2\analysis_results\best_model_analysis"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
print("Loading data...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open(RESULT_FILE, 'r', encoding='utf-8') as f:
    results_data = json.load(f)

# 2. Process Annotations
qa_map = {}
for item in test_data['annotations']:
    qid = item['id']
    qa_map[qid] = {
        'question': item['question'],
        'gt_answer': item['answers']
    }

# 3. Process Results
records = []
for res in results_data['results']:
    try:
        qid_list = res.get('id', [])
        if not qid_list: continue
        qid = qid_list[0]
        
        gens = res.get('gens', {})
        pred_ans = str(list(gens.values())[0]) if gens else ""
        
        q_info = qa_map.get(qid)
        if q_info:
            records.append({
                'question_id': qid,
                'question': q_info['question'],
                'ground_truth': q_info['gt_answer'],
                'prediction': pred_ans,
                'is_correct': pred_ans.lower().strip() == q_info['gt_answer'].lower().strip(),
                'q_len': len(q_info['question'].split()),
                'a_len': len(q_info['gt_answer'].split())
            })
    except Exception as e:
        print(f"Error processing qid {res.get('id')}: {e}")

df = pd.DataFrame(records)

# 4. Categorize Question (Vietnamere 5W1H)
def categorize(q):
    q = q.lower()
    if any(x in q for x in ['tại sao', 'vì sao']): return 'Why'
    if any(x in q for x in ['như thế nào', 'ra sao']): return 'How'
    if any(x in q for x in ['ở đâu', 'đâu']): return 'Where'
    if any(x in q for x in ['ai']): return 'Who'
    if any(x in q for x in ['bao nhiêu', 'mấy']): return 'Count'
    if any(x in q for x in ['cái gì', 'gì']): return 'What'
    if any(x in q for x in ['khi nào', 'bao giờ']): return 'When'
    return 'Other'

df['category'] = df['question'].apply(categorize)

# 5. Analysis - Accuracy by Category
cat_stats = df.groupby('category').agg(count=('is_correct', 'count'), accuracy=('is_correct', 'mean')).sort_values('accuracy', ascending=False)

# 6. Analysis - Accuracy by Question Length
len_bins = [0, 5, 10, 15, 20, 100]
len_labels = ['1-5', '6-10', '11-15', '16-20', '20+']
df['q_len_grp'] = pd.cut(df['q_len'], bins=len_bins, labels=len_labels)
len_stats = df.groupby('q_len_grp').agg(count=('is_correct', 'count'), accuracy=('is_correct', 'mean'))

# 7. Visualization
sns.set_theme(style="whitegrid")

# Plot 1: Accuracy by Type
plt.figure(figsize=(10, 6))
sns.barplot(x=cat_stats.index, y=cat_stats['accuracy'], palette='viridis')
plt.title('Accuracy by Question Type (Best Model)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_type.png'))

# Plot 2: Accuracy by length
plt.figure(figsize=(10, 6))
sns.barplot(x=len_stats.index, y=len_stats['accuracy'], palette='magma')
plt.title('Accuracy by Question Length (Words)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_length.png'))

# 8. Error analysis
wrong_df = df[~df['is_correct']]
top_wrong = wrong_df.groupby(['ground_truth', 'prediction']).size().reset_index(name='count').sort_values('count', ascending=False).head(20)

# Save
cat_stats.to_csv(os.path.join(OUTPUT_DIR, 'cat_stats.csv'))
len_stats.to_csv(os.path.join(OUTPUT_DIR, 'len_stats.csv'))
top_wrong.to_csv(os.path.join(OUTPUT_DIR, 'top_errors.csv'), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, 'full_analysis.csv'), index=False)

print(f"Analysis complete. Results in {OUTPUT_DIR}")
print(f"Overall Accuracy: {df['is_correct'].mean():.4f}")
