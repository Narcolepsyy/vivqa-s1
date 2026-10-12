
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
TEST_FILE = r"d:\My\University\NCKH\Code\ViVQAv2\data\vivqa_v2\vivqa_v2_test.json"
RESULT_FILE = r"d:\My\University\NCKH\Code\ViVQAv2\saved_models\saaa_region_x152++_faster_rcnn_vivqav2\test_results.json"
OUTPUT_DIR = r"d:\My\University\NCKH\Code\ViVQAv2\analysis_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
print("Loading data...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open(RESULT_FILE, 'r', encoding='utf-8') as f:
    results_data = json.load(f)

# 2. Process Annotations (Ground Truth & Questions)
# specific to ViVQAv2 format seen: {'id': ..., 'image_id': ..., 'question': ..., 'answers': ...}
qa_map = {}
for item in test_data['annotations']:
    # annotations use 'id' as question_id
    qid = item['id']
    qa_map[qid] = {
        'question': item['question'],
        'gt_answer': item['answers'] # strict string match
    }

# 3. Process Results
# format seen: {'id': [8117], 'filename': ['...'], 'gens': {'0_0': '2'}, 'gts': {'0_0': '4'}}
records = []
for res in results_data['results']:
    # Extract ID
    try:
        # id is a list, e.g. [8117]
        qid_list = res.get('id', [])
        if not qid_list:
            continue
        qid = qid_list[0]
        
        # Extract prediction
        # gens is dict {'0_0': 'val'}
        gens = res.get('gens', {})
        pred_ans = str(list(gens.values())[0]) if gens else ""
        
        # Get Question info
        q_info = qa_map.get(qid)
        
        if q_info:
            records.append({
                'question_id': qid,
                'question': q_info['question'],
                'ground_truth': q_info['gt_answer'],
                'prediction': pred_ans,
                'is_correct': pred_ans.lower().strip() == q_info['gt_answer'].lower().strip()
            })
    except Exception as e:
        print(f"Error processing result item: {res}, Error: {e}")

df = pd.DataFrame(records)
print(f"Processed {len(df)} records.")

# 4. Define Question Categories (Vietnamese)
def categorize_question(q):
    q = q.lower()
    if 'tại sao' in q:
        return 'Why'
    elif 'như thế nào' in q or 'ra sao' in q:
        return 'How'
    elif 'cái gì' in q or 'gì' in q:
        return 'What'
    elif 'ở đâu' in q or 'đâu' in q:
        return 'Where'
    elif 'khi nào' in q or 'bao giờ' in q:
        return 'When'
    elif 'ai' in q:
        return 'Who'
    elif 'bao nhiêu' in q or 'mấy' in q:
        return 'Count'
    elif 'có' in q and 'không' in q: 
        # overly simple heuristic for Yes/No in Vietnamese "Có ... không?"
        return 'Yes/No'
    else:
        return 'Other'

df['category'] = df['question'].apply(categorize_question)

# 5. Analysis
overall_acc = df['is_correct'].mean()
print(f"Overall Accuracy (calculated): {overall_acc:.4f}")
print(f"Reported Accuracy in JSON: {results_data.get('Accuracy', 'N/A')}")

# Group by category
cat_stats = df.groupby('category').agg(
    count=('question_id', 'count'),
    accuracy=('is_correct', 'mean')
).sort_values('accuracy', ascending=False)

print("\nAccuracy by Question Type:")
print(cat_stats)

# Save stats to CSV
cat_stats.to_csv(os.path.join(OUTPUT_DIR, 'accuracy_by_type.csv'))
df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_details.csv'), index=False, encoding='utf-8-sig')

# 6. Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=cat_stats.index, y=cat_stats['accuracy'], palette='viridis')
plt.title('Accuracy by Question Type')
plt.ylabel('Accuracy')
plt.xlabel('Question Type')
plt.ylim(0, 1.0)
for index, row in enumerate(cat_stats.itertuples()):
    plt.text(index, row.accuracy + 0.01, f'{row.accuracy:.2f}', ha='center', color='black')

plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_type.png'))
print(f"Plots saved to {OUTPUT_DIR}")

# 7. Error Analysis (Top wrong predictions)
wrong_df = df[~df['is_correct']]
if not wrong_df.empty:
    top_wrong = wrong_df.groupby(['ground_truth', 'prediction']).size().reset_index(name='count').sort_values('count', ascending=False).head(10)
    print("\nTop 10 Common Errors (Ground Truth -> Prediction):")
    print(top_wrong)
    top_wrong.to_csv(os.path.join(OUTPUT_DIR, 'top_errors.csv'), index=False)
