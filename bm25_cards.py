import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import time
import csv
import os
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from taxonomy.v3 import taxonomy
from logger import logger


def dcg_at_k(relevance_scores, k):
    relevance_scores = relevance_scores[:k]
    return sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

def ndcg_at_k(retrieved_binary, relevant_count, k):
    if relevant_count == 0:
        return 0.0
    ideal_relevance = [1] * min(relevant_count, k) + [0] * max(0, k - relevant_count)
    dcg = dcg_at_k(retrieved_binary, k)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def get_ap_score(narrative_id, claims, hits, total_relevant):
    y_true_for_ap = [1 if claim == narrative_id else 0 for claim in claims]
    y_scores_for_ap = [0] * len(claims)
    for hit in hits:
        y_scores_for_ap[hit['corpus_id']] = hit['score']

    ap = average_precision_score(y_true_for_ap, y_scores_for_ap) if total_relevant > 0 else 0.0
    return ap


csv_filename = 'retrieval_cards_bm25.csv'
data_path = "./data/cards/training/test.csv"

training_data = pd.read_csv(data_path)
claims = training_data['claim'].tolist()
texts = training_data['text'].tolist()

total_docs = len(texts)
relevant_counts = Counter(claims)
lowest_count = min(relevant_counts.values())
Ks = [1, 3, 10, 50, 100, 500, 1000, 1500, 2000, 2500, lowest_count]

logger.info("Tokenizing corpus for BM25...")
tokenized_corpus = [doc.split(" ") for doc in texts]

logger.info("Creating BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)
logger.info("BM25 index created.")

all_rows = []
final_results = {}
model_name = "bm25"

logger.info(f"Starting evaluation for model: {model_name}")
final_results[model_name] = {}
start_time = time.time()

overall_metrics = {
    'precision': defaultdict(list),
    'recall_at_k': defaultdict(list),
'f1': defaultdict(list),
    'ndcg': defaultdict(list),
    'ap': defaultdict(list),
    'r_precision': defaultdict(list),
    'recall_normalized_at_100': defaultdict(list),
    'balanced_accuracy': defaultdict(list),
    'mrr': [],
    'roc_auc': [],
}

fails = ["1_5", "1_8", "2_2", "2_4", "2_5", "3_4", "3_5", "3_6", "4_3", "5_3"]
narratives_processed = 0
r_precision_weighted_pairs = []

model_overall = {'model_name': model_name}
model_per_narr = {}

for narrative_id, narrative in tqdm(taxonomy.items(), desc=f"Processing {model_name}"):
    if narrative_id in fails:
        continue
    narrative = narrative[0]
    narratives_processed += 1
    
    # --- BM25 Retrieval ---
    tokenized_query = narrative.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Create the 'hits' structure similar to the original script
    top_k_retrieval = max(Ks) * 2
    ranked_indices = np.argsort(doc_scores)[::-1]
    hits = [{'corpus_id': i, 'score': doc_scores[i]} for i in ranked_indices[:top_k_retrieval]]
    
    retrieved_claims = [claims[hit['corpus_id']] for hit in hits]
    correct_hits = [1 if retrieved_claim == narrative_id else 0 for retrieved_claim in retrieved_claims]

    total_relevant = relevant_counts[narrative_id]

    ap = get_ap_score(narrative_id, claims, hits, total_relevant)
    overall_metrics['ap'][narrative_id].append(ap)

    if total_relevant > 0:
        k_fixed = 100
        retrieved_relevant_at_100 = sum(correct_hits[:k_fixed])
        max_possible_at_100 = min(k_fixed, total_relevant)
        recall_normalized_at_100 = (
            retrieved_relevant_at_100 / max_possible_at_100 if max_possible_at_100 > 0 else 0.0
        )
        r_k = total_relevant
        r_precision = sum(correct_hits[:r_k]) / r_k if r_k > 0 else 0.0
        r_precision_weighted_pairs.append((r_precision * total_relevant, total_relevant))
    else:
        recall_normalized_at_100 = 0.0
        r_precision = 0.0
    
    overall_metrics['recall_normalized_at_100'][narrative_id].append(recall_normalized_at_100)
    overall_metrics['r_precision'][narrative_id].append(r_precision)

    key_prefix = f"{narrative_id}"
    final_results[model_name][key_prefix] = {
        "NormalizedRecall@100": recall_normalized_at_100,
        "R-Precision": r_precision,
    }

    for k in Ks:
        retrieved_k = correct_hits[:k]
        TP = sum(retrieved_k)
        FP = k - TP
        FN = max(total_relevant - TP, 0)
        TN = max(total_docs - total_relevant - FP, 0)

        precision_k = TP / k if k > 0 else 0.0
        recall_k = TP / total_relevant if total_relevant > 0 else 0.0
        f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

        ndcg_k = ndcg_at_k(retrieved_k, total_relevant, k)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        balanced_acc_k = 0.5 * (tpr + tnr)

        overall_metrics['precision'][k].append(precision_k)
        overall_metrics['recall_at_k'][k].append(recall_k)
        overall_metrics['f1'][k].append(f1_k)
        overall_metrics['ndcg'][k].append(ndcg_k)
        overall_metrics['balanced_accuracy'][k].append(balanced_acc_k)

        final_results[model_name][key_prefix][f"P@{k}"] = precision_k
        final_results[model_name][key_prefix][f"R@{k}"] = recall_k
        final_results[model_name][key_prefix][f"F1@{k}"] = f1_k
        final_results[model_name][key_prefix][f"NDCG@{k}"] = ndcg_k
        final_results[model_name][key_prefix][f"BalancedAcc@{k}"] = balanced_acc_k

        model_per_narr[f"{key_prefix}_P@{k}"] = precision_k
        model_per_narr[f"{key_prefix}_R@{k}"] = recall_k
        model_per_narr[f"{key_prefix}_F1@{k}"] = f1_k
        model_per_narr[f"{key_prefix}_NDCG@{k}"] = ndcg_k
        model_per_narr[f"{key_prefix}_BalancedAcc@{k}"] = balanced_acc_k

    try:
        first_rel_rank = next(i for i, x in enumerate(correct_hits) if x == 1)
        rr = 1.0 / (first_rel_rank + 1)
    except StopIteration:
        rr = 0.0
    overall_metrics['mrr'].append(rr)
    final_results[model_name][key_prefix]["MRR"] = rr
    model_per_narr[f"{key_prefix}_MRR"] = rr

    model_per_narr[f"{key_prefix}_AP"] = ap

    map_narrative = float(np.mean(overall_metrics['ap'][narrative_id]))
    model_per_narr[f"{narrative_id}_MAP"] = map_narrative


end_time = time.time()
elapsed_time = end_time - start_time

model_overall['elapsed_time'] = elapsed_time
model_overall['narratives_processed'] = narratives_processed

all_aps = [ap for narrative_aps in overall_metrics['ap'].values() for ap in narrative_aps]
model_overall['Overall_MAP'] = float(np.mean(all_aps)) if all_aps else 0.0

all_r_precs = [r for r_list in overall_metrics['r_precision'].values() for r in r_list]
model_overall['Overall_R-Precision'] = np.mean(all_r_precs) if all_r_precs else 0.0

total_tp_weighted = sum(tp for tp, _ in r_precision_weighted_pairs)
total_rel_weighted = sum(rel for _, rel in r_precision_weighted_pairs)
model_overall['Overall_Weighted_R-Precision'] = total_tp_weighted / total_rel_weighted if total_rel_weighted > 0 else 0.0

all_recall_norm_100 = [r for recalls in overall_metrics['recall_normalized_at_100'].values() for r in recalls]
model_overall['Overall_NormalizedRecall@100'] = float(np.mean(all_recall_norm_100)) if all_recall_norm_100 else 0.0

for k in Ks:
    p_list = overall_metrics['precision'][k]
    r_list = overall_metrics['recall_at_k'][k]
    f1_list = overall_metrics['f1'][k]
    ndcg_list = overall_metrics['ndcg'][k]
    bal_list = overall_metrics['balanced_accuracy'][k]

    model_overall[f"Overall_P@{k}"] = float(np.mean(p_list)) if p_list else 0.0
    model_overall[f"Overall_R@{k}"] = float(np.mean(r_list)) if r_list else 0.0
    model_overall[f"Overall_F1@{k}"] = float(np.mean(f1_list)) if f1_list else 0.0
    model_overall[f"Overall_NDCG@{k}"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    model_overall[f"Overall_BalancedAcc@{k}"] = float(np.mean(bal_list)) if bal_list else 0.0

model_overall['Overall_MRR'] = float(np.mean(overall_metrics['mrr'])) if overall_metrics['mrr'] else 0.0

if overall_metrics['roc_auc']:
    model_overall['Overall_ROC-AUC'] = float(np.mean(overall_metrics['roc_auc']))
else:
    model_overall['Overall_ROC-AUC'] = None

row_combined = {**{'model_name': model_name}, **model_overall, **model_per_narr}
all_rows.append(row_combined)

logger.info(f"{model_name}: Overall_R-Precision = {model_overall['Overall_R-Precision']:.4f}")
logger.info(f"{model_name}: Overall_Weighted_R-Precision = {model_overall['Overall_Weighted_R-Precision']:.4f}")
logger.info(f"{model_name}: Overall_MAP = {model_overall['Overall_MAP']:.4f}")


overall_prefixes = ("Overall_", "elapsed_time", "narratives_processed")
all_overall_keys = []
all_per_keys = []

for row in all_rows:
    for k in row.keys():
        if k == 'model_name':
            continue
        if k.startswith(overall_prefixes):
            if k not in all_overall_keys:
                all_overall_keys.append(k)
        else:
            if k not in all_per_keys:
                all_per_keys.append(k)

all_overall_keys = sorted(all_overall_keys)
all_per_keys = sorted(all_per_keys)

fieldnames = ['model_name'] + all_overall_keys + all_per_keys

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row.get(k, None) for k in fieldnames})

logger.info(f"\nResults successfully saved to {csv_filename}")
