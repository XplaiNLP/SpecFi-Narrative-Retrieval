import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import time
import csv
import pickle
import os
import openai


import numpy as np
from collections import defaultdict, Counter
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

def get_detailed_instruct(query: str) -> str:
    task_description = 'Given a narrative description as a query, retrieve passages that serve this narrative; can be entailed from the narrative; can be aligned logically with the narrative'
    #task_description = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {task_description}\nQuery: {query}'

def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=1000):
    client = openai.OpenAI()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with OpenAI"):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return torch.tensor(embeddings)


model_names = [
  # "openai_large",
  #'thenlper/gte-base',
  #'thenlper/gte-large',
   # 'intfloat/multilingual-e5-large-instruct',
    #'Alibaba-NLP/gte-multilingual-base',
  #   'Qwen/Qwen3-Embedding-0.6B',
   'Qwen/Qwen3-Embedding-4B',
    #'intfloat/e5-mistral-7b-instruct',
   #  'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
   #  'Alibaba-NLP/gte-Qwen2-7B-instruct',
]


openai.api_key = os.getenv("OPENAI_API_KEY")
Ks = [1, 3, 10, 50, 100, 500, 1000, 1500, 2000, 2500]

csv_filename = 'rertrieval_cards_all_prompt2.csv'
data_path = "./data/cards/training/test.csv"

training_data = pd.read_csv(data_path)
claims = training_data['claim'].tolist()
texts = training_data['text'].tolist()

total_docs = len(texts)
relevant_counts = Counter(claims)
lowest_count = min(relevant_counts.values())
Ks = [1, 3, 10, 50, 100, 500, 1000, 1500, 2000, 2500, lowest_count]
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 1


all_rows = []
final_results = {}
for model_name in model_names:
    if model_name == "openai_large":
        logger.info(f"Model name: {model_name}")
        final_results[model_name] = {}
        start_time = time.time()

        embeddings_path = "data/training/text_talone_embeddings_openai.pkl"
        if os.path.exists(embeddings_path):
            logger.info(f"Loading embeddings from {embeddings_path}")
            with open(embeddings_path, "rb") as f:
                text_embeddings = pickle.load(f)
        else:
            logger.info("Generating embeddings with OpenAI API...")
            text_embeddings = get_openai_embeddings(texts)
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            with open(embeddings_path, "wb") as f:
                pickle.dump(text_embeddings, f)
        
        text_embeddings = text_embeddings.to(device)

        overall_metrics = {
            'precision': defaultdict(list),
            'ndcg': defaultdict(list),
            'ap': defaultdict(list),
            'recall_at_k': defaultdict(list),
            'f1': defaultdict(list),
            'r_precision': defaultdict(list),
            'recall_normalized_at_100': defaultdict(list),
            'balanced_accuracy': defaultdict(list),
            'mrr': [],
            'roc_auc': [],
        }
        fails = ["1_5", "1_8", "2_2", "2_4", "2_5", "3_4", "3_5", "3_6", "4_3", "5_3"]
        narratives_processed = 0
        r_precision_scores = []
        r_precision_weighted_pairs = []

        model_overall = {'model_name': model_name}
        model_per_narr = {}

        for narrative_id, narrative in tqdm(taxonomy.items(), desc=f"Processing {model_name}"):
            if narrative_id in fails:
                continue

            narratives_processed += 1
            ap_scores = []
            r_precision_scores = []

            if any(s in model_name for s in ["Qwen", "intfloat", "instruct", "nvidia"]):
                subnarrative_input = get_detailed_instruct(narrative)
            else:
                subnarrative_input = narrative            
            
            query_embedding = get_openai_embeddings(subnarrative_input)[0].to(device)

            hits = util.semantic_search(query_embedding, text_embeddings, top_k=max(Ks) * 2)[0]
            retrieved_claims = [claims[hit['corpus_id']] for hit in hits]
            correct_hits = [1 if retrieved_claim == narrative_id else 0 for retrieved_claim in retrieved_claims]

            total_relevant = relevant_counts[narrative_id]

            ap = get_ap_score(narrative_id, claims, hits, total_relevant)
            ap_scores.append(ap)
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
                print(narrative_id, r_precision, r_precision * total_relevant, total_relevant)
            else:
                recall_normalized_at_100 = 0.0
                r_precision = 0.0
            
            r_precision_scores.append(r_precision)
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
                ndcg_k = ndcg_at_k(retrieved_k, total_relevant, k)
                recall_k = TP / total_relevant if total_relevant > 0 else 0.0
                f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

                tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                balanced_acc_k = 0.5 * (tpr + tnr)

                overall_metrics['precision'][k].append(precision_k)
                overall_metrics['ndcg'][k].append(ndcg_k)
                overall_metrics['balanced_accuracy'][k].append(balanced_acc_k)
                overall_metrics['recall_at_k'][k].append(recall_k)
                overall_metrics['f1'][k].append(f1_k)

                final_results[model_name][key_prefix][f"P@{k}"] = precision_k
                final_results[model_name][key_prefix][f"R@{k}"] = recall_k
                final_results[model_name][key_prefix][f"F1@{k}"] = f1_k
                final_results[model_name][key_prefix][f"NDCG@{k}"] = ndcg_k
                final_results[model_name][key_prefix][f"BalancedAcc@{k}"] = balanced_acc_k

                model_per_narr[f"{key_prefix}_P@{k}"] = precision_k
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

            if ap_scores:
                map_narrative = float(np.mean(ap_scores))
                model_per_narr[f"{narrative_id}_MAP"] = map_narrative
        
    else:
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

        logger.info(f"Model name: {model_name}")
        final_results[model_name] = {}
        start_time = time.time()

        text_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        overall_metrics = {
            'precision': defaultdict(list),
            'ndcg': defaultdict(list),
            'recall_at_k': defaultdict(list),
            'f1': defaultdict(list),
            'ap': defaultdict(list),
            'r_precision': defaultdict(list),
            'recall_normalized_at_100': defaultdict(list),
            'balanced_accuracy': defaultdict(list),
            'mrr': [],
            'roc_auc': [],
        }

        fails = ["1_5", "1_8", "2_2", "2_4", "2_5", "3_4", "3_5", "3_6", "4_3", "5_3"]
        narratives_processed = 0

        model_overall = {'model_name': model_name}
        model_per_narr = {}

        for narrative_id, narrative in tqdm(taxonomy.items(), desc=f"Processing {model_name}"):
            if narrative_id in fails:
                continue

            narratives_processed += 1
            ap_scores = []
            r_precision_scores = []
            r_precision_weighted_pairs = []

            if any(s in model_name for s in ["Qwen", "intfloat", "instruct", "nvidia"]):
                subnarrative_input = get_detailed_instruct(narrative)
            else:
                subnarrative_input = narrative

            query_embedding = model.encode(subnarrative_input, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, text_embeddings, top_k=max(Ks) * 2)[0]
            retrieved_claims = [claims[hit['corpus_id']] for hit in hits]
            correct_hits = [1 if retrieved_claim == narrative_id else 0 for retrieved_claim in retrieved_claims]

            total_relevant = relevant_counts[narrative_id]

            ap = get_ap_score(narrative_id, claims, hits, total_relevant)
            ap_scores.append(ap)
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

            r_precision_scores.append(r_precision)
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
                ndcg_k = ndcg_at_k(retrieved_k, total_relevant, k)
                recall_k = TP / total_relevant if total_relevant > 0 else 0.0
                f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

                tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                balanced_acc_k = 0.5 * (tpr + tnr)

                overall_metrics['precision'][k].append(precision_k)
                overall_metrics['ndcg'][k].append(ndcg_k)
                overall_metrics['balanced_accuracy'][k].append(balanced_acc_k)
                overall_metrics['recall_at_k'][k].append(recall_k)
                overall_metrics['f1'][k].append(f1_k)

                final_results[model_name][key_prefix][f"P@{k}"] = precision_k
                final_results[model_name][key_prefix][f"R@{k}"] = recall_k
                final_results[model_name][key_prefix][f"F1@{k}"] = f1_k
                final_results[model_name][key_prefix][f"NDCG@{k}"] = ndcg_k
                final_results[model_name][key_prefix][f"BalancedAcc@{k}"] = balanced_acc_k

                model_per_narr[f"{key_prefix}_P@{k}"] = precision_k
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

            if ap_scores:
                map_narrative = float(np.mean(ap_scores))
                model_per_narr[f"{narrative_id}_MAP"] = map_narrative
            if r_precision_scores:
                model_per_narr[f"{narrative_id}_Mean_R-Precision"] = np.mean(r_precision_scores)
            if r_precision_weighted_pairs:
                total_tp = sum([tp for tp, _ in r_precision_weighted_pairs])
                total_rel = sum([rel for _, rel in r_precision_weighted_pairs])
                model_per_narr[f"{narrative_id}_Weighted_R-Precision"] = total_tp / total_rel if total_rel > 0 else 0.0


    end_time = time.time()
    elapsed_time = end_time - start_time

    model_overall['elapsed_time'] = elapsed_time
    model_overall['narratives_processed'] = narratives_processed

    all_aps = [ap for narrative_aps in overall_metrics['ap'].values() for ap in narrative_aps]
    overall_map = float(np.mean(all_aps)) if all_aps else 0.0
    model_overall['Overall_MAP'] = overall_map

    all_r_precs = [r for r_list in overall_metrics['r_precision'].values() for r in r_list]
    print(all_r_precs)
    model_overall['Overall_R-Precision'] = np.mean(all_r_precs) if all_r_precs else 0.0
    print(model_overall['Overall_R-Precision'])

    total_tp_weighted = 0
    total_rel_weighted = 0
    for narrative_id in overall_metrics['r_precision']:
        rel = relevant_counts[narrative_id]
        for r in overall_metrics['r_precision'][narrative_id]:
            total_tp_weighted += r * rel
            total_rel_weighted += rel
    model_overall['Overall_Weighted_R-Precision'] = total_tp_weighted / total_rel_weighted if total_rel_weighted > 0 else 0.0
    print(model_overall['Overall_Weighted_R-Precision'])


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

    row_combined = {**{'model_name': model_name}, **model_overall, **model_per_narr}
    all_rows.append(row_combined)

    logger.info(f"{model_name}: Overall_R-Precision = {model_overall['Overall_R-Precision']:.4f}")

overall_prefixes = ("Overall_", "elapsed_time", "narratives_processed")

all_overall_keys = []
all_per_keys = []

for row in all_rows:
    for k in row.keys():
        if k == 'model_name':
            continue
        if k.startswith(overall_prefixes) or k in {"elapsed_time", "narratives_processed"}:
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