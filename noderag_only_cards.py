"""
NOTE: For this to run, you need to monkey patch the NodeRAG library.
Add the following function to the NodeSearch class within the library files:

    def search_topk(self,query:str, topk=500):

        query_embedding = np.array(self.config.embedding_client.request(query),dtype=np.float32)

        retrieval_test = Retrieval(self.config,self.id_to_text,self.accurate_id_to_text,self.id_to_type)
        HNSW_results = self.hnsw.search(query_embedding,HNSW_results=topk)
        retrieval_test.HNSW_results_with_distance = HNSW_results

        narr_texts = []
        scores = []
        ids = []

        for i, id in enumerate(retrieval_test.HNSW_results):
            if id in self.id_to_text.keys():
                narr_texts.append(self.id_to_text[id])
                ids.append(id)


        retrieval_test = Retrieval(self.config,self.id_to_text,self.accurate_id_to_text,self.id_to_type)
        HNSW_results = self.hnsw.search(query_embedding,HNSW_results=topk)
        retrieval_test.HNSW_results_with_distance = HNSW_results

        for el, el2 in retrieval_test.HNSW_results_with_distance:
            scores.append(el.item())
        scores = scores[:len(ids)]

        return narr_texts, scores
"""





import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
import torch
import time
import csv


from taxonomy.v3 import taxonomy

csv_filename = 'rertrieval_cards_all_nr_only3.csv'
data_path = "data/training/test.csv"

training_data = pd.read_csv(data_path)
claims = training_data['claim'].tolist()
texts = training_data['text'].tolist()

relevant_counts = Counter(claims) 

from NodeRAG import NodeConfig, NodeSearch

config = NodeConfig.from_main_folder("./noderag")

# Initialize search engine
search = NodeSearch(config)


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
model_names = [
    'thenlper/gte-base',
    # 'thenlper/gte-large',

    #'Alibaba-NLP/gte-multilingual-base',
  #  'sileod/deberta-v3-large-tasksource-nli',
  #   'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
   #  'Qwen/Qwen3-Embedding-0.6B',
   # 'Qwen/Qwen3-Embedding-4B',
   #  'intfloat/multilingual-e5-large-instruct',
   # 'intfloat/e5-mistral-7b-instruct',
   #  'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
   #  'Alibaba-NLP/gte-Qwen2-7B-instruct',

    #    'Qwen/Qwen3-Embedding-8B',

#      'Linq-AI-Research/Linq-Embed-Mistral',
 #    'nvidia/NV-Embed-v2'
]


all_rows = []
final_results = {}

for model_name in model_names:
    try:
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load model {model_name}. Error: {e}")
        continue

    print("#####")
    print("Model name:", model_name)
    final_results[model_name] = {}
    start_time = time.time()

    def get_detailed_instruct(query: str) -> str:
        task_description = 'Given a narrative description as a query, retrieve passages that serve this narrative; can be entailed from the narrative; can be aligned logically with the narrative'
        return f'Instruct: {task_description}\nQuery: {query}'

    #text_embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    Ks = [1, 2, 3, 10, 20, 100, 500, 1000, 1500, 2000, 2500, 21]

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

    overall_metrics = {
        'precision': defaultdict(list),
        'recall_at_k': defaultdict(list),
        'ndcg': defaultdict(list),
        'ap': defaultdict(list),
        'r_precision': defaultdict(list),
        'recall_relative_10_percent': defaultdict(list),
        'recall_normalized_at_100': defaultdict(list),
        'y_true': [],
        'y_pred': [],
    }

    fails = ["1_5", "1_8", "2_2", "2_4", "2_5", "3_4", "3_5", "3_6", "4_3", "5_3"]
    narratives_processed = 0
    model_row = {'model_name': model_name}

    for narrative_id, subnarratives in tqdm(taxonomy.items(), desc=f"Processing {model_name}"):
        if narrative_id in fails:
            continue

        narratives_processed += 1
        ap_scores = []
        r_precision_scores = []
        r_precision_weighted_pairs = []
        narr_count = 0

        for subnarrative in subnarratives:
            if any(s in model_name for s in ["Qwen", "intfloat", "instruct", "nvidia"]):
                subnarrative_input = get_detailed_instruct(subnarrative)
            else:
                subnarrative_input = subnarrative

            narr_texts, scores = search.search_topk(subnarrative, topk=10000)
            #print(len(hits))
            #retrieved_claims = [claims[hit] for hit in hits]
            retrieved_claims = []
            check = []
            y_scores_for_ap = [0] * len(claims)

            for i, el in enumerate(narr_texts[:len(claims)]):
                for j, el2 in enumerate(texts):
                    if el[:500] in el2 and el[:500] not in check:
                        retrieved_claims.append(claims[j])
                        #print(j)
                        check.append(texts[j])
                        try:
                           y_scores_for_ap[j] = 1/(1+scores[i])
                        except Exception as e:
                            print(e)
                            print(j)
            correct_hits = [1 if retrieved_claim == narrative_id else 0 for retrieved_claim in retrieved_claims]

            total_relevant = relevant_counts[narrative_id]
            y_true_for_ap = [1 if claim == narrative_id else 0 for claim in claims]

            ap = average_precision_score(y_true_for_ap, y_scores_for_ap) if total_relevant > 0 else 0.0
            ap_scores.append(ap)
            overall_metrics['ap'][narrative_id].append(ap)

            if total_relevant > 0:
                k_relative = int(0.1 * total_relevant)
                recall_relative = sum(correct_hits[:k_relative]) / total_relevant if k_relative > 0 else 0.0
                k_fixed = 100
                retrieved_relevant_at_100 = sum(correct_hits[:k_fixed])
                max_possible_at_100 = min(k_fixed, total_relevant)
                recall_normalized_at_100 = retrieved_relevant_at_100 / max_possible_at_100 if max_possible_at_100 > 0 else 0.0
                r_k = total_relevant
                r_precision = sum(correct_hits[:r_k]) / r_k if r_k > 0 else 0.0
                r_precision_weighted_pairs.append((r_precision * total_relevant, total_relevant))

                #print(r_precision)
            else:
                recall_relative = 0.0
                recall_normalized_at_100 = 0.0
                r_precision = 0.0

            print(narrative_id, r_precision, ap)
            r_precision_scores.append(r_precision)
            overall_metrics['recall_relative_10_percent'][narrative_id].append(recall_relative)
            overall_metrics['recall_normalized_at_100'][narrative_id].append(recall_normalized_at_100)
            overall_metrics['r_precision'][narrative_id].append(r_precision)

            final_results[model_name][f"{narrative_id}_{narr_count}"] = {
                "Recall@10%": recall_relative,
                "NormalizedRecall@100": recall_normalized_at_100,
                "R-Precision": r_precision
            }
            model_row[f"{narrative_id}_{narr_count}_Recall@10%"] = recall_relative
            model_row[f"{narrative_id}_{narr_count}_NormalizedRecall@100"] = recall_normalized_at_100
            model_row[f"{narrative_id}_{narr_count}_R-Precision"] = r_precision

            for k in Ks:
                retrieved_k = correct_hits[:k]
                precision = sum(retrieved_k) / k if k > 0 else 0.0
                recall = sum(retrieved_k) / total_relevant if total_relevant > 0 else 0.0
                ndcg = ndcg_at_k(retrieved_k, total_relevant, k)

                overall_metrics['precision'][k].append(precision)
                overall_metrics['recall_at_k'][k].append(recall)
                overall_metrics['ndcg'][k].append(ndcg)

                final_results[model_name][f"{narrative_id}_{narr_count}"][f"P@{k}"] = precision
                final_results[model_name][f"{narrative_id}_{narr_count}"][f"NDCG@{k}"] = ndcg

                model_row[f"{narrative_id}_{narr_count}_P@{k}"] = precision
                model_row[f"{narrative_id}_{narr_count}_NDCG@{k}"] = ndcg

            #model_row[f"{narrative_id}_{narr_count}_AP"] = ap
            narr_count += 1

        # if ap_scores:
        #     map_narrative = np.mean(ap_scores)
        #     model_row[f"{narrative_id}_MAP"] = map_narrative
        if r_precision_scores:
            mean_r_prec = np.mean(r_precision_scores)
            model_row[f"{narrative_id}_Mean_R-Precision"] = mean_r_prec
        if r_precision_weighted_pairs:
            total_tp = sum([tp for tp, _ in r_precision_weighted_pairs])
            total_rel = sum([rel for _, rel in r_precision_weighted_pairs])
            model_row[f"{narrative_id}_Weighted_R-Precision"] = total_tp / total_rel if total_rel > 0 else 0.0

    end_time = time.time()
    elapsed_time = end_time - start_time
    model_row['elapsed_time'] = elapsed_time
    model_row['narratives_processed'] = narratives_processed

    all_aps = [ap for narrative_aps in overall_metrics['ap'].values() for ap in narrative_aps]
    overall_map = np.mean(all_aps) if all_aps else 0.0
    model_row['Overall_MAP'] = overall_map

    all_r_precs = [r for r_list in overall_metrics['r_precision'].values() for r in r_list]
    model_row['Overall_R-Precision'] = np.mean(all_r_precs) if all_r_precs else 0.0

    print(f"{model_name}: { model_row['Overall_R-Precision']}")

    total_tp_weighted = 0
    total_rel_weighted = 0
    for narrative_id in overall_metrics['r_precision']:
        rel = relevant_counts[narrative_id]
        for r in overall_metrics['r_precision'][narrative_id]:
            total_tp_weighted += r * rel
            total_rel_weighted += rel
    model_row['Overall_Weighted_R-Precision'] = total_tp_weighted / total_rel_weighted if total_rel_weighted > 0 else 0.0
    print(model_row['Overall_Weighted_R-Precision'])


    for k in Ks:
        model_row[f"Overall_P@{k}"] = np.mean(overall_metrics['precision'][k]) if overall_metrics['precision'][k] else 0.0
        model_row[f"Overall_R@{k}"] = np.mean(overall_metrics['recall_at_k'][k]) if overall_metrics['recall_at_k'][k] else 0.0
        model_row[f"Overall_NDCG@{k}"] = np.mean(overall_metrics['ndcg'][k]) if overall_metrics['ndcg'][k] else 0.0

    all_recall_relative = [r for recalls in overall_metrics['recall_relative_10_percent'].values() for r in recalls]
    model_row["Overall_Recall@10%"] = np.mean(all_recall_relative) if all_recall_relative else 0.0

    all_recall_norm_100 = [r for recalls in overall_metrics['recall_normalized_at_100'].values() for r in recalls]
    model_row["Overall_NormalizedRecall@100"] = np.mean(all_recall_norm_100) if all_recall_norm_100 else 0.0

    all_rows.append(model_row)

if all_rows:
    all_keys = sorted(set().union(*(row.keys() for row in all_rows)))
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults successfully saved to {csv_filename}")
else:
    print("\nNo results were generated to save.")
