# pip install pandas numpy tqdm openai sentence-transformers scikit-learn NodeRAG
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch
import time
import csv
import openai
import os
import requests

# import google.generativeai as genai
import logging
import datetime
import json
from taxonomy.v3 import taxonomy


client = openai.OpenAI(
    api_key="sk-proj-this-is-not-a-key", base_url="http://127.0.0.1:11434/v1"
)

n = 1
hyde_n = 10
run_name = f"oi_dyn_dr_emb_local_4b_narr{n}_{hyde_n}"

not_in_dataset = ["1_5", "1_8", "2_2", "2_4", "2_5", "3_4", "3_5", "3_6", "4_3", "5_3"]

# RUN_OPTION = 'DR'
# RUN_OPTION = 'STATIC'
RUN_OPTION = "NodeRAG"

os.makedirs(run_name, exist_ok=True)
results_csv_filename = f"{run_name}/results.csv"
all_gen_csv_filename = f"{run_name}/all_hyde.csv"

from logger import logger


# #data_path = "data/training/training.csv"
data_path = "data/training/test.csv"
training_data = pd.read_csv(data_path)
claims = training_data["claim"].tolist()
texts = training_data["text"].tolist()


relevant_counts = Counter(claims)
lowest_count = min(relevant_counts.values())


def get_openai_embeddings(texts, model="text-embedding-3-large", batch_size=1000):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with OpenAI"):
        batch = texts[i : i + batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return torch.tensor(embeddings)


import pickle

embeddings_path = "data/training/test2_text_embeddings_openai.pkl"


from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "mps"
batch_size = 1
run_name = "Qwen/Qwen3-Embedding-4B"
model_local = SentenceTransformer(run_name, device=device, trust_remote_code=True)
text_embeddings_local = model_local.encode(
    texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
)


def get_detailed_instruct(query: str) -> str:
    # task_description = "'Given a narrative description as a query, retrieve passages that serve this narrative; can be entailed from the narrative; can be aligned logically with the narrative'"
    task_description = "Given a text as a query retrieve relevant passages that align with narratives similar to the query"
    # task_description = instruct_models_task_desc
    # task_description = 'Given a web search query, retrieve relevant passages that answer the query'

    return f"Instruct: {task_description}\nQuery: {query}"


def get_hyde_emb_local(query, text_embeddings_training, texts_training):
    query = get_detailed_instruct(query)
    query_embedding = model_local.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, text_embeddings_training, top_k=1)[0]
    retrieved_claims = [texts_training[hit["corpus_id"]] for hit in hits]
    # print(retrieved_claims[0])
    return retrieved_claims[0]


def get_local_embeddings(texts):
    embeddings = model_local.encode(texts, convert_to_tensor=True)
    return embeddings


import time


def generate_hypotheticals(query, examples, n=1, temp=1):

    system_prompt = "You are a disinformation investigator. Your first step is to generate short disinformation texts that sound like actual ones. You get a disinformation narrative and return a disinformation text that aligns with that narrative. Return only 1 single text!"

    hypotheticals = []

    prompt_u = "You are a disinformation investigator. Given a disinformation narrative, generate a short, realistic text (such as a news excerpt, blog post, or social media post) that supports or aligns with that narrative. The text should sound plausible and could be found in the wild."
    prompt = f"""{prompt_u}

    Here are some examples:

    {examples}

    Narrative: {query}
    Text:
    """
    for _ in range(n):
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    # model="gpt-4o",
                    model="hf.co/mlabonne/gemma-3-27b-it-abliterated-GGUF:Q8_0",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    #    max_tokens=256,
                )

                text = response.choices[0].message.content.strip()
                logger.info(text)

                if text:
                    hypotheticals.append(text)
                    break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)
        else:
            hypotheticals.append("")
    return hypotheticals



import re


if RUN_OPTION == "NodeRAG":
    from NodeRAG import NodeConfig, NodeSearch
    config = NodeConfig.from_main_folder(r"./noderag_training")
    search = NodeSearch(config)




def noderag_hyde(query, n_hyde=1):
    ans = search.answer(query)
    hyde_raw = ans.retrieval_info
    units_raw = hyde_raw.split("------------high_level_element-------------")
    text = units_raw[1]
    parsed_items = [
        re.sub(r"^\d{1,2}\.\s*", "", line)
        for line in text.split("\n")
        if re.match(r"^\d{1,2}\.", line)
    ]
    # print(len(parsed_items))
    parsed_items = parsed_items[:30]
    hypotheticals = parsed_items[:5]
    return hypotheticals  # [:n_hyde]

if RUN_OPTION == "DR":
    """dynamic few shot training"""
    data_path_training = "data/training/training.csv"
    training_data_training = pd.read_csv(data_path_training)
    claims_training = training_data_training['claim'].tolist()
    texts_training = training_data_training['text'].tolist()
    #text_embeddings_training = get_openai_embeddings(texts_training)
    text_embeddings_training = get_local_embeddings(texts_training)

    all_hypos = {}
    all_hypos_emb = {}
    for el in taxonomy:
        if el not in not_in_dataset:
            narrative = taxonomy[el][0]
            all_hypos[narrative] = ""
            top_1 = get_hyde_emb_local(narrative, text_embeddings_training, texts_training)
            #top_1 = noderag_hyde(narrative, 1)[0]
        # top_1 = get_hyde_emb_oi(narrative, texts_training, text_embeddings_training)
            all_hypos[narrative] = top_1


if RUN_OPTION == "NodeRAG":
    all_hypos = {}
    for el in taxonomy:
        if el not in fails:
            narrative = taxonomy[el][0]
            all_hypos[narrative] = ""
            #top_1 = get_hyde_emb_local(narrative)
            top_1 = noderag_hyde(narrative, 1)[0]
            #top_1 = get_hyde_emb_oi(narrative)
        # logger.debug(top_1)
        # top_1 = rephrase_sample_text(narrative, top_1)
        # logger.info(top_1)
            all_hypos[narrative] = top_1



if RUN_OPTION == "STATIC":
    """static few shot test"""
    data_path_training = "data/training/training.csv"
    training_data_training = pd.read_csv(data_path_training)
    claims_training = training_data_training["claim"].tolist()
    texts_training = training_data_training["text"].tolist()
    claims_done = []
    all_hypos = {}
    for i, el in enumerate(claims_training):
        if el not in not_in_dataset:
            if el not in claims_done and el != "0_0":
                narrative_id = el
                narrative = taxonomy[narrative_id][0]
                all_hypos[narrative] = ""
                claims_done.append(el)
                all_hypos[narrative] = texts_training[i]


# print(all_hypos)
logger.debug(all_hypos)


def create_examples(hypos):
    examples = "\n\n"
    for el in hypos:
        new_ex = "Narrative: " + el + "\n" + "Text: " + hypos[el] + "\n\n"
        examples += new_ex
    return examples


examples = create_examples(all_hypos)
logger.debug(examples)

def get_hyde_scores(query, narrative_id, n_hyde=3):
    hyde_docs = generate_hypotheticals(query, examples, n=10, temp=1)
    hyde_docs = [get_detailed_instruct(doc) for doc in (hyde_docs) if doc.strip()]

    if not hyde_docs:
        return torch.zeros(len(text_embeddings_local))
    hyde_embeddings = get_local_embeddings(hyde_docs)
    all_sims = []
    for emb in hyde_embeddings:
        sims = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), text_embeddings_local, dim=1
        )
        all_sims.append(sims)
    agg_sims = torch.stack(all_sims).max(dim=0).values
    return agg_sims, hyde_docs


def dcg_at_k(relevance_scores, k):
    relevance_scores = relevance_scores[:k]
    return sum(
        [(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)]
    )


def ndcg_at_k(retrieved_binary, relevant_count, k):
    if relevant_count == 0:
        return 0.0
    ideal_relevance = [1] * min(relevant_count, k) + [0] * max(0, k - relevant_count)
    dcg = dcg_at_k(retrieved_binary, k)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


Ks = [1, 3, 5, 10, 25, 50, 100, 500, 1000, 1500, 2000, 2500, lowest_count]
all_rows = []
run_row = {"run_name": run_name}
final_results = {}
final_results[run_name] = {}
overall_metrics = defaultdict(lambda: defaultdict(list))

narratives_processed = 0

all_generated_texts = {}

start_time = time.time()
for i in range(n):
    run_row = {"run_name": str(i) + "_" + run_name}
    for narrative_id, subnarrative in tqdm(
        taxonomy.items(), desc=f"Processing {run_name}"
    ):
        if narrative_id in not_in_dataset:
            continue

        subnarrative = subnarrative[0]

        all_gen_key = str(narrative_id) + "_" + str(n)
        all_generated_texts[all_gen_key] = {}

        narratives_processed += 1
        ap_scores = []
        r_precision_scores = []
        r_precision_weighted_pairs = []

        narr_count = 0
        logger.info(
            f"Narratives processed: {narratives_processed}, Narrative ID: {narrative_id}"
        )

        sims, hyde_docs = get_hyde_scores(subnarrative, narrative_id, n_hyde=10)
        all_generated_texts[all_gen_key]["query"] = subnarrative
        all_generated_texts[all_gen_key]["hyde_docs"] = hyde_docs

        topk_scores, topk_indices = torch.topk(
            sims,
            k=
            #  max(Ks) * 2
            len(texts),
        )
        retrieved_claims = [claims[i] for i in topk_indices.tolist()]
        correct_hits = [
            1 if retrieved_claim == narrative_id else 0
            for retrieved_claim in retrieved_claims
        ]

        total_relevant = relevant_counts[narrative_id]
        y_true_for_ap = [1 if claim == narrative_id else 0 for claim in claims]
        y_scores_for_ap = torch.zeros(len(claims))
        for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()):
            y_scores_for_ap[idx] = score

        ap = (
            average_precision_score(y_true_for_ap, y_scores_for_ap.tolist())
            if total_relevant > 0
            else 0.0
        )
        ap_scores.append(ap)
        overall_metrics["ap"][narrative_id].append(ap)

        if total_relevant > 0:
            k_relative = int(0.1 * total_relevant)
            recall_relative = (
                sum(correct_hits[:k_relative]) / total_relevant
                if k_relative > 0
                else 0.0
            )
            k_fixed = 100
            retrieved_relevant_at_100 = sum(correct_hits[:k_fixed])
            max_possible_at_100 = min(k_fixed, total_relevant)
            recall_normalized_at_100 = (
                retrieved_relevant_at_100 / max_possible_at_100
                if max_possible_at_100 > 0
                else 0.0
            )
            r_k = total_relevant
            r_precision = sum(correct_hits[:r_k]) / r_k if r_k > 0 else 0.0
            r_precision_weighted_pairs.append(
                (r_precision * total_relevant, total_relevant)
            )
            logger.debug(f"Narrative ID: {narrative_id}: R-Precision: {r_precision}")
        else:
            recall_relative = 0.0
            recall_normalized_at_100 = 0.0
            r_precision = 0.0

        r_precision_scores.append(r_precision)
        overall_metrics["recall_relative_10_percent"][narrative_id].append(
            recall_relative
        )
        overall_metrics["recall_normalized_at_100"][narrative_id].append(
            recall_normalized_at_100
        )
        overall_metrics["r_precision"][narrative_id].append(r_precision)

        for k in Ks:
            retrieved_k = correct_hits[:k]
            precision = sum(retrieved_k) / k if k > 0 else 0.0
            recall = sum(retrieved_k) / total_relevant if total_relevant > 0 else 0.0
            ndcg = ndcg_at_k(retrieved_k, total_relevant, k)

            overall_metrics["precision"][k].append(precision)
            overall_metrics["recall_at_k"][k].append(recall)
            overall_metrics["ndcg"][k].append(ndcg)

        run_row[f"{narrative_id}_{narr_count}_AP"] = ap
        narr_count += 1

        if ap_scores:
            run_row[f"{narrative_id}_MAP"] = np.mean(ap_scores)
        if r_precision_scores:
            run_row[f"{narrative_id}_Mean_R-Precision"] = np.mean(r_precision_scores)
        if r_precision_weighted_pairs:
            total_tp = sum([tp for tp, _ in r_precision_weighted_pairs])
            total_rel = sum([rel for _, rel in r_precision_weighted_pairs])
            run_row[f"{narrative_id}_Weighted_R-Precision"] = (
                total_tp / total_rel if total_rel > 0 else 0.0
            )

    end_time = time.time()
    run_row["elapsed_time"] = end_time - start_time
    run_row["narratives_processed"] = narratives_processed

    all_aps = [
        ap for narrative_aps in overall_metrics["ap"].values() for ap in narrative_aps
    ]
    run_row["Overall_MAP"] = np.mean(all_aps) if all_aps else 0.0
    all_r_precs = [
        r for r_list in overall_metrics["r_precision"].values() for r in r_list
    ]
    run_row["Overall_R-Precision"] = np.mean(all_r_precs) if all_r_precs else 0.0
    logger.debug(run_row["Overall_R-Precision"])

    total_tp_weighted = 0
    total_rel_weighted = 0
    for narrative_id in overall_metrics["r_precision"]:
        rel = relevant_counts[narrative_id]
        for r in overall_metrics["r_precision"][narrative_id]:
            total_tp_weighted += r * rel
            total_rel_weighted += rel
    run_row["Overall_Weighted_R-Precision"] = (
        total_tp_weighted / total_rel_weighted if total_rel_weighted > 0 else 0.0
    )
    logger.debug(run_row["Overall_Weighted_R-Precision"])

    for k in Ks:
        run_row[f"Overall_P@{k}"] = (
            np.mean(overall_metrics["precision"][k])
            if overall_metrics["precision"][k]
            else 0.0
        )
        run_row[f"Overall_R@{k}"] = (
            np.mean(overall_metrics["recall_at_k"][k])
            if overall_metrics["recall_at_k"][k]
            else 0.0
        )
        run_row[f"Overall_NDCG@{k}"] = (
            np.mean(overall_metrics["ndcg"][k]) if overall_metrics["ndcg"][k] else 0.0
        )

    all_rows.append(run_row)

all_keys = sorted(set().union(*(row.keys() for row in all_rows)))
with open(results_csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(all_rows)
logger.info(f"\nResults successfully saved to {results_csv_filename}")


all_generated_texts_keys = set()
for item in all_generated_texts.values():
    all_generated_texts_keys.update(item.keys())
all_generated_texts_keys = sorted(list(all_generated_texts_keys))

with open(all_gen_csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_generated_texts_keys)
    writer.writeheader()
    writer.writerows(all_generated_texts.values())
logger.info(f"\All hypotheticaly successfully saved to {all_gen_csv_filename}")
