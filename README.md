# SpecFi-Narrative-Retrieval


## About

This repo documents our evaluation scripts for the paper **Exploring Contingency: Retrieving Disinformation Narratives with Speculative Fiction Generation**. Please note: This repository accompanies the paper and documents the approach taken for reference. It is not a procuction-ready implementation and may require modifications to execute.

## Overview over evaluation scripts

| **Dataset** | **Setup** | **Filename** |
|---|---|---|
|CARDS|Sparse Retrieval|bm25_cards.py|
|CARDS|Dense Retrieval|dr_cards.py|
|CARDS|SpecFi|specfi_cards.py|
|CARDS|NodeRAG only|noderag_only_cards.py|
|CO|Sparse Retrieval|co_cards.py|
|CO|Dense Retrieval|dr_co.py|
|CO|SpecFi|specfi_co.py|
|PN|Sparse Retrieval|pn_cards.py|
|PN|Dense Retrieval|dr_pn.py|
|PN|SpecFi|pn_specfi.py|





## Notes
- scripts that require LLM calls are intended to work with open source versions. To run, please start an openai API compatible service like vllm, llama.cpp/llama-server or `ollama run hf.co/unsloth/phi-4-GGUF:Q8_0` 
- NodeRAG requires preprocessed files, e.g., in the directory ./noderag. Follow the [NodeRAG documentation](https://terry-xu-666.github.io/NodeRAG_web/docs/) and use each text sample with its own .txt file as an input.


## Data
The datasets are available at:
- CARDS: https://www.nature.com/articles/s41598-021-01714-4#data-availability
- Climate Obstruction: https://github.com/climate-nlp/climate-obstruction-narratives
- PolyNarrative: https://propaganda.math.unipd.it/semeval2025task10/

The CARDS dataset has two columns: text and claim. "claim" contains the narrative_ids. The other two datasets need to be mapped accordingly in a preprocessing step, where texts mapped to multiple narrative ids should be labeled with the collected ids separated by a simicolon.
