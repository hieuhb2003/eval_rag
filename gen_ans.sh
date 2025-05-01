python gen_ans.py \
    --result_path "/home/hungpv/projects/train_embedding/nanographrag/results_musique/bge_naive_query_en_doc_vi.json" \
    --query_path "/home/hungpv/projects/TN/data/data_musique/dev_queries_en.json" \
    --docs_path "/home/hungpv/projects/TN/data/data_musique/filter_corpus_vi.json" \
    --output_dir "./results" \
    --only_first_model \
    --language English \
    --top_k 10

python gen_ans.py \
    --result_path "/home/hungpv/projects/train_embedding/nanographrag/results_musique/query_en_merge.json" \
    --query_path "/home/hungpv/projects/TN/data/data_musique/dev_queries_en.json" \
    --docs_path "/home/hungpv/projects/TN/data/data_musique/filter_corpus_vi.json" \
    --output_dir "./results" \
    --only_first_model \
    --language English \
    --top_k 10