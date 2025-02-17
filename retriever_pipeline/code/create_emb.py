# %%

from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import numpy as np 
from tqdm import tqdm
import torch
import argparse
import os
import re

# %%

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# %%

def extract_marked_queries(text):
    return re.findall(r'<mark>(.*?)</mark>', text)

# breakpoint()

# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create embeddings for queries and documents.')
    parser.add_argument('--questdir', type=str, default='/home/aneema_umass_edu/quest/dataset/', help='Path to saved quest data')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--runname', type=str, required=True)
    parser.add_argument('--embtype', type=str, default='conj', choices=['conj', 'noconj'], help='Specify if the composite queries should have conjuctions.')
    parser.add_argument('--useckpt', action='store_true', help='Use checkpoint if this flag is set')
    parser.add_argument('--ckptpath', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--embdir', type=str, default='/scratch/workspace/aneema_umass_edu-set-query/quest/embeds', help='Path to the embeddings directory')
    parser.add_argument('--skipdocs', action='store_true', help='Skip generating document embeddings')


    args = parser.parse_args()

    # print(args)
    # breakpoint()

    query_jsonl = load_jsonl(f"{args.questdir}/{args.mode}/{args.mode}.jsonl") 
    doc_jsonl = load_jsonl(f'{args.questdir}/documents.jsonl') 

    # %%

    docs = [d['text'] for d in doc_jsonl]
    if args.embtype == 'conj':
        queries = [t['query'] for t in query_jsonl]
        args.embtype = ''
    elif args.embtype == 'noconj':
        queries = [' '.join(extract_marked_queries(t['original_query'])) for t in query_jsonl]
    # docs = docs[:1000]

    # %%

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.useckpt:
        model = SentenceTransformer(args.ckptpath)
    else:
        model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    

    qemb = model.encode(queries, show_progress_bar=True)

    # Save the query embeddings to a file
    os.makedirs(f"{args.embdir}/{args.runname}/{args.mode}", exist_ok=True)
    np.save(f"{args.embdir}/{args.runname}/{args.mode}/{args.embtype}queryembs.npy", qemb)

    

    if not args.skipdocs:
        dembs = []

        batch = 128

        for i in tqdm(range(0, len(docs), batch), desc='Computing document embeddings'):
            batch_docs = docs[i:i+batch] 
            dembs.append(model.encode(batch_docs, batch_size=batch))

        demb = np.vstack(dembs)

        np.save(f"{args.embdir}/{args.runname}/{args.mode}/docembs.npy", demb)

# %%






