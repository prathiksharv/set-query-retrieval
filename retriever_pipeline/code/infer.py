# %%

import sys
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/workspace/aneema_umass_edu-set-query/.cache'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm
from src.common import jsonl_utils, example_utils, document_utils
import argparse

# %%

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for sentence embeddings")
    parser.add_argument('--questdir', type=str, default='/home/aneema_umass_edu/quest/dataset/', help='Path to saved quest data')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--runname', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--embdir', type=str, default='/scratch/workspace/aneema_umass_edu-set-query/quest/embeds', help='Path to the embeddings directory')
    parser.add_argument('--embtype', type=str, default='conj', choices=['conj', 'noconj'], help='Specify if the composite queries should have conjuctions.')
    parser.add_argument('--outdir', type=str, default='/home/aneema_umass_edu/quest/out/gtr-t5', help='Path to the output directory')

    return parser.parse_args()

args = parse_args()

if args.embtype == 'conj':
    args.embtype = ''

# %%

model = SentenceTransformer('sentence-transformers/gtr-t5-base',
                                # device=device
                            )

# %%

qemb = np.load(f'{args.embdir}/{args.runname}/{args.mode}/{args.embtype}queryembs.npy')
demb = np.load(f'{args.embdir}/{args.runname}/{args.mode}/docembs.npy')

# %%

similarity = model.similarity(qemb, demb)

# %%

queries = example_utils.read_examples(f'{args.questdir}/{args.mode}/{args.mode}.jsonl')
documents = document_utils.read_documents(f'{args.questdir}/documents.jsonl')

# breakpoint()

# %%

k = args.k

predictions = []

# breakpoint()

for i in tqdm(range(len(similarity))):
    example = queries[i]
    top_k_indices = np.argsort(similarity[i])[-k:].numpy()[::-1]
    docs = []
    scores = []
    for ind in top_k_indices:
        docs.append(documents[ind].title)
        scores.append(similarity[i][ind].item())

    predictions.append(
        example_utils.Example(
            original_query=example.original_query,
            query=example.query,
            docs=docs,
            scores=scores,
            metadata=example.metadata
        )
    )

# breakpoint()

os.makedirs(f'{args.outdir}/{args.runname}/{args.mode}', exist_ok=True)
example_utils.write_examples(f'{args.outdir}/{args.runname}/{args.mode}/{args.embtype}k{args.k}.jsonl', predictions)

