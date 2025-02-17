import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/workspace/aneema_umass_edu-set-query/.cache'

import torch
import wandb
import random
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
import pandas as pd
from datasets import Dataset
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback
from collections import deque

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common import tsv_utils
import argparse

parser = argparse.ArgumentParser(description="Train a SentenceTransformer model.")
parser.add_argument("--runname", type=str, default='v0.0.3_gtr-base-20eps-smoothearlystop', help="Run name.")
parser.add_argument("--train_data", type=str, default='dataset/train/train_query_doc.csv', help="Path to the training data file.")
parser.add_argument("--val_data", type=str, default='dataset/val/val_query_doc.csv', help="Path to the validation data file.")
parser.add_argument("--output_dir", type=str, default='intermediate-files/checkpoints', help="Directory to save the model checkpoints.")
parser.add_argument("--wandb_dir", type=str, default="intermediate-files/wandb", help="Path to the Weights & Biases directory.")
parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")

args = parser.parse_args()

runname = args.runname
wandb.init(
    project="set-based-retrieval", 
    name=f"{runname}", 
    dir=args.wandb_dir
)

model = SentenceTransformer('sentence-transformers/gtr-t5-base')


doc_text_ids = tsv_utils.read_tsv('/home/aneema_umass_edu/quest/dataset/doc_text_map.tsv', max_splits=1)
doc_idx_to_text = {idx: doc_text for idx, doc_text in doc_text_ids}

train_query_doc_pairs = pd.read_csv(args.train_data)
val_query_doc_pairs = pd.read_csv(args.val_data)

train_queries = train_query_doc_pairs['query'].tolist()
train_docs = train_query_doc_pairs['doc'].tolist()

val_queries = val_query_doc_pairs['query'].tolist()
val_docs = val_query_doc_pairs['doc'].tolist()

# train_neg_indices = [random.randint(0, len(doc_idx_to_text) - 1) for _ in range(len(train_docs))]
# val_neg_indices = [random.randint(0, len(doc_idx_to_text) - 1) for _ in range(len(val_docs))]

# train_neg_docs = [doc_idx_to_text[str(idx)].replace("\t", " ").replace("'''", "") for idx in train_neg_indices]
# val_neg_docs = [doc_idx_to_text[str(idx)].replace("\t", " ").replace("'''", "") for idx in val_neg_indices]

train_dataset = Dataset.from_dict({
    "anchor": train_queries,
    "positive": train_docs,
    # "negative": train_neg_docs
})

val_dataset = Dataset.from_dict({
    "anchor": val_queries,
    "positive": val_docs,
    # "negative": val_neg_docs
})


with open('/home/aneema_umass_edu/quest/dataset/irrel_docs.txt', 'r') as file:
    irrel_docs = [line.strip() for line in file.readlines()]


loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
# loss = losses.MultipleNegativesRankingLoss(model)

class MRREvaluationCallback(TrainerCallback):
    def __init__(self, model, queries, relevant_docs, irrelevant_docs):
        self.model = model
        self.queries = queries
        self.relevant_docs = relevant_docs
        self.irrelevant_docs = irrelevant_docs

    def on_evaluate(self, args, state, control, **kwargs):
        mrr_score = self.evaluate_mrr()
        print(f"MRR Score: {mrr_score}")
        wandb.log({"eval/MRR": mrr_score})

    def evaluate_mrr(self):
        query_embeddings = self.model.encode(self.queries, convert_to_tensor=True)
        relevant_doc_embeddings  = self.model.encode(self.relevant_docs, convert_to_tensor=True)
        irrelevant_doc_embeddings = self.model.encode(self.irrelevant_docs, convert_to_tensor=True)

        reciprocal_ranks = []
        for query_embedding, relevant_doc_embedding in zip(query_embeddings, relevant_doc_embeddings):
            doc_embeddings = torch.cat((relevant_doc_embedding.unsqueeze(0), irrelevant_doc_embeddings), dim=0)
            similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)

            sorted_indices = torch.argsort(similarities, descending=True)

            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, threshold=0.001, metric="eval_loss", minimize=True, window_size=5, save_path_last=None):
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.minimize = minimize
        self.best_score = float("inf") if minimize else -float("inf")
        self.counter = 0
        self.window_size = window_size
        self.metric_window = deque(maxlen=window_size)
        self.save_path_last = save_path_last 

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_score = metrics.get(self.metric)
        
        if current_score is not None:
            self.metric_window.append(current_score)
            smoothened_score = sum(self.metric_window) / len(self.metric_window)
            
            improvement = (self.best_score - smoothened_score) if self.minimize else (smoothened_score - self.best_score)
            if improvement > self.threshold:
                self.best_score = smoothened_score
                self.counter = 0 
            else:
                self.counter += 1

            if self.counter >= self.patience:
                print(f"Stopping early after {self.counter} evaluations without improvement.")

                if self.save_path_last:
                    control.should_training_stop = True
                    
                    save_path = self.save_path_last + f"-{state.global_step}"
                    os.makedirs(save_path, exist_ok=True)
                    
                    trainer.save_model(save_path)
                    print(f"Last model saved at {save_path}")


mrr_callback = MRREvaluationCallback(model, val_queries, val_docs, irrel_docs)

last_model_dir = f"{args.output_dir}/{runversion}_{runname}/checkpoint-last"
early_stopping_callback = EarlyStoppingCallback(patience=5, threshold=0, metric="eval_loss", window_size=5, save_path_last=last_model_dir)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"{args.output_dir}/{runversion}_{runname}",
    # # Optional training parameters:
    num_train_epochs=args.num_train_epochs,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    # eval_steps=100,
    eval_steps = 2,
    save_strategy="steps",
    # save_steps=1000,
    save_steps=10,
    save_total_limit=5,
    # logging_steps=10,
    logging_steps=1,
    run_name=f"{runversion}_{runname}",
    report_to="wandb", 

    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,

    # learning_rate=1e-3,

    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    callbacks=[mrr_callback, early_stopping_callback]
)

trainer.train()

trainer.save_model(f"{args.output_dir}/{runversion}_{runname}/checkpoint-best" + f"-{model.model_card_data.best_model_step}")