# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from transformers import TrainingArguments, Trainer

# # Step 1: Load Pretrained T5 Model & Tokenizer
# model_name = "t5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Step 2: Load Training Data (Your TSV Format)
# data_path = "/home/prumalevishw_umass_edu/iesl-set-query-retrieval/reranker_pipeline/quest/outputs/formatted-test-k20.tsv" #update to training data
# df = pd.read_csv(data_path, sep="\t", names=["query", "document", "label"])

# # Convert "relevant" → 1, "not relevant" → 0
# df["label"] = df["label"].apply(lambda x: "1" if x.lower() == "relevant" else "0")

# # Convert to Hugging Face Dataset
# dataset = Dataset.from_pandas(df)

# # Step 3: Tokenization Function
# def preprocess_function(examples):
#     inputs = [f"Query: {q} Document: {d}" for q, d in zip(examples["query"], examples["document"])]
#     targets = [str(label) for label in examples["label"]]  # Binary labels ("1" or "0")

#     model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
#     labels = tokenizer(targets, padding="max_length", truncation=True, max_length=4)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # Apply Tokenization
# tokenized_dataset = dataset.map(preprocess_function, batched=True)

# # Step 4: Training Arguments
# training_args = TrainingArguments(
#     output_dir="./t5_finetuned",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# # Step 5: Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
# )

# # Step 6: Fine-Tune the Model
# trainer.train()

# # Step 7: Save the Fine-Tuned Model
# model.save_pretrained("t5_cross_encoder")
# tokenizer.save_pretrained("t5_cross_encoder")

# print("Fine-tuning complete! Model saved at: t5_cross_encoder")


import torch
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Custom T5 for classification
class T5ForBinaryClassification(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.d_model, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Load model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForBinaryClassification.from_pretrained(model_name)

# Load and preprocess data
data_path = "/home/prumalevishw_umass_edu/iesl-set-query-retrieval/reranker_pipeline/quest/outputs/formatted-short-test-k20.tsv"
df = pd.read_csv(data_path, sep="\t", names=["query", "document", "label"])
df = df.dropna(subset=["query", "document", "label"])
df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "relevant" else 0)
train_data, eval_data = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

def preprocess_function(examples):
    inputs = [f"Query: {q} Document: {d}" for q, d in zip(examples["query"], examples["document"])]
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    model_inputs["labels"] = torch.tensor([int(label) for label in examples["label"]], dtype=torch.long)
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="reranker_pipeline/quest/fine-tuned-cross-encoder/t5_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train and save
trainer.train()
model.save_pretrained("reranker_pipeline/quest/fine-tuned-cross-encoder/t5_cross_encoder")
tokenizer.save_pretrained("reranker_pipeline/quest/fine-tuned-cross-encoder/t5_cross_encoder")
print("✅ Fine-tuning complete! Model saved at: t5_cross_encoder")