from transformers import T5Tokenizer

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Save the SentencePiece model
tokenizer.save_pretrained("reranker_pipeline/quest/models")
