from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Load fine-tuned model
model_path = "t5_cross_encoder"  # Path to fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Define input/output files
input_file = "/home/prumalevishw_umass_edu/iesl-set-query-retrieval/reranker_pipeline/quest/outputs/inference_output-k20.txt"
output_file = "/home/prumalevishw_umass_edu/iesl-set-query-retrieval/reranker_pipeline/quest/outputs/t5x_scores.jsonl"

# Read input file containing query-document pairs
with open(input_file, "r") as f:
    inputs = f.readlines()

# Process inputs and get scores
outputs = []
for text in inputs:
    model_inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**model_inputs, max_new_tokens=1)  # Binary classification output
    score = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Store in JSON format
    outputs.append({"query_doc_pair": text.strip(), "score": float(score)})

# Save results in `t5x_scores.jsonl`
with open(output_file, "w") as f:
    for entry in outputs:
        f.write(json.dumps(entry) + "\n")

print(f"Scoring complete! Saved at: {output_file}")
