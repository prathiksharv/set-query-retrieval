import pandas as pd

# Read TSV file
input_file = "reranker_pipeline/quest/outputs/formatted-test-k20.tsv"
output_file = "reranker_pipeline/quest/outputs/formatted-short-test-k20.tsv"

# Read the file
df = pd.read_csv(input_file, sep="\t")

# Select first 1000 rows
df_subset = df.head(1000)

# Write to a new file
df_subset.to_csv(output_file, sep="\t", index=False)

print(f"Copied first 1000 rows to {output_file}")
