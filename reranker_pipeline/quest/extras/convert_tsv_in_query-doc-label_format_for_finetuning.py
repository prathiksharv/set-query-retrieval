import pandas as pd

# Input and Output Paths
input_file = "reranker_pipeline/quest/outputs/test-k20.tsv"  # Update the path
output_file = "reranker_pipeline/quest/outputs/formatted-test-k20.tsv"  # Update the path

# Read input file
data = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            # Remove "query:" and "doc:" prefixes and extract label
            query_start = line.find("query:") + len("query:")
            doc_start = line.find("doc:") + len("doc:")
            
            query = line[query_start:doc_start - len("doc:")].strip()
            doc = line[doc_start:].strip()

            label = None
            if doc.endswith("not relevant"):
                label = "not relevant"
                doc = doc[: -len("not relevant")].strip()  # Remove the label from the doc
            elif doc.endswith("relevant"):
                label = "relevant"
                doc = doc[: -len("relevant")].strip()  # Remove the label from the doc
            
            if label:
                data.append([query, doc, label])

# Create DataFrame
df = pd.DataFrame(data, columns=['query', 'doc', 'label'])

# Save to TSV
df.to_csv(output_file, sep='\t', index=False)

print(f"Extracted {len(df)} rows and saved to {output_file}")
