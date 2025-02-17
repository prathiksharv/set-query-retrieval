## Introduction  

This research project investigates the performance of various retrieval models on composite queries involving logical operators such as AND, OR, NOT, AND NOT, AND AND, and OR OR. The primary objective is to evaluate whether a set-theoretic approach to retrieval, which decomposes composite queries into individual subqueries and combines results based on set operations, outperforms direct composite retrieval performed.  

Through this study, we aim to shed light on the strengths and limitations of retrieval models in handling complex query structures, providing insights into their efficacy for use cases requiring logical query reasoning.  


## Framework  

We leverage the **QUEST** framework, a benchmark for evaluating retrieval models on composite queries with implicit set operations.  

### Key Components  
- **Dataset**: 3,357 queries with crowdworker annotations for naturalness, fluency, and relevance.  
- **Data Splits**: Train, validation, and test sets for model training, tuning, and evaluation.  
- **Evaluation**: Metrics like precision, recall, and operation-specific performance to identify challenges with specific composite operations.  
- **Metadata**: Includes atomic queries, composite operations, and annotation details.  


## Dataset  

We use the QUEST dataset and in-house scraped atomic query-doc data for our experiments, which benchmarks retrieval performance on composite queries.  

### Data Processing  
We preprocess the dataset to generate a CSV with the following structure:  
- **Query**: The composite query.  
- **Positive Document**: Relevant document(s).  
- **Negative Document (if available)**: Non-relevant document(s) for tasks like contrastive learning.  

This format ensures compatibility with retrieval models and streamlines training and evaluation.  


## Experimental Setup

The experiments follow a structured flow to evaluate the performance of retrieval models on composite queries:

1. **Training the Model**  
   - Train the model using some train csv file, which includes the columns: query, positive document, and negative document (if available).  
   - Training scripts: **train.py** and **train_atomic.py**. Model checkpoints are saved during training.

2. **Generating Embeddings**  
   - Use the best model checkpoint to generate query and document embeddings for the test set.  
   - Embedding generation scripts: **create_emb.py** and **create_atomic_emb.py**. Saved embeddings are used for inference.

3. **Retrieval Inference**  
   - Perform retrieval inference by calculating the similarity between the query and document embeddings.  
   - Retrieve the top-k documents based on similarity and save the output in **jsonl** format.  
   - Inference scripts: **infer.py** and **atomic_infer.py**.

4. **Evaluation**  
   - Compare the retrieved results with the **gold standard** (QUEST **jsonl** format).  
   - Calculate metrics such as precision, recall, and F1-score using **run_eval.py** from the QUEST framework.  


## Useful Commands

### 1. Training the models

Implicit training -
```
python ./src/akshay/train.py \
--runname <runname> \
--train_data <path to quest train data csv> \
--val_data <path to quest val data csv> \
--output_dir <path to model checkpoints dir> \
--wandb_dir <path to wandb dir> \
--num_train_epochs <number of training epochs> \
--batch_size <batch size>
```

Explicit training -
```
python ./src/akshay/train_atomic.py \
--runname <runname> \
--train_data <path to train data csv> \
--output_dir <path to model checkpoints dir> \
--wandb_dir <path to wandb dir> \
--num_train_epochs <number of training epochs> \
--batch_size <batch size>
```

### 2. Generating query/doc embeddings

Composite query-doc embedding generation -
```
python ./src/akshay/create_emb.py \
--questdir <path to saved quest data dir> \
--mode <train/test> \
--runname <runname> \
--useckpt \
--ckptpath <path to the checkpoint> \
--embdir <directory to save embeddings>
```

Atomic query embedding generation -
```
python ./src/akshay/create_atomic_emb.py \
--questdir <path to saved quest data dir> \
--mode <train/test> \
--runname <runname> \
--useckpt \
--ckptpath <path to the checkpoint> \
--embdir <directory to save embeddings>
```

### 3. Run inference

Running inference of composite query embeddings -
```
python ./src/akshay/infer.py \
--questdir <path to saved quest data dir> \
--mode <train/test> \
--runname <runname> \
--k <top k documents to be retrieved> \
--embdir <directory to save embeddings> \
--outdir <directory to store jsonl outputs>
```

Running inference on atomic query embeddings -
```
python ./src/akshay/atomic_infer.py \
--questdir <path to saved quest data dir> \
--mode <train/test> \
--runname <runname> \
--k <top k documents to be retrieved> \
--embdir <directory to save embeddings> \
--norm <normalize similarity scores in [0, 1] or not> \
--notimpact <impact of not in score aggregation> \
--outdir <directory to store jsonl outputs>
```

### 4. Metric evaluation

Metric evaluation wrt quest gold standard retrieval outputs -
```
python ./eval/run_eval.py \
--gold <path to quest dataset dir>/test/test.jsonl \
--pred <path to model jsonl retrieval outputs>
```

## Experiments

We conducted four main types of experiments to evaluate retrieval models on composite queries, each with different training and testing setups.

### 1. Implicit Training; Implicit Infer
In this experiment, the model is trained and tested on composite queries, replicating the results reported in the QUEST paper. We used **BM25** and **GTR-T5-Large** models for both training and inference.

- **Training**: The model is trained on composite queries.
- **Inference**: The model is tested on composite queries, evaluating direct retrieval based on query-doc pairs.

### 2. Implicit Training; Explicit Infer
In this experiment, the model is trained composite queries and tested on composite queries using set-theoretic score aggregation.

- **Training**: The model is trained on composite queries.
- **Inference**: During testing, set-theoretic score aggregation methods are applied to combine atomic query results. The aggregation method varies based on the composite operation:
    - **NOOP**: `s`
    - **OR**: `s1 + s2 - s1*s2`
    - **AND**: `s1*s2`
    - **NOT**: `s1 * (1 - s2)`
    - **OR OR**: `s1 + s2 + s3 - (s1*s2 + s2*s3 + s3*s1) + s1*s2*s3`
    - **AND AND**: `s1*s2*s3`
    - **AND NOT**: `s1*s2*(1 - s3)`

### 3. Explicit Training; Explicit Infer
In this setup, the model is trained on **atomic queries** (using in-house-scraped data, as QUEST doesn’t provide data for atomic query-doc retrieval) and tested on composite queries using set-theoretic score aggregation.

- **Training**: The model is trained on atomic queries, directly related to documents.
- **Inference**: During testing, set-theoretic score aggregation methods are applied to combine atomic query results.

### 4. Explicit Training; Implicit Infer
Here, the model is trained on atomic queries (like the previous experiment) but tested directly on composite queries, without set-theoretic aggregation.

- **Training**: The model is trained on atomic queries using in-house data.
- **Inference**: The model is tested on composite queries directly, evaluating retrieval based on the composite query as a whole.

### 5. Implicit Training with Aggregated Loss; Explicit Test (Future Experiment)
This experiment involves training the model on composite queries with an aggregated loss based on the atomic constituents. Testing will use atomic score aggregation.

- **Training**: Composite queries are used for training, with the loss being the aggregated loss from atomic constituents.
- **Inference**: During testing, atomic score aggregation methods are applied to composite queries.


## Results

Detailed results can be found logged [here](https://docs.google.com/spreadsheets/d/16QRLzbokQ76ixj9jamXXAHlg7GvcwRASHWWaUHZlswA/edit?usp=sharing) in the `gtr-t5` sheet.

### 1. Implicit Training; Implicit Infer
This setup was able to reproduce the results reported in the QUEST paper and even slightly outperform them. The model performed well on composite queries, showing strong results across various composite operations.

- **Overall Performance**: The performance was consistent with or better than the QUEST benchmarks.
- **Snapshot**:  
   ![alt text](snapshots/image-1.png)

### 2. Implicit Training; Explicit Infer
This experiment saw a decline in performance compared to the previous setup, especially for queries involving negation or complex operations such as **NOT**, **AND AND**, and **AND NOT**. These composite operations did not perform as well as in the implicit training-infer setup.

- **Overall Performance**: Decline in performance, particularly for the operations involving negation.
- **Operation-Specific Performance**: The performance drop was most notable in **NOT**, **AND AND**, and **AND NOT** queries.  
- **Snapshot of Overall Performance**:  
   ![alt text](snapshots/image-2.png)

- **Snapshot of Operation-Specific Performance**:  
   ![alt text](snapshots/image-3.png)

### 3. Explicit Training; Explicit Infer
Training the model on explicit atomic queries (in-house-scraped data) resulted in an overall improvement in performance. This may be attributed to the larger size of the explicit training dataset compared to the QUEST data. However, **NOT** and **AND NOT** queries still performed below expectations compared to the **Implicit Training; Implicit Infer** setup.

- **Overall Performance**: Significant improvement due to larger in-house data.
- **Snapshot of Overall Performance**:  
   ![alt text](snapshots/image-4.png)

- **Snapshot of Operation-Specific Performance**:  
   ![alt text](snapshots/image-5.png)

- **Tweaked Aggregation for **NOT** and **AND NOT** Queries**:  
   After tweaking the aggregation function for **NOT** and **AND NOT** queries (reducing the contribution of the negation atomic query), the performance improved substantially.  
- **Snapshot of Performance after Tweaked Aggregation**:  
   ![alt text](snapshots/image-6.png)
   ![alt text](snapshots/image-7.png)

### 4. Explicit Training; Implicit Infer
This setup performed comparably to **Explicit Training; Explicit Infer** but excelled in **AND AND** queries. It showed that **AND AND** queries benefit from implicit inference, as the performance dropped in the **Implicit Training; Explicit Infer** setup for the same query type.

- **Overall Performance**: Comparable to **Explicit Training; Explicit Infer** with some decline in other operations but a notable improvement in **AND AND** queries.
- **Snapshot of Performance**:  
   ![alt text](snapshots/image-8.png)
   ![alt text](snapshots/image-9.png)


## Implementation Details

The training is based on the **Sentence Transformers (SBERT)** library, using the **sentence-transformers/gtr-t5-large** model. Key aspects of the implementation are as follows:

1. **Model**:  
   The **gtr-t5-large** model is used for generating embeddings for composite queries and retrieval tasks.

2. **Loss Function**:  
   The **CachedMultipleNegativesRankingLoss** is used to train the model with positives and in-batch negatives, in line with the QUEST paper. The "cached" version enables training with a higher batch size on limited compute resources, essential for reproducing QUEST’s 512 batch size on a single GPU.

3. **Training Setup**:  
   - **Trainer**: **SentenceTransformerTrainer** is used to handle training.
   - **Evaluation**: **MRREvaluationCallback** is used for performance evaluation on validation data.
   - **Early Stopping**: **EarlyStoppingCallback** halts training if the evaluation loss doesn’t improve over several steps.

4. **Compute Resources**:  
   All experiments were conducted on a single GPU, with the cached loss enabling efficient training on larger batches.


## Code Directory Structure

The codebase is organized into the following directory structure currently (over evolving):
```
dataset/                    # Dataset-related files (raw data, processed data)
├── train           # Training data (query, positive doc, negative doc)
├── val              # Validation data
├── test            # Test data
├── hardnegs            # Hard negatives data
├── filtered          # Atomic query in-house scrapped data
└── documents.jsonl  # QUEST documents data

src/akshay             # Experiment-related scripts
├── train.py             # Script to train the model
├── train_atomic.py      # Script for training on atomic queries
├── create_emb.py        # Script to generate embeddings for test set
├── create_atomic_emb.py        # Script to generate atomic embeddings for test set
├── infer.py             # Script to run retrieval inference
└── atomic_infer.py             # Script to run explicit retrieval inference

out/                  # Model output files (jsonl)
└── gtr-t5/<run-name>/test      # All the output jsonl files for that model (run)
```

Some of the bigger files (checkpoints, embeddings) are saved on my scratch space `/scratch/workspace/aneema_umass_edu-set-query/quest/`. In that directory `checkpoints` directory has all the model checkpoints based on the `run-name` and the `embeds` directory has all the generated query and doc embeddings with name `queryembs.npy` and `docembs.npy` respectively. Below is the basic directory structure of `/scratch/workspace/aneema_umass_edu-set-query/quest/` -
```
checkpoints/   # Has all the training checkpoints saved for every run
embeds/        # Has query/doc embeddings for several runs
wandb/         # Has all the wandb logs
```






