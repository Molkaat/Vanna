# Testing VANNA

## Testing v1.0.0

### Hyperparameters
- Number of DDLs = 14
- Documentation = 0
- Sql question pairs = 116
- n_of_ddls to retreive = 2
- n_documentation to retrieve = 2
- n_of_sql to retrieve = 2
- LLM = gpt-3.5-turbo
- Temperature = 0.7
- Embdedding model = https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
- underlying vector database = ChromaDB
- System prompt = EMPTY
- Chunk_size = Each statement is treated as a chunck, each documentation is treadted as a chunk, each ddl is a single chunk (GPT's answer from web).

### Outcomes
- Ambiguos questions were replaced to include important context.
- The limit clause were removed from the sql statements so that the autonomous compare process between the ground truth sql and the model-generated sql is more sensible.