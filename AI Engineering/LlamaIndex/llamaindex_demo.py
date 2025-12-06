from llama_index import SimpleDirectoryReader, StorageContext, TreeIndex, load_index_from_storage

# --- Loading data and creating the index ---
resume = SimpleDirectoryReader("Private-Data").load_data()
new_index = TreeIndex(resume).from_documents(resume)

# --- Running a query ---
query_engine = new_index.as_query_engine()
response = query_engine.query("When did X join the company?")

# --- Saving and loading the context ---
new_index.storage_context.persist()

# --- Loading the context ---
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)


# --- Chat ---
response = query_engine.chat("When did X join the company?")
print(response)
