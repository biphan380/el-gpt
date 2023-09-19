from llama_index import VectorStoreIndex, download_loader

SimpleWebPageReader = download_loader("SimpleWebPageReader")

loader = SimpleWebPageReader()
documents = loader.load_data(urls=['https://www.canlii.org/en/on/laws/stat/rso-1990-c-h19/latest/rso-1990-c-h19.html?autocompleteStr=Human&autocompletePos=1#sec5_smooth'])

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query('How long do I have to bring an application before the human rights tribunal?')
print(str(response))