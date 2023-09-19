from llama_index import VectorStoreIndex, download_loader

SimpleWebPageReader = download_loader("SimpleWebPageReader")

loader = SimpleWebPageReader()
documents = loader.load_data(urls=['https://www.canlii.org/en/on/laws/stat/rso-1990-c-h19/latest/rso-1990-c-h19.html?autocompleteStr=Human&autocompletePos=1#sec5_smooth'])

index = VectorStoreIndex.from_documents(documents)

with open('documents.txt', 'w') as file:
    for doc in documents:
        # Write page_content
        file.write("Page Content:\n")
        file.write(doc.get_content())
        file.write("\n\n")

        # Write metadata
        file.write("Metadata:\n")
        for key, value in doc.metadata.items():
            file.write(f"{key}: {value}\n")

        # Add a separator between documents
        file.write("\n" + "-"*50 + "\n\n")