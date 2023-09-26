from llama_index.schema import Document
from typing import List

def write_documents_to_file(documents: List[Document], filename: str = 'documents.txt'):
    """
    Write the content and metadata of a list of Document objects to a file.

    :param documents: List of Document objects to be written to file
    :param filename: Name of the file to which the documents will be written
    """
    with open(filename, 'w') as file:
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

# Example usage:
# documents = [Document(...), Document(...), ...]  # Assuming you have a list of Document objects
# write_documents_to_file(documents)
