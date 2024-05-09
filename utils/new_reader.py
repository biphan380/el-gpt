from llama_index.core import SimpleDirectoryReader
from typing import Callable, Dict, Generator, List, Optional, Type
from llama_index.core.schema import Document
from llama_index.legacy.readers.file.base import DEFAULT_FILE_READER_CLS

# Looks like SimpleDirectoryReader was changed a lot 
# If we want each pdf to be its own Document object, 
# We need to extend the new load_data function, which calls load_file
# Or we extend the PDFReader
# Hold off on this for now because customizing data ingestion
# may not be the best ROI right now. 
class CustomDirectoryReader(SimpleDirectoryReader):
    
    def load_data(self) -> List[Document]:
        """We attempt to Load data from the input directory.
        We want each pdf to be its own Document object, not be 
        split up into N Documents objects for N pages.
        
        Returns:
            List[Document]: A list of Document objects. But each file is 
            its own Document."""
        
        documents = []
        supported_suffixes = SimpleDirectoryReader.supported_suffix_fn().keys()

        for input_file in self.input_files:
            metadata: Optional[dict] = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            # Add the title of the file to the metadata
            title = input_file.stem  # This gives the file name without the extension
            if metadata is None:
                metadata = {'title': title}
            else:
                metadata.update({'title': title})

            file_suffix = input_file.suffix.lower()
            if (file_suffix in supported_suffixes or file_suffix in self.file_extractor):
                # use file readers
                if file_suffix not in self.file_extractor:
                    # instantiate file reader if not only
                    reader_cls = self.supported_suffix_fn()[file_suffix]  # Get the reader class from the method
                    self.file_extractor[file_suffix] = reader_cls()

                reader = self.file_extractor[file_suffix]
                docs = reader.load_data(input_file, extra_info=metadata)

                # Combine the texts of all the documents into one single document
                combined_text = ' '.join(doc.text for doc in docs)
                combined_document = Document(text=combined_text, metadata=metadata or {})
                if self.filename_as_id:
                    combined_document.id_ = str(input_file)

                documents.append(combined_document)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors, encoding=self.encoding) as f:
                    data = f.read()

                doc = Document(text=data, metadata=metadata or {})
                if self.filename_as_id:
                    doc.id_ = str(input_file)

                documents.append(doc)

        return documents
