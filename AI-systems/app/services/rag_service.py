import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class ResumeRAGService:
    def __init__(self):
        # Using a fast, highly-rated Hugging Face embedding model
        # all-MiniLM-L6-v2 is great for quick, local CPU/GPU execution
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Local directory to store the Chroma vector database
        self.persist_directory = "./chroma_db"
        
        # Text splitter to break the resume into logical chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def ingest_resume(self, pdf_file_path: str, candidate_id: str) -> bool:
        """
        Reads a candidate's PDF resume, splits it into chunks, 
        and saves it to the vector database tagged with their ID.
        """
        try:
            # 1. Load the PDF
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()

            # 2. Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # 3. Add metadata so we only search THIS candidate's resume later
            for chunk in chunks:
                chunk.metadata["candidate_id"] = candidate_id

            # 4. Embed and store in Chroma
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            return True
        except Exception as e:
            print(f"Error ingesting resume: {e}")
            return False

    def query_candidate_context(self, candidate_id: str, query: str) -> str:
        """
        Searches a specific candidate's resume for relevant information.
        """
        # Load the existing database
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        
        # Perform a similarity search, filtering ONLY for this candidate
        results = db.similarity_search(
            query=query,
            k=3, # Return the top 3 most relevant chunks
            filter={"candidate_id": candidate_id}
        )
        
        if not results:
            return "No relevant information found in the candidate's resume for this topic."
            
        # Combine the retrieved chunks into a single text block for the LLM to read
        context = "\n---\n".join([doc.page_content for doc in results])
        return context

# Instantiate for use in your tools
resume_db = ResumeRAGService()