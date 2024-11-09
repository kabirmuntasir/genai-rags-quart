# backend/document_processor.py
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
import logging
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000")),
                 chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200")),
                 similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))):
        # Directory setup
        self.base_dir = "documents"
        self.input_dir = f"{self.base_dir}/input"
        self.processed_dir = f"{self.base_dir}/processed"
        self.failed_dir = f"{self.base_dir}/failed"
        
        for dir_path in [self.input_dir, self.processed_dir, self.failed_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize Azure OpenAI
        if not all([os.getenv("AZURE_OPENAI_KEY"), os.getenv("AZURE_OPENAI_ENDPOINT")]):
            raise ValueError("Azure OpenAI credentials not found in environment")
            
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
        )

        # Initialize collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=os.getenv("COLLECTION_NAME", "webpage_docs")
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with error handling"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap"""
        return self.text_splitter.split_text(text)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Azure OpenAI"""
        try:
            response = self.azure_client.embeddings.create(
                model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logging.error(f"Error getting embeddings: {str(e)}")
            raise

    def calculate_similarity(self, chunk1: str, chunk2: str) -> float:
        """Calculate cosine similarity between two chunks"""
        try:
            vec1 = self._get_embeddings([chunk1])[0]
            vec2 = self._get_embeddings([chunk2])[0]
            return cosine_similarity([vec1], [vec2])[0][0]
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def remove_similar_chunks(self, chunks: List[str]) -> List[str]:
        """Remove chunks that are too similar"""
        unique_chunks = []
        for chunk in chunks:
            is_unique = True
            for unique_chunk in unique_chunks:
                similarity = self.calculate_similarity(chunk, unique_chunk)
                if similarity > self.similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_chunks.append(chunk)
        return unique_chunks

    def process_document(self, file_path: str) -> bool:
        """Process a single document"""
        try:
            filename = os.path.basename(file_path)
            logging.info(f"Processing {filename}")
            
            text = self.extract_text_from_pdf(file_path)
            if not text:
                raise Exception("No text extracted from PDF")
            
            chunks = self.chunk_text(text)
            logging.info(f"Created {len(chunks)} initial chunks")
            
            unique_chunks = self.remove_similar_chunks(chunks)
            logging.info(f"Reduced to {len(unique_chunks)} unique chunks")
            
            embeddings = self._get_embeddings(unique_chunks)
            
            self.collection.add(
                embeddings=embeddings,
                documents=unique_chunks,
                ids=[f"{filename}_{i}" for i in range(len(unique_chunks))],
                metadatas=[{
                    "source": filename,
                    "chunk_index": i,
                    "processed_date": datetime.now().isoformat()
                } for i in range(len(unique_chunks))]
            )
            
            os.rename(
                file_path,
                os.path.join(self.processed_dir, filename)
            )
            return True
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            try:
                os.rename(
                    file_path,
                    os.path.join(self.failed_dir, os.path.basename(file_path))
                )
            except Exception as move_error:
                logging.error(f"Error moving failed file: {str(move_error)}")
            return False

    def process_directory(self) -> Dict[str, int]:
        """Process all documents in input directory"""
        stats = {"processed": 0, "failed": 0}
        
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(self.input_dir, filename)
                success = self.process_document(file_path)
                if success:
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1

        return stats

    def query_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the document collection"""
        try:
            query_embedding = self._get_embeddings([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return [{
                "text": doc,
                "metadata": meta
            } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        except Exception as e:
            logging.error(f"Error querying documents: {str(e)}")
            return []

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Azure OpenAI"""
        try:
            system_message = """You are a helpful AI assistant. Use the provided context to answer questions.
            If you cannot find the answer in the context, say so."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
            
            deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o").replace("azure/", "")
            
            response = self.azure_client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error generating the response."