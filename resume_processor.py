from llama_parse import LlamaParse
from pathlib import Path
import os
import logging
from typing import Optional, Dict, Any
import tempfile
import requests

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeProcessor:
    def __init__(self, storage_dir: str = "resume_index", llama_cloud_api_key: str = None):
        """
        Initialize resume processor.
        
        Args:
            storage_dir: Directory where the processed index will be stored
            llama_cloud_api_key: API key for Llama Cloud services
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.llama_cloud_api_key = llama_cloud_api_key
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
    def process_file(self, file_input: str) -> Dict[str, Any]:
        """
        Process a resume file and create searchable index.
        
        Args:
            file_input: Local file path or Google Drive URL
            
        Returns:
            Dict containing status and any error messages
        """
        # Initialize file_path to None to avoid UnboundLocalError
        file_path = None
        
        try:
            # Check for API key
            if not self.llama_cloud_api_key:
                return {
                    "success": False,
                    "error": "Llama Cloud API key is required"
                }

            # Handle file input
            file_path = self._get_file_path(file_input)
            if not file_path:
                return {
                    "success": False,
                    "error": "Failed to obtain file"
                }

            # Parse document
            try:
                documents = LlamaParse(
                    api_key=self.llama_cloud_api_key,
                    result_type='markdown',
                    system_prompt="this is a resume, gather related facts together and format it as bullet points with header"
                ).load_data(file_path)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to parse document: {str(e)}"
                }

            # Create index
            try:
                index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=self.embed_model
                )
                index.storage_context.persist(persist_dir=self.storage_dir)
                
                return {
                    "success": True,
                    "index_location": str(self.storage_dir),
                    "num_nodes": len(index.ref_doc_info)
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create/save index: {str(e)}"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
        finally:
            # Clean up temporary file if it was downloaded from Google Drive
            if file_path and isinstance(file_path, str) and file_path.startswith("https://drive.google.com"):
                try:
                    Path(file_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {str(e)}")

    def _get_file_path(self, file_input: str) -> Optional[str]:
        """Get local file path from input (download if URL)."""
        # If it's a local file
        if os.path.isfile(file_input):
            return file_input
            
        # If it's a Google Drive URL
        if 'drive.google.com' in file_input:
            return self._download_drive_file(file_input)
            
        return None

    def _download_drive_file(self, url: str) -> Optional[str]:
        """Download file from Google Drive."""
        try:
            # Extract file ID
            file_id = None
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
                
            if not file_id:
                logger.error("Could not extract file ID from Google Drive URL")
                return None

            # Download file
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                return temp_file.name

        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    pass