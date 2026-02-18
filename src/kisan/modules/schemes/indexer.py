"""Document indexer for government schemes."""

from pathlib import Path

from kisan.core.config import Settings
from kisan.core.exceptions import IndexingError
from kisan.core.logging import logger
from kisan.modules.schemes.embeddings import SchemeEmbeddings
from kisan.schemas.scheme import IndexingResult
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService
from kisan.utils.pdf import process_pdf_for_indexing


class SchemeIndexer:
    """Index government scheme documents for RAG retrieval."""

    def __init__(
        self,
        llm_service: LLMService,
        vectordb_service: VectorDBService,
        settings: Settings,
    ):
        self.embeddings = SchemeEmbeddings(llm_service)
        self.vectordb = vectordb_service
        self.settings = settings

    async def index_pdf(self, pdf_path: str | Path) -> IndexingResult:
        """Index a single PDF document."""
        pdf_path = Path(pdf_path)

        try:
            # Delete existing chunks for this PDF to make re-indexing idempotent
            self.vectordb.delete_by_source(pdf_path.name)

            # Extract and chunk the PDF
            chunks = process_pdf_for_indexing(
                pdf_path,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )

            if not chunks:
                return IndexingResult(
                    documents_processed=1,
                    chunks_created=0,
                    collection_name=self.settings.qdrant_collection,
                    success=False,
                    errors=["No content extracted from PDF"],
                )

            # Generate embeddings
            embeddings = await self.embeddings.embed_documents(chunks)

            # Index in vector database
            self.vectordb.index_documents(
                embeddings=embeddings,
                documents=chunks,
            )

            logger.info(f"Indexed {len(chunks)} chunks from {pdf_path.name}")

            return IndexingResult(
                documents_processed=1,
                chunks_created=len(chunks),
                collection_name=self.settings.qdrant_collection,
                success=True,
                errors=[],
            )

        except Exception as e:
            logger.error(f"Failed to index {pdf_path}: {e}")
            return IndexingResult(
                documents_processed=1,
                chunks_created=0,
                collection_name=self.settings.qdrant_collection,
                success=False,
                errors=[str(e)],
            )

    async def index_directory(self, directory: str | Path) -> IndexingResult:
        """Index all PDF files in a directory."""
        directory = Path(directory)

        if not directory.exists():
            raise IndexingError(f"Directory not found: {directory}")

        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            return IndexingResult(
                documents_processed=0,
                chunks_created=0,
                collection_name=self.settings.qdrant_collection,
                success=True,
                errors=["No PDF files found in directory"],
            )

        total_chunks = 0
        errors = []

        for pdf_path in pdf_files:
            result = await self.index_pdf(pdf_path)
            total_chunks += result.chunks_created
            errors.extend(result.errors)

        logger.info(
            f"Indexed {len(pdf_files)} PDFs with {total_chunks} total chunks"
        )

        return IndexingResult(
            documents_processed=len(pdf_files),
            chunks_created=total_chunks,
            collection_name=self.settings.qdrant_collection,
            success=len(errors) == 0,
            errors=errors,
        )

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self.vectordb.delete_collection()
        logger.info("Cleared scheme index")
