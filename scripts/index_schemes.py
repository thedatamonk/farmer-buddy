#!/usr/bin/env python3
"""Script to index government scheme PDF documents."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kisan.core.config import get_settings
from kisan.core.logging import logger, setup_logging
from kisan.modules.schemes.indexer import SchemeIndexer
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService


async def main():
    """Index all PDFs in the data/schemes directory."""
    setup_logging()

    settings = get_settings()
    logger.info("Starting scheme indexing")

    # Check for PDFs
    schemes_dir = Path(__file__).parent.parent / "data" / "schemes"
    if not schemes_dir.exists():
        logger.error(f"Schemes directory not found: {schemes_dir}")
        print(f"Please create the directory and add PDF files: {schemes_dir}")
        sys.exit(1)

    pdf_files = list(schemes_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in schemes directory")
        print(f"No PDF files found in: {schemes_dir}")
        print("Please add PDF documents about government schemes to index.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF files to index:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Initialize services
    llm_service = LLMService(settings)
    vectordb_service = VectorDBService(settings)

    # Create indexer and process
    indexer = SchemeIndexer(llm_service, vectordb_service, settings)

    print("\nIndexing documents...")
    result = await indexer.index_directory(schemes_dir)

    print(f"\n{'='*50}")
    print("Indexing Complete!")
    print(f"{'='*50}")
    print(f"Documents processed: {result.documents_processed}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Collection: {result.collection_name}")
    print(f"Success: {result.success}")

    if result.errors:
        print("\nErrors encountered:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
