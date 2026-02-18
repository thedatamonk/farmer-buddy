#!/usr/bin/env python3
"""Script to index government scheme PDF documents."""

import argparse
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
    parser = argparse.ArgumentParser(description="Index scheme PDFs")
    parser.add_argument(
        "--force", action="store_true", help="Re-index all PDFs, replacing existing chunks"
    )
    args = parser.parse_args()

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

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    if args.force:
        print("\n--force flag set: will re-index all PDFs")

    # Initialize services
    llm_service = LLMService(settings)
    vectordb_service = VectorDBService(settings)
    vectordb_service.ensure_collection()

    # Create indexer and process per-PDF
    indexer = SchemeIndexer(llm_service, vectordb_service, settings)

    total_chunks = 0
    indexed_count = 0
    skipped_count = 0
    errors = []

    print("\nProcessing documents...")
    for pdf in pdf_files:
        existing = vectordb_service.count_by_source(pdf.name)
        if existing > 0 and not args.force:
            print(f"  Skipping {pdf.name} ({existing} chunks already indexed)")
            skipped_count += 1
            continue

        action = "Re-indexing" if existing > 0 else "Indexing"
        print(f"  {action} {pdf.name}...")
        result = await indexer.index_pdf(pdf)
        total_chunks += result.chunks_created
        indexed_count += 1
        errors.extend(result.errors)

    print(f"\n{'='*50}")
    print("Indexing Complete!")
    print(f"{'='*50}")
    print(f"PDFs indexed: {indexed_count}")
    print(f"PDFs skipped: {skipped_count}")
    print(f"Chunks created: {total_chunks}")
    print(f"Collection: {settings.qdrant_collection}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
