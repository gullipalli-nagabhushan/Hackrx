#!/usr/bin/env python3
"""
Batch Query Runner

Reads requests from a text file (multiple JSON objects) or a Python list,
executes document Q&A using the project's query pipeline, and writes
request-response pairs to an output file.

Usage:
  python batch_query_runner.py --input Questions_in_JSON.txt --output results.json

Environment:
  Requires OPENAI_API_KEY and PINECONE_API_KEY for full functionality.
  Runs in mock/fallback mode if keys are missing.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from document_processor import DocumentProcessor, DocumentChunk
from query_engine import QueryEngine
from vector_store import PineconeVectorStore
from database import DatabaseManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class BatchQueryService:
    """High-level service to process document queries in batch."""

    def __init__(self) -> None:
        self.document_processor = DocumentProcessor()
        self.vector_store = PineconeVectorStore()
        self.query_engine = QueryEngine()
        self.database_manager = DatabaseManager()
        self._initialized = False

        # Cache processed documents to avoid re-ingestion on repeats
        self._processed_documents: Dict[str, bool] = {}

    async def initialize(self) -> None:
        if self._initialized:
            return
        await asyncio.gather(
            self.vector_store.initialize(),
            self.query_engine.initialize(),
            self.database_manager.initialize(),
        )
        self._initialized = True

    async def _ensure_document_processed(self, document_url: str) -> None:
        if self._processed_documents.get(document_url):
            return

        # Clear vector cache before loading a new document to prevent mixing
        await self.vector_store.clear_cache()

        chunks = await self.document_processor.process_document(document_url)
        if chunks:
            # Normalize to expected type with 'content' attribute
            normalized_chunks: List[DocumentChunk] = []
            for c in chunks:
                content = getattr(c, "content", None)
                if content is None:
                    content = getattr(c, "page_content", None)
                if content is None:
                    content = str(c)
                metadata = getattr(c, "metadata", {}) or {}
                normalized_chunks.append(DocumentChunk(content=content, metadata=metadata))

            await self.vector_store.add_documents(normalized_chunks)
            self._processed_documents[document_url] = True
            logger.info("Document processed and added to vector store: %s", document_url)
        else:
            logger.warning("No chunks extracted for document: %s", document_url)
            self._processed_documents[document_url] = True  # Avoid retry loop

    async def answer_questions(self, document_url: str, questions: List[str]) -> List[str]:
        await self._ensure_document_processed(document_url)

        async def answer_one(q: str) -> str:
            try:
                embedding = await self.query_engine.generate_query_embedding(q)
                relevant_chunks = await self.vector_store.similarity_search(
                    embedding, top_k=10, query_text=q, document_url=document_url
                )
                answer = await self.query_engine.generate_answer(q, relevant_chunks, document_url)
                return answer
            except Exception as e:  # Defensive: return a concise message per question
                logger.error("Failed answering question: %s", e)
                return f"Error: {str(e)[:120]}"

        tasks = [answer_one(q) for q in questions]
        return await asyncio.gather(*tasks)


def parse_requests_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse multiple JSON objects from a text file into request dicts.

    Expects objects of the form: {"documents": "...", "questions": [ ... ]}
    Ignores any objects that lack required keys or that only contain answers.
    """
    # Greedy-safe object extraction
    objects = re.findall(r"\{[\s\S]*?\}", text)
    requests: List[Dict[str, Any]] = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "documents" in obj and "questions" in obj:
            if isinstance(obj["questions"], list):
                requests.append({"documents": obj["documents"], "questions": obj["questions"]})
    return requests


def load_requests(input_path: Optional[str], inline_requests: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if inline_requests:
        return inline_requests
    if not input_path:
        raise ValueError("Either inline requests or --input file path must be provided")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_requests_from_text(text)


def write_results(output_path: str, results: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d request-response pairs to %s", len(results), output_path)


async def run_batch(requests: List[Dict[str, Any]], output_path: str) -> None:
    service = BatchQueryService()
    await service.initialize()

    results: List[Dict[str, Any]] = []

    for req in requests:
        documents = req.get("documents", "")
        questions = req.get("questions", [])
        if not documents or not questions:
            logger.warning("Skipping invalid request without documents/questions: %s", req)
            continue

        logger.info("Processing %d questions for: %s", len(questions), documents)
        answers = await service.answer_questions(documents, questions)

        results.append({
            "request": {
                "documents": documents,
                "questions": questions,
            },
            "answers": answers,
        })

    write_results(output_path, results)

    # Cleanup to avoid unclosed session warnings
    try:
        await service.document_processor.cleanup()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process requests and write request-response pairs")
    parser.add_argument("--input", type=str, default="Questions_in_JSON.txt", help="Path to input text file")
    parser.add_argument("--output", type=str, default="batch_results.json", help="Path to output JSON file")
    parser.add_argument("--requests-file", type=str, default=None, help="Optional JSON file containing a list of {documents, questions}")
    args = parser.parse_args()

    # Support running without an input file if developer modifies inline list below
    inline_requests: Optional[List[Dict[str, Any]]] = None
    # Example format:
    # inline_requests = [
    #     {
    #         "documents": "https://example.com/doc.pdf",
    #         "questions": ["Q1?", "Q2?"]
    #     }
    # ]

    if args.requests_file:
        with open(args.requests_file, "r", encoding="utf-8") as rf:
            inline_requests = json.load(rf)

    requests = load_requests(args.input, inline_requests)
    if not requests:
        raise SystemExit("No valid requests found to process.")

    asyncio.run(run_batch(requests, args.output))


if __name__ == "__main__":
    main()


