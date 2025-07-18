#!/usr/bin/env python3
"""
Postprocess script for reranking retrieval results using Qwen3-Reranker-8B model.
This script takes JSON files containing retrieval results and reranks them based on 
legal relevance scores.
"""

import torch
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def format_instruction(instruction, query, doc):
    """Format the instruction for the reranker model."""
    if instruction is None:
        raise ValueError("Instruction cannot be None")
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def process_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length):
    """Process input pairs for the model."""
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True,
                           return_tensors="pt", max_length=max_length)
    return inputs


@torch.no_grad()
def compute_logits(inputs, model, token_true_id, token_false_id):
    """Compute relevance scores for the input pairs."""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


class QwenReranker:
    """Qwen3-Reranker-8B model wrapper for legal document reranking."""

    def __init__(self, model_name="Qwen/Qwen3-Reranker-8B", device=None, use_flash_attention=False):
        """Initialize the reranker model.

        Args:
            model_name: Name of the model to use
            device: Device to run the model on (auto-detect if None)
            use_flash_attention: Whether to use flash attention for better performance
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left')

        print(f"Loading model on {self.device}...")
        if use_flash_attention and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name).to(self.device).eval()

        # Set up special tokens
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        # Set up prompt template
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(
            suffix, add_special_tokens=False)

        # Legal-specific instruction
        self.instruction = """Judge whether the provided legal article correctly answers or directly applies to the legal query given below.  
Note that the answer must strictly be "yes" or "no".  

Đánh giá xem điều luật sau có quy chiếu chính xác hoặc áp dụng trực tiếp cho câu hỏi pháp luật bên dưới hay không.  
Chỉ trả lời "yes" hoặc "no", không giải thích thêm."""

    def rerank_batch(self, queries: List[str], documents: List[str], batch_size: int = 8) -> List[float]:
        """Rerank a batch of query-document pairs.

        Args:
            queries: List of query strings
            documents: List of document strings
            batch_size: Batch size for processing

        Returns:
            List of relevance scores
        """
        if len(queries) != len(documents):
            raise ValueError("Number of queries and documents must match")

        all_scores = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Reranking batches"):
            batch_queries = queries[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]

            # Format input pairs
            pairs = [
                format_instruction(self.instruction, query, doc)
                for query, doc in zip(batch_queries, batch_docs)
            ]

            # Process inputs
            inputs = process_inputs(pairs, self.tokenizer, self.prefix_tokens,
                                    self.suffix_tokens, self.max_length)

            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            # Compute scores
            scores = compute_logits(
                inputs, self.model, self.token_true_id, self.token_false_id)
            all_scores.extend(scores)

        return all_scores


def rerank_retrieval_results(input_file: str, output_file: str, reranker: QwenReranker,
                             batch_size: int = 8) -> None:
    """Rerank retrieval results from a JSON file.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file  
        reranker: QwenReranker instance
        batch_size: Batch size for processing
    """
    print(f"Loading retrieval results from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} questions...")

    for item in tqdm(data, desc="Processing questions"):
        question_text = item.get('text', '')
        if not question_text and 'question_id' in item:
            # If no text field, this might be from retrieval_results.json format
            # We'll need to look up the question text from another source
            print(
                f"Warning: No question text found for {item.get('question_id', 'unknown')}")
            continue

        relevant_articles = item.get('relevant_articles', [])

        if not relevant_articles:
            continue

        # Prepare queries and documents
        queries = [question_text] * len(relevant_articles)
        documents = [article['text'] for article in relevant_articles]

        # Get reranking scores
        scores = reranker.rerank_batch(queries, documents, batch_size)

        # Add scores to articles and sort by score (descending)
        for article, score in zip(relevant_articles, scores):
            article['rerank_score'] = score

        # Sort by rerank score in descending order
        relevant_articles.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Update the item
        item['relevant_articles'] = relevant_articles

    # Save reranked results
    print(f"Saving reranked results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Reranking completed!")


def process_folder(input_folder: str, output_folder: str, reranker: QwenReranker,
                   batch_size: int = 8) -> None:
    """Process all JSON files in a folder.

    Args:
        input_folder: Path to input folder containing JSON files
        output_folder: Path to output folder for reranked files
        reranker: QwenReranker instance
        batch_size: Batch size for processing
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        output_file = output_path / f"reranked_{json_file.name}"
        print(f"\nProcessing {json_file.name}...")

        try:
            rerank_retrieval_results(str(json_file), str(
                output_file), reranker, batch_size)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Rerank retrieval results using Qwen3-Reranker-8B")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSON file or folder containing JSON files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSON file or folder for reranked results")
    parser.add_argument("--batch_size", "-b", type=int, default=8,
                        help="Batch size for processing (default: 8)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run model on (cuda/cpu, auto-detect if not specified)")
    parser.add_argument("--flash_attention", action="store_true",
                        help="Use flash attention for better performance (requires CUDA)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-8B",
                        help="Model name to use for reranking")

    args = parser.parse_args()

    # Initialize reranker
    print("Initializing Qwen3-Reranker...")
    reranker = QwenReranker(
        model_name=args.model_name,
        device=args.device,
        use_flash_attention=args.flash_attention
    )

    # Check if input is file or folder
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        rerank_retrieval_results(
            args.input, args.output, reranker, args.batch_size)
    elif input_path.is_dir():
        # Process folder
        process_folder(args.input, args.output, reranker, args.batch_size)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return


if __name__ == "__main__":
    main()
