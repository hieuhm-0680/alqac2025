#!/usr/bin/env python3
"""
Script for answering legal multiple choice and yes/no questions using QwenReranker.
This approach uses reranking scores to determine the most likely answer by calculating 
relevance between questions, context, and possible answers.
"""

import torch
import json
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_instruction(instruction, query, doc):
    """Format the instruction for the reranker model."""
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
    """Qwen3-Reranker-8B model wrapper for answer ranking."""

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

        # Question-specific instructions
        self.instruction_multiple_choice = """Judge whether the provided answer choice is the best and most accurate response to the given legal question according to the context provided.
Note that the answer must strictly be "yes" or "no".  

Đánh giá xem đáp án được cung cấp có phải là câu trả lời chính xác và phù hợp nhất cho câu hỏi pháp luật theo ngữ cảnh được cung cấp hay không.
Chỉ trả lời "yes" hoặc "no", không giải thích thêm."""

        self.instruction_true_false = """Judge whether the given legal statement is TRUE according to the legal context provided.
Note that the answer must strictly be "yes" or "no".  

Đánh giá xem nhận định pháp luật đã cho có ĐÚNG theo ngữ cảnh pháp luật được cung cấp hay không.
Chỉ trả lời "yes" hoặc "no", không giải thích thêm."""

        self.instruction_true_false_negative = """Judge whether the given legal statement is FALSE according to the legal context provided.
Note that the answer must strictly be "yes" or "no".  

Đánh giá xem nhận định pháp luật đã cho có SAI theo ngữ cảnh pháp luật được cung cấp hay không.
Chỉ trả lời "yes" hoặc "no", không giải thích thêm."""

    def score_batch(self, queries: List[str], documents: List[str], instruction: str,
                    batch_size: int = 8) -> List[float]:
        """Score a batch of query-document pairs.

        Args:
            queries: List of query strings
            documents: List of document strings
            instruction: The instruction to use for reranking
            batch_size: Batch size for processing

        Returns:
            List of relevance scores
        """
        if len(queries) != len(documents):
            raise ValueError("Number of queries and documents must match")

        all_scores = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
            batch_queries = queries[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]

            # Format input pairs
            pairs = [
                format_instruction(instruction, query, doc)
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


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def rerank_multiple_choice(questions: List[Dict[str, Any]], reranker: QwenReranker,
                           output_file: str, batch_size: int = 8) -> None:
    """Rerank multiple choice questions and save results.

    For each question, score each choice against the question + context.
    The choice with the highest score is the predicted answer.
    """
    results = []

    for question in tqdm(questions, desc="Processing multiple choice questions"):
        if question['question_type'] != "Trắc nghiệm" or 'choices' not in question:
            continue

        question_text = question['text']
        context = question.get('context', '')
        choices = question['choices']
        question_id = question['question_id']

        # Create a combined query with question and context
        query = f"{question_text}\n\n{context}" if context else question_text

        # Score each choice
        choice_scores = {}
        choice_queries = []
        choice_documents = []

        for option, choice_text in choices.items():
            choice_queries.append(query)
            choice_documents.append(choice_text)

        # Get scores for all choices
        scores = reranker.score_batch(
            choice_queries,
            choice_documents,
            reranker.instruction_multiple_choice,
            batch_size
        )

        # Associate scores with choices
        for i, (option, choice_text) in enumerate(choices.items()):
            choice_scores[option] = scores[i]

        # Find the choice with the highest score
        best_choice = max(choice_scores.items(), key=lambda x: x[1])
        predicted_answer = best_choice[0]

        # Add results to output
        result = {
            "question_id": question_id,
            "question_type": "Trắc nghiệm",
            "text": question_text,
            "choice_scores": choice_scores,
            "predicted_answer": predicted_answer
        }
        results.append(result)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved multiple choice results to {output_file}")


def rerank_true_false(questions: List[Dict[str, Any]], reranker: QwenReranker,
                      output_file: str, batch_size: int = 8) -> None:
    """Rerank true/false questions and save results.

    For each question, score it with both "true" and "false" statements.
    The statement with the highest score is the predicted answer.
    """
    results = []

    for question in tqdm(questions, desc="Processing true/false questions"):
        if question['question_type'] != "Đúng/Sai":
            continue

        question_text = question['text']
        context = question.get('context', '')
        question_id = question['question_id']

        # Create queries
        query_text = f"{question_text}\n\n{context}" if context else question_text

        # Create a list of queries and documents
        queries = [query_text, query_text]
        documents = ["Nhận định này đúng.", "Nhận định này sai."]

        # Get scores using different instructions for true and false statements
        true_score = reranker.score_batch(
            [query_text],
            ["Nhận định này đúng."],
            reranker.instruction_true_false,
            batch_size
        )[0]

        false_score = reranker.score_batch(
            [query_text],
            ["Nhận định này sai."],
            reranker.instruction_true_false_negative,
            batch_size
        )[0]

        # Determine answer based on scores
        scores = {"Đúng": true_score, "Sai": false_score}
        predicted_answer = "Đúng" if true_score > false_score else "Sai"

        # Add results to output
        result = {
            "question_id": question_id,
            "question_type": "Đúng/Sai",
            "text": question_text,
            "scores": scores,
            "predicted_answer": predicted_answer
        }
        results.append(result)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved true/false results to {output_file}")


def process_dataset(input_file: str, output_dir: str, reranker: QwenReranker, batch_size: int = 8) -> None:
    """Process a dataset containing questions and save rerank results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading questions from {input_file}...")
    questions = load_data(input_file)

    # Split questions by type
    multiple_choice_questions = [
        q for q in questions if q['question_type'] == "Trắc nghiệm" and 'choices' in q]
    true_false_questions = [
        q for q in questions if q['question_type'] == "Đúng/Sai"]

    print(f"Found {len(multiple_choice_questions)} multiple choice questions")
    print(f"Found {len(true_false_questions)} true/false questions")

    # Process multiple choice questions
    if multiple_choice_questions:
        output_mc = os.path.join(output_dir, "multiple_choice_results.json")
        rerank_multiple_choice(multiple_choice_questions,
                               reranker, output_mc, batch_size)

    # Process true/false questions
    if true_false_questions:
        output_tf = os.path.join(output_dir, "true_false_results.json")
        rerank_true_false(true_false_questions, reranker,
                          output_tf, batch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Answer legal multiple choice and yes/no questions using QwenReranker")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSON file containing questions")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Output directory for result files")
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

    # Process the dataset
    process_dataset(args.input, args.output_dir, reranker, args.batch_size)


if __name__ == "__main__":
    main()
