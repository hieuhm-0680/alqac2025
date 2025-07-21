import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def setup_model(model_name_or_path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """Load the model and tokenizer."""
    print(f"Loading model {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Use torch_dtype to load in 16-bit precision if available
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model loaded on {device} with torch.float16")
    else:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        print(f"Model loaded on CPU")

    return model, tokenizer


def generate_true_false_prompt(question: str, context: str) -> str:
    """Generate prompt for True/False questions."""
    human_message = f"""Dưới đây là một câu hỏi Đúng/Sai trong lĩnh vực pháp luật. Hãy trả lời "Đúng" hoặc "Sai" dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh: 
{context}

Câu hỏi: {question}

Câu trả lời phải là "Đúng" hoặc "Sai". Hãy phân tích kỹ ngữ cảnh pháp luật và đưa ra câu trả lời chính xác."""

    prompt = f"""Bạn là một trợ lý AI trong lĩnh vực pháp luật. Người dùng sẽ cung cấp cho bạn một nhiệm vụ. Mục tiêu của bạn là hoàn thành nhiệm vụ theo đúng yêu cầu của người dùng.

### Human: {human_message}

### Assistant:"""
    return prompt


def generate_multiple_choice_prompt(question: str, context: str, choices: Dict[str, str]) -> str:
    """Generate prompt for multiple choice questions."""
    choices_text = "\n".join([f"{k}: {v}" for k, v in choices.items()])

    human_message = f"""Dưới đây là một câu hỏi trắc nghiệm trong lĩnh vực pháp luật. Hãy chọn một đáp án đúng (A, B, C hoặc D) dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh: 
{context}

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Câu trả lời phải là một trong các lựa chọn sau: "A", "B", "C" hoặc "D". Hãy phân tích kỹ ngữ cảnh pháp luật và đưa ra đáp án chính xác nhất."""

    prompt = f"""Bạn là một trợ lý AI trong lĩnh vực pháp luật. Người dùng sẽ cung cấp cho bạn một nhiệm vụ. Mục tiêu của bạn là hoàn thành nhiệm vụ theo đúng yêu cầu của người dùng.

### Human: {human_message}

### Assistant:"""
    return prompt


def generate_free_text_prompt(question: str, context: str) -> str:
    """Generate prompt for free-text questions."""
    human_message = f"""Dưới đây là một câu hỏi tự luận trong lĩnh vực pháp luật. Hãy trả lời câu hỏi dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh: 
{context}

Câu hỏi: {question}

Hãy phân tích kỹ ngữ cảnh pháp luật và đưa ra câu trả lời chính xác, ngắn gọn và đầy đủ."""

    prompt = f"""Bạn là một trợ lý AI trong lĩnh vực pháp luật. Người dùng sẽ cung cấp cho bạn một nhiệm vụ. Mục tiêu của bạn là hoàn thành nhiệm vụ một cách trung thực nhất có thể. Trong khi thực hiện nhiệm vụ, hãy suy nghĩ từng bước một và biện minh cho các bước của bạn.


### Human: {human_message}

### Assistant:"""
    return prompt


def extract_answer_from_response(response: str, question_type: str) -> str:
    """Extract the actual answer from the model's response."""
    response = response.strip()

    if question_type == "Đúng/Sai":
        # Look for "Đúng" or "Sai" in the response
        if "đúng" in response.lower() and "sai" not in response.lower():
            return "Đúng"
        elif "sai" in response.lower() and "đúng" not in response.lower():
            return "Sai"

        words = response.split()
        if words:
            first_word = words[0].lower()
            if first_word == "đúng":
                return "Đúng"
            elif first_word == "sai":
                return "Sai"

        # Look for phrases indicating true/false
        if "không đúng" in response.lower() or "không chính xác" in response.lower():
            return "Sai"
        if "là đúng" in response.lower() or "chính xác" in response.lower():
            return "Đúng"

        return "Đúng" if "đúng" in response.lower() else "Sai"

    elif question_type == "Trắc nghiệm":
        for option in ["A", "B", "C", "D"]:
            patterns = [
                f"đáp án {option}", f"đáp án là {option}",
                f"chọn {option}", f"lựa chọn {option}",
                f"câu trả lời là {option}", f"câu trả lời: {option}", 
            ]
            for pattern in patterns:
                if pattern.lower() in response.lower():
                    return option

        # Look for option followed by period or at beginning
        for option in ["A", "B", "C", "D"]:
            if response.startswith(option) or f"{option}." in response:
                return option

        # Extract first letter if it's a valid option
        if response and response[0].upper() in ["A", "B", "C", "D"]:
            return response[0].upper()

        # Count occurrences of each option
        counts = {option: response.lower().count(f" {option.lower()} ")
                  for option in ["A", "B", "C", "D"]}
        if max(counts.values()) > 0:
            return max(counts.items(), key=lambda x: x[1])[0]

        # Default to the first valid option mentioned
        for char in response:
            if char.upper() in ["A", "B", "C", "D"]:
                return char.upper()

        return "C: fallback"

    else:  # Tự luận
        # For free-text questions, clean up the response
        lines = response.split('\n')
        filtered_lines = []
        for line in lines:
            if not line.strip():
                continue
            if any(phrase in line.lower() for phrase in ["dựa trên", "dựa vào", "theo ngữ cảnh", "phân tích"]):
                continue
            filtered_lines.append(line)

        cleaned_response = ' '.join(filtered_lines)
        return cleaned_response.strip()


def generate_answer(model, tokenizer, question: Dict[str, Any], max_length: int = 512) -> str:
    """Generate an answer for a question using the model."""
    question_text = question["text"]
    question_type = question["question_type"]
    context = question.get("context", "")

    # Select the appropriate prompt based on the question type
    if question_type == "Đúng/Sai":
        prompt = generate_true_false_prompt(question_text, context)
    elif question_type == "Trắc nghiệm":
        choices = question.get("choices", {})
        prompt = generate_multiple_choice_prompt(
            question_text, context, choices)
    else:  # Tự luận
        prompt = generate_free_text_prompt(question_text, context)

    # Generate response using the model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.7,
            num_beams=3,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer part (after the prompt)
    answer = full_response[len(prompt):]

    # Process the answer based on question type
    processed_answer = extract_answer_from_response(answer, question_type)

    return processed_answer


def process_questions(data: List[Dict[str, Any]], model, tokenizer) -> List[Dict[str, Any]]:
    """Process all questions and generate answers."""
    results = []

    # Create a progress bar
    progress_bar = tqdm(
        total=len(data), desc="Processing questions", unit="question")

    for i, question in enumerate(data):
        question_id = question['question_id']
        progress_bar.set_description(f"Processing: {question_id}")

        # Skip questions without context
        if "context" not in question or not question["context"]:
            progress_bar.write(
                f"Skipping question {question_id} - no context provided")
            progress_bar.update(1)
            continue

        answer = generate_answer(model, tokenizer, question)

        result = {
            "question_id": question_id,
            "answer": answer
        }
        results.append(result)

        # Use tqdm.write to avoid interference with the progress bar
        progress_bar.write(f"Question: {question['text']}")
        progress_bar.write(f"Answer: {answer}")
        progress_bar.write("-" * 50)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()
    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers for legal questions using an LLM")
    parser.add_argument("--input", type=str, default="dataset/task2/qid_qtext_qtype_rel_context_combined.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output", type=str, default="output/answers.json",
                        help="Path to the output JSON file")
    parser.add_argument("--model", type=str, default="vilm/vietcuna-7b-v3",
                        help="Name or path to the LLM model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (cuda or cpu)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load data
    data = load_data(args.input)

    # Setup model and tokenizer
    model, tokenizer = setup_model(args.model, args.device)

    # Process questions
    results = process_questions(data, model, tokenizer)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
