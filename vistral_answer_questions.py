import json
import argparse
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def setup_model(model_name_or_path: str = "Viet-Mistral/Vistral-7B-Chat", device: str = "cuda") -> Tuple[Any, Any]:
    """Load the Vistral model and tokenizer."""
    print(f"Loading model {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Use appropriate torch_dtype based on device
    if device == "cuda" and torch.cuda.is_available():
        try:
            # Try to use bfloat16 first (for newer GPUs)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                use_cache=True,
            )
            print(f"Model loaded on {device} with torch.bfloat16")
        except Exception as e:
            print(
                f"Could not load with bfloat16, falling back to float16: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                use_cache=True
            )
            print(f"Model loaded on {device} with torch.float16")
    else:
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        print(f"Model loaded on CPU")

    return model, tokenizer


def create_system_prompt() -> str:
    """Create the system prompt for the Vistral model."""
    system_prompt = "Bạn là một trợ lý AI chuyên gia trong lĩnh vực pháp luật. "
    system_prompt += "Nhiệm vụ của bạn là trả lời các câu hỏi pháp luật dựa trên ngữ cảnh được cung cấp. "
    system_prompt += "Hãy phân tích kỹ thông tin trong ngữ cảnh pháp luật và đưa ra câu trả lời chính xác. "
    system_prompt += "Khi trả lời, hãy luôn suy nghĩ một cách cẩn thận và đưa ra lập luận của bạn trong thẻ <thinking></thinking>. "
    system_prompt += "Sau khi suy nghĩ, đặt câu trả lời chính thức của bạn trong thẻ <answer></answer>. "
    system_prompt += "Với câu hỏi Đúng/Sai, câu trả lời chính thức phải là 'Đúng' hoặc 'Sai'. "
    system_prompt += "Với câu hỏi trắc nghiệm, câu trả lời chính thức phải là 'A', 'B', 'C' hoặc 'D'. "
    system_prompt += "Với câu hỏi tự luận, câu trả lời chính thức phải ngắn gọn, đầy đủ và chính xác dựa trên ngữ cảnh pháp luật."
    return system_prompt


def generate_true_false_conversation(question: str, context: str) -> List[Dict[str, str]]:
    system_prompt = create_system_prompt()

    user_message = f"""Dưới đây là một câu hỏi Đúng/Sai trong lĩnh vực pháp luật. Hãy trả lời "Đúng" hoặc "Sai" dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh pháp luật:
{context}

Câu hỏi: {question}

Đầu tiên, hãy suy nghĩ và phân tích kỹ ngữ cảnh pháp luật trong thẻ <thinking></thinking>.
Sau đó, đưa ra câu trả lời cuối cùng là "Đúng" hoặc "Sai" trong thẻ <answer></answer>."""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    return conversation


def generate_multiple_choice_conversation(question: str, context: str, choices: Dict[str, str]) -> List[Dict[str, str]]:
    """Generate conversation for multiple choice questions."""
    system_prompt = create_system_prompt()

    choices_text = "\n".join([f"{k}: {v}" for k, v in choices.items()])

    user_message = f"""Dưới đây là một câu hỏi trắc nghiệm trong lĩnh vực pháp luật. Hãy chọn một đáp án đúng (A, B, C hoặc D) dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh pháp luật:
{context}

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Đầu tiên, hãy suy nghĩ và phân tích kỹ mỗi lựa chọn dựa trên ngữ cảnh pháp luật trong thẻ <thinking></thinking>.
Sau đó, đưa ra câu trả lời cuối cùng là một trong các lựa chọn: "A", "B", "C" hoặc "D" trong thẻ <answer></answer>."""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    return conversation


def generate_free_text_conversation(question: str, context: str) -> List[Dict[str, str]]:
    """Generate conversation for free-text questions."""
    system_prompt = create_system_prompt()

    user_message = f"""Dưới đây là một câu hỏi tự luận trong lĩnh vực pháp luật. Hãy trả lời câu hỏi dựa vào ngữ cảnh pháp luật được cung cấp.

Ngữ cảnh pháp luật:
{context}

Câu hỏi: {question}

Đầu tiên, hãy suy nghĩ và phân tích kỹ ngữ cảnh pháp luật trong thẻ <thinking></thinking>.
Sau đó, đưa ra câu trả lời cuối cùng chính xác, ngắn gọn và đầy đủ trong thẻ <answer></answer>."""

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    return conversation


def extract_answer_from_response(response: str, question_type: str) -> tuple:
    """Extract the actual answer and thinking from the model's response."""
    response = response.strip()

    # Extract thinking part if available
    thinking = ""
    thinking_match = re.search(
        r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract answer part if available
    answer_text = ""
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Use the extracted answer directly
        if answer_text:
            if question_type == "Đúng/Sai":
                # Ensure the answer is properly capitalized
                if "đúng" in answer_text.lower() and "sai" not in answer_text.lower():
                    return "Đúng", thinking
                elif "sai" in answer_text.lower() and "đúng" not in answer_text.lower():
                    return "Sai", thinking
                # Extract first word if it's "Đúng" or "Sai"
                words = answer_text.split()
                if words:
                    first_word = words[0].lower()
                    if first_word == "đúng":
                        return "Đúng", thinking
                    elif first_word == "sai":
                        return "Sai", thinking

            elif question_type == "Trắc nghiệm":
                # Extract option letter from the answer
                for option in ["A", "B", "C", "D"]:
                    if option in answer_text or option.lower() in answer_text:
                        return option, thinking

                # If no option found directly, check for first letter
                if answer_text and answer_text[0].upper() in ["A", "B", "C", "D"]:
                    return answer_text[0].upper(), thinking

            elif question_type == "Tự luận":
                # Return the answer as is for free-text questions
                return answer_text.strip(), thinking

    # If no <answer> tag or appropriate answer found, fall back to analyzing the whole response

    if question_type == "Đúng/Sai":
        # Look for "Đúng" or "Sai" in the response
        if "đúng" in response.lower() and "sai" not in response.lower():
            return "Đúng", thinking
        elif "sai" in response.lower() and "đúng" not in response.lower():
            return "Sai", thinking

        # Look for phrases indicating true/false
        if "không đúng" in response.lower() or "không chính xác" in response.lower():
            return "Sai", thinking
        if "là đúng" in response.lower() or "chính xác" in response.lower():
            return "Đúng", thinking

        # Default fallback
        return "Đúng" if "đúng" in response.lower() else "Sai", thinking

    elif question_type == "Trắc nghiệm":
        # Look for clear option statements
        for option in ["A", "B", "C", "D"]:
            patterns = [
                f"đáp án {option}", f"đáp án là {option}",
                f"chọn {option}", f"lựa chọn {option}",
                f"câu trả lời là {option}", f"câu trả lời: {option}",
                f"lựa chọn đúng là {option}", f"câu trả lời đúng là {option}"
            ]
            for pattern in patterns:
                if pattern.lower() in response.lower():
                    return option, thinking

        # Look for option at beginning or followed by a period
        for option in ["A", "B", "C", "D"]:
            if response.startswith(option) or f"{option}." in response:
                return option, thinking

        # Default to the first valid option mentioned
        for char in response:
            if char.upper() in ["A", "B", "C", "D"]:
                return char.upper(), thinking

        return "A", thinking  # Ultimate fallback

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
        return cleaned_response.strip(), thinking


def generate_answer(model, tokenizer, question: Dict[str, Any]) -> tuple:
    """Generate an answer for a question using the Vistral model."""
    question_text = question["text"]
    question_type = question["question_type"]
    context = question.get("context", "")

    # Select the appropriate conversation based on the question type
    if question_type == "Đúng/Sai":
        conversation = generate_true_false_conversation(question_text, context)
    elif question_type == "Trắc nghiệm":
        choices = question.get("choices", {})
        conversation = generate_multiple_choice_conversation(
            question_text, context, choices)
    else:  # Tự luận
        conversation = generate_free_text_conversation(question_text, context)

    # Apply chat template and generate response using the model
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt")

    # Move input_ids to the appropriate device
    input_ids = input_ids.to(model.device)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=500,  # Increased from 200 to allow for more thinking
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=0.1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the model output, skipping the input part
    response = tokenizer.batch_decode(
        output_ids[:, input_ids.size(1):],
        skip_special_tokens=True
    )[0].strip()

    # Process the answer based on question type
    processed_answer, thinking = extract_answer_from_response(
        response, question_type)

    return processed_answer, response, thinking


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

        # Generate answer and get full response and thinking
        answer, full_response, thinking = generate_answer(
            model, tokenizer, question)

        result = {
            "question_id": question_id,
            "full_response": full_response,
            "thinking": thinking,
            "answer": answer
        }
        results.append(result)

        # Use tqdm.write to avoid interference with the progress bar
        progress_bar.write(f"Question: {question['text']}")
        progress_bar.write(f"Thinking: {thinking}")
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
        description="Generate answers for legal questions using Vistral-7B-Chat")
    parser.add_argument("--input", type=str, default="dataset/task2/qid_qtext_qtype_rel_context_combined.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output", type=str, default="output/vistral_answers.json",
                        help="Path to the output JSON file")
    parser.add_argument("--model", type=str, default="Viet-Mistral/Vistral-7B-Chat",
                        help="Name or path to the LLM model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")

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
