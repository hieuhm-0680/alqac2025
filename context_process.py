import json
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from src.models.schemas import QuestionType
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

splitter = RecursiveCharacterTextSplitter(separators=["\n\n", r"\n\d+\. Sửa đổi", "\n\d+", "\n", " ", ""], is_separator_regex=True, chunk_size=1200, chunk_overlap=200)


def build_worker_prompt(question, current_chunk, previous_summary=None):
    system_prompt = (
        "You are Qwen, a helpful assistant specialized in multi-step document comprehension. "
        "Your task is to extract relevant information from the current text to help answer the user's question in Vietnamese. "
        "Work carefully and avoid hallucinating facts."
    )

    if previous_summary:
        user_prompt = f"""
Câu hỏi:
{question}

Tài liệu hiện tại:
{current_chunk}

Tóm tắt trước đó:
{previous_summary}

Instructions:
1. Carefully read the current Vietnamese text and the previous summary.
2. Identify any new information from the current text that is relevant to the question.
3. Combine the new information with the previous summary if needed, removing redundant parts.
4. If the current text contains no new relevant information, just pass the previous summary unchanged.
5. Output must be in Vietnamese, concise, and factual.
6. Do not make up any information.
"""
    else:
        user_prompt = f"""
Câu hỏi:
{question}

Tài liệu hiện tại:
{current_chunk}

Instructions:
1. Extract any information from the current text that is relevant to the question.
2. Summarize only the relevant information. If there's nothing relevant, output: "No relevant information found."
3. Output must be in Vietnamese, concise, and factual.
4. Do not make up any information.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def call_worker_agent(question, chunk, previous_summary=None):
    messages = build_worker_prompt(question, chunk, previous_summary)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.strip()
    if 'no relevant information found' in response.lower():
        return ""
    return response

def prepare_question(question):
    if question['question_type'] == QuestionType.TRUE_FALSE:
        return question['text']
    elif question['question_type'] == QuestionType.MULTIPLE_CHOICE:
        return question['text'] + "\n\n" + " ".join([f"{key}: {value}" for key, value in question.get('choices', {}).items()])
    elif question['question_type'] == QuestionType.FREE_TEXT:
        return question['text']
    else:
        raise ValueError(f"Unsupported question type: {question['question_type']}")

def run_process(questions, article_mapping):
    results = []
    for question in questions:
        state = {
            'question_id': question['question_id'],
            'question_type': question['question_type'],
            'text': question['text'],
            'choices': question.get('choices', {}),
            'relevant_articles': question.get('relevant_articles', []),
            'worker_responses': []
        }
        question_text = prepare_question(question)
        response = None

        for i, item in enumerate(question['relevant_articles'], 1):
            law_id = item['law_id']
            articles_id = item['article_id']
            text = article_mapping[law_id][articles_id]
            chunks = splitter.split_text(text)
            for j, chunk in enumerate(chunks, 1):
                response = call_worker_agent(question_text, chunk, response)
                state['worker_responses'].append({
                    'worker': f"worker_{i}_{j}",
                    'response': response,
                    'text': chunk
                    })
        results.append(state)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", "-q", type=str, required=True,
                        help="Optional JSON file containing questions (question_id and text)")
    parser.add_argument("--law_path", "-l", type=str, required=True,
                        help="Path to the law file")
    parser.add_argument("--output_path", "-o", type=str, required=True,
                        help="Path to save the output file")

    args = parser.parse_args()

    with open(args.questions, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(args.law_path, 'r', encoding='utf-8') as f:
        laws = json.load(f)

    article_mapping = {
        item['id']: {article["id"]: article['text'] for article in item["articles"]}
        for item in laws
    }
    
    results = run_process(questions, article_mapping)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()