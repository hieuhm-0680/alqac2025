import json
from tqdm import tqdm
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from src.models.schemas import QuestionType
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None

prompt_language = "en"

splitter = RecursiveCharacterTextSplitter(separators=["\n\n", r"\n\d+\. Sửa đổi", r"\n\d+", "\n", " ", ""], is_separator_regex=True, chunk_size=1200, chunk_overlap=200)

#####################################################
SYSTEM_PROMPT_QWEN25 = (
        "You are a helpful assistant specialized in multi-step document comprehension. "
        "Your task is to extract relevant information from the current text to help answer the user's question in Vietnamese. "
        "Work carefully and avoid hallucinating facts."
    )
USER_PROMPT_QWEN25INSTRUCT = """
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
5. Output must be in Vietnamese, concise, and factual, DO NOT GIVE THE ANSWER.
6. Do not make up any information.
"""

USER_PROMPT_QWEN25INSTRUCT_NO_SUMMARY = """
Câu hỏi:
{question}

Tài liệu hiện tại:
{current_chunk}

Instructions:
1. Extract any information from the current text that is relevant to the question.
2. Summarize only the relevant information.
3. Output must be in Vietnamese, concise, and factual, DO NOT GIVE THE ANSWER.
4. Do not make up any information.
"""
#####################################################

SYSTEM_PROMPT_QWEN25_VI = (
    "Bạn là một trợ lý hữu ích chuyên về phân tích văn bản. "
    "Nhiệm vụ của bạn là trích xuất thông tin liên quan từ văn bản hiện tại để giúp trả lời câu hỏi của người dùng. "
    "Hãy làm việc cẩn thận và tránh tạo ra thông tin không chính xác."
)

USER_PROMPT_QWEN25_VI_NO_SUMMARY = """
Câu hỏi:
{question}

Tài liệu hiện tại:
{current_chunk}

Hướng dẫn:
1. Trích xuất bất kỳ thông tin nào từ văn bản hiện tại liên quan đến câu hỏi.
2. Tóm tắt chỉ thông tin liên quan.
3. Đầu ra phải bằng tiếng Việt, ngắn gọn và chính xác, TUYỆT ĐỐI KHÔNG ĐƯA RA NHẬN ĐỊNH VỀ ĐÁP ÁN.
"""

USER_PROMPT_QWEN25_VI = """
Câu hỏi:
{question}

Tài liệu hiện tại:
{current_chunk}

Tóm tắt trước đó:
{previous_summary}

Hướng dẫn:
1. Đọc kỹ văn bản tiếng Việt hiện tại và tóm tắt trước đó.
2. Xác định bất kỳ thông tin mới nào từ văn bản hiện tại liên quan đến câu hỏi.
3. Kết hợp thông tin mới với tóm tắt trước đó nếu cần, loại bỏ các phần trùng lặp.
4. Nếu văn bản hiện tại không chứa thông tin mới nào liên quan, chỉ cần giữ nguyên tóm tắt trước đó.
5. Đầu ra phải bằng tiếng Việt, ngắn gọn và chính xác, TUYỆT ĐỐI KHÔNG ĐƯA RA NHẬN ĐỊNH VỀ ĐÁP ÁN.
"""


def build_worker_prompt(question, current_chunk, previous_summary=None):
    system_prompt = SYSTEM_PROMPT_QWEN25 if prompt_language == "en" else SYSTEM_PROMPT_QWEN25_VI

    if previous_summary:
        user_prompt = USER_PROMPT_QWEN25INSTRUCT.format(
            question=question,
            current_chunk=current_chunk,
            previous_summary=previous_summary
        ) if prompt_language == "en" else USER_PROMPT_QWEN25_VI.format(
            question=question,
            current_chunk=current_chunk,
            previous_summary=previous_summary
        )
    else:
        user_prompt = USER_PROMPT_QWEN25INSTRUCT_NO_SUMMARY.format(
            question=question,
            current_chunk=current_chunk
        ) if prompt_language == "en" else USER_PROMPT_QWEN25_VI_NO_SUMMARY.format(
            question=question,
            current_chunk=current_chunk
        )

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
    return response

def prepare_question(question):
    if question['question_type'] == QuestionType.TRUE_FALSE:
        template = f"Phát biểu sau có đúng hay sai: {question['text']}?"
        return template
    elif question['question_type'] == QuestionType.MULTIPLE_CHOICE:
        template = f"Đâu là lựa chọn đúng cho câu hỏi sau: {question['text']}\n\n"
        template += "Các lựa chọn:\n"
        template += "\n".join([f"{key}: {value}" for key, value in question['choices'].items()])
        return template
    elif question['question_type'] == QuestionType.FREE_TEXT:
        return question['text']
    else:
        raise ValueError(f"Unsupported question type: {question['question_type']}")
    
def prepare_chunk(chunk, law_id):
    return f"...(đây là trích dẫn từ luật {law_id}) \n\n{chunk}".strip()

def run_process(questions, article_mapping):
    results = []
    for question in tqdm(questions, desc="Processing questions"):
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
                chunk = prepare_chunk(chunk, law_id)
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

    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Name of the model to use")
    parser.add_argument("--language", "-lang", type=str, default="en",
                        help="Language of the model)")
    args = parser.parse_args()

    with open(args.questions, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(args.law_path, 'r', encoding='utf-8') as f:
        laws = json.load(f)

    article_mapping = {
        item['id']: {article["id"]: article['text'] for article in item["articles"]}
        for item in laws
    }
    
    global model, tokenizer, prompt_language
    model_name = args.model_name
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt_language = args.language
    
    results = run_process(questions, article_mapping)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()