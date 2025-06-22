import yaml
import json
from typing import List, Dict


class ALQAC2025System:
    """Main system for ALQAC2025 tasks."""
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.retriever = Retriever(self.config, law_documents)
        self.answerer = Answerer(self.config, law_documents)
        self.evaluator = Evaluator()

        # Data storage
        self.law_documents: Dict[str, Document] = {}
        self.training_questions: List[TrainingQuestion] = []
        self.is_initialized = False

    def process_task1(self, split: str = 'test') -> List[Dict]:
        """Process Task 1: Legal Document Retrieval."""
        data = self.data_loader.load_data(split)
        results = []
        for item in data:
            articles = self.retriever.retrieve(item['text'])
            results.append({
                "question_id": item['question_id'],
                "relevant_articles": articles
            })
        return results

    def process_task2(self, split: str = 'test') -> List[Dict]:
        """Process Task 2: Legal Question Answering."""
        data = self.data_loader.load_data(split)
        results = []
        for item in data:
            articles = self.retriever.retrieve(item['text'])
            answer = self.answerer.answer(item, articles)
            results.append({
                "question_id": item['question_id'],
                "answer": answer
            })
        return results

    def evaluate(self, task: str, predictions: List[Dict], split: str = 'dev') -> Dict[str, float]:
        """Evaluate predictions for a given task."""
        ground_truth = self.data_loader.load_data(split)
        if task == 'task1':
            return self.evaluator.evaluate_retrieval(predictions, ground_truth)
        else:  # task2
            return self.evaluator.evaluate_qa(predictions, ground_truth)

    def save_submission(self, predictions: List[Dict], output_path: str):
        """Save predictions in the required JSON format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)