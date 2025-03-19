import os
from transformers import pipeline
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QFileDialog, QVBoxLayout, QWidget

def summarize_text(file_path, model, tokenizer, max_tokens=512):
    try:
        with open(file_path, 'r') as file:
            text = file.read()

        # Tokenize and split into chunks
        tokens = tokenizer.encode(text, truncation=False)
        chunk_size = max_tokens
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            input_text = tokenizer.decode(chunk, skip_special_tokens=True)
            summary = model(input_text, max_length=150, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        print(f"Token length for {file_path}: {len(tokens)}")

        # Combine the summaries
        return " ".join(summaries)

    except Exception as e:
        return f"An error occurred: {str(e)}"


def compare_models(file_paths, models):
    results = []
    for file_path in file_paths:
        file_result = {"file": file_path}
        for model_name, model_info in models.items():
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            summary = summarize_text(file_path, model, tokenizer)
            file_result[model_name] = summary
        results.append(file_result)
    return results

# Main Application Class
class SummarizationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text Summarization Tool")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.open_button = QPushButton("Open Files", self)
        self.open_button.clicked.connect(self.open_files)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)

        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.result_text)

        self.reference_summary = '''Artificial intelligence (AI) refers to the simulation of human intelligence in machines that can learn, reason, and make decisions. AI applications span web search engines, recommendation systems, autonomous vehicles, and healthcare innovations. Historical milestones include Deep Blue defeating a chess champion, AlphaGo excelling in the game of Go, and GPT models achieving human-level performance on various tasks.
                                    Key techniques in AI include machine learning (supervised, unsupervised, and reinforcement learning), neural networks, and deep learning, which have revolutionized tasks like image recognition and natural language processing. AI poses challenges related to ethical concerns, such as bias, privacy, and transparency, as well as potential risks from autonomous weapons and misinformation.
                                    AI's societal impacts range from transforming industries to raising questions about employment and regulation. Governments and organizations are investing in AI research and policy frameworks to address these challenges, including initiatives like the Global Partnership on AI. Philosophical debates about AI's nature, intelligence, and potential risks continue to shape the field, with efforts to develop explainable and ethical AI systems.'''

    def evaluate_summaries(self, reference, bert_summary, gpt_summary):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bert_scores = scorer.score(reference, bert_summary)
        gpt_scores = scorer.score(reference, gpt_summary)
        return bert_scores, gpt_scores

    def open_files(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Open Text Files", "", "Text Files (*.txt);;All Files (*)")

        if file_paths:
            results = compare_models(file_paths, summarization_models)
            self.result_text.clear()
            for result in results:
                self.result_text.append(f"File: {result['file']}\n")
                for model_name, summary in result.items():
                    if model_name != 'file':
                        self.result_text.append(f"Model ({model_name}): {summary}\n\n")

                self.result_text.append(f"File: {result['file']}\n")
                bert_summary = result.get("BERT", "No summary generated.")
                gpt_summary = result.get("GPT", "No summary generated.")
                
                self.result_text.append(f"Model (BERT): {bert_summary}\n")
                self.result_text.append(f"Model (GPT): {gpt_summary}\n\n")

                # Calculate ROUGE scores
                bert_scores, gpt_scores = self.evaluate_summaries(self.reference_summary, bert_summary, gpt_summary)

                # Display ROUGE scores
                self.result_text.append("BERT ROUGE Scores:\n")
                self.result_text.append(f"ROUGE-1: {bert_scores['rouge1']}\n")
                self.result_text.append(f"ROUGE-2: {bert_scores['rouge2']}\n")
                self.result_text.append(f"ROUGE-L: {bert_scores['rougeL']}\n\n")

                self.result_text.append("GPT ROUGE Scores:\n")
                self.result_text.append(f"ROUGE-1: {gpt_scores['rouge1']}\n")
                self.result_text.append(f"ROUGE-2: {gpt_scores['rouge2']}\n")
                self.result_text.append(f"ROUGE-L: {gpt_scores['rougeL']}\n\n")

summarization_models = {
    "BERT": {
        "model": pipeline("summarization", model="facebook/bart-large-cnn"),
        "tokenizer": AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    },
    "GPT": {
        "model": pipeline("summarization", model="t5-small"),
        "tokenizer": AutoTokenizer.from_pretrained("t5-small")
    }
}

# Run the application
if __name__ == "__main__":
    app = QApplication([])
    window = SummarizationApp()
    window.show()
    app.exec_()
