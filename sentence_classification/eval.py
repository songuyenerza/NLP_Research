import json
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from transformers import AutoTokenizer
from intent_customer.function import rdrsegmenter_data, pre_process_text, sanitize_filename
from intent_customer.model import PhoBertMultiLabelClassifierCustomer

OUTPUT = "./eval1609"
os.makedirs(OUTPUT, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=os.path.join(OUTPUT, "inference_log.log"),  # Log file path
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

class InferenceIntentAnalysis:
    """Class for inference and evaluating the model text by text."""

    def __init__(self, path_weights):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        self.path_weights = path_weights
        self.label_mapping = None
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.device = device
        self.__load_label_mapping()

    def __load_label_mapping(self):
        try:
            label_mapping_path = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/output1309/weights/label_mapping.json"
            with open(label_mapping_path) as f:
                self.label_mapping = json.load(f)
            logging.info(f"Label mapping loaded successfully.")
            logging.info(f"Label mapping::: {self.label_mapping}.")
        except Exception as e:
            logging.error(f"Error loading label mapping: {str(e)}")
            raise

    def load_model(self):
        """Load the trained model with saved weights."""
        self.model = PhoBertMultiLabelClassifierCustomer(len(self.label_mapping)).to(self.device)
        self.model.load_state_dict(torch.load(self.path_weights))
        self.model = self.model.eval()
        logging.info("Model loaded successfully.")

    def infer_single_text(self, text):
        """
        Run inference on a single text and return the predicted label.
        
        Args:
            text (str): Input text.
        
        Returns:
            list: Predicted labels.
        """
        t_start = time.time()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            max_score, max_index = torch.max(outputs, dim=1)
            print(f"[TIME infer ] {time.time() - t_start}")
            return max_index.item()

    def run_inference(self, csv_file):
        """
        Run inference on a CSV file, process text one by one, and log incorrect predictions.
        """
        try:
            # Load the test data
            df = pd.read_csv(csv_file)

            # Process the texts
            test_texts = df['text'].values
            true_labels = df['intent'].values  # Ground truth labels

            incorrect_cases = []
            label_list = [label_str for label_str in self.label_mapping.keys()]
            correct_predictions = {label: 0 for label in self.label_mapping.values()}
            incorrect_predictions = {label: {other_label: 0 for other_label in self.label_mapping.values()} for label in self.label_mapping.values()}
            total_predictions = {label: 0 for label in self.label_mapping.values()}
            count_all = 0
            count_false = 0
            for i, text in enumerate(test_texts):
                # Preprocess text
                try:
                    processed_text = pre_process_text([text])
                    processed_text = rdrsegmenter_data(processed_text)[0]

                    if true_labels[i] in self.label_mapping:
                        # Get ground truth and predicted labels
                        true_label = self.label_mapping[true_labels[i]]
                        predicted_label = self.infer_single_text(processed_text)

                        # Update prediction counts
                        total_predictions[true_label] += 1
                        count_all += 1

                        if predicted_label == true_label:
                            correct_predictions[true_label] += 1
                        else:
                            count_false += 1
                            incorrect_predictions[true_label][predicted_label] += 1
                            incorrect_cases.append({
                                "text": text,
                                "predicted": label_list[predicted_label],
                                "actual": label_list[true_label]
                            })
                            print(f"Incorrect case - Text: {text}, Predicted: {label_list[predicted_label]}, Actual: {label_list[true_label]}")
                            print(f">>>>>>   ACC :::: {(count_all - count_false) / count_all} ||| {count_all}")
                except:
                    pass

            # Save incorrect cases to CSV
            incorrect_df = pd.DataFrame(incorrect_cases)
            incorrect_df.to_csv(os.path.join(OUTPUT, 'incorrect.csv'), index=False)
            logging.info(f"Incorrect cases saved to {os.path.join(OUTPUT, 'incorrect.csv')}")

            # Generate pie charts for each class
            self.generate_pie_charts(correct_predictions, incorrect_predictions, total_predictions)

        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            raise

    def generate_pie_charts(self, correct_predictions, incorrect_predictions, total_predictions):
        """
        Generate pie charts for each class showing correct and incorrect predictions.
        """
        for class_name, class_label in self.label_mapping.items():
            fig, ax = plt.subplots()

            # Labels and sizes for the pie chart
            labels = ['Correct'] + [f'Incorrect to {other_class}' for other_class in incorrect_predictions[class_label] if incorrect_predictions[class_label][other_class] > 0]
            sizes = [correct_predictions[class_label]] + [incorrect_predictions[class_label][other_class] for other_class in incorrect_predictions[class_label] if incorrect_predictions[class_label][other_class] > 0]
            
            # Generate pie chart
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            sanitized_class_name = sanitize_filename(class_name)
            plt.title(f"{sanitized_class_name} : {class_label} : {total_predictions[class_label]}")
            plt.savefig(os.path.join(OUTPUT, f'{class_label}_{sanitized_class_name}_prediction_breakdown.png'))

        plt.close()

# Example usage
path_weights = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/output1309/weights/best_model.pth"
csv_file = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/train_data.csv"

# Initialize the inference class
inference_model = InferenceIntentAnalysis(path_weights)
# Load the trained model
inference_model.load_model()
# Run inference on the CSV file
inference_model.run_inference(csv_file)
