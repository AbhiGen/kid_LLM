import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForCausalLM, AutoTokenizer

class Explainer:
    def __init__(self, model_path="../models/nutrition_llm"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.explainer = LimeTextExplainer(class_names=["nutrition_recommendation"])

    def predict_proba(self, text):
        """
        Generates a probability distribution over the vocabulary for the given text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get the logits for the last token
        logits = outputs.logits[:, -1, :]
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities.numpy()

    def explain(self, text):
        """
        Generates an explanation for the model's prediction on the given text.
        """
        explanation = self.explainer.explain_instance(
            text, self.predict_proba, num_features=10
        )
        return explanation.as_list()

    def get_confidence_score(self, text):
        """
        Calculates a confidence score for the model's prediction.
        """
        probabilities = self.predict_proba([text])[0]
        # Use the max probability as the confidence score
        confidence = np.max(probabilities)
        return confidence

if __name__ == "__main__":
    # This is a placeholder for a fine-tuned model.
    # In a real scenario, you would load your fine-tuned model here.
    # For now, we'll use the base model for demonstration purposes.
    explainer = Explainer(model_path="microsoft/DialoGPT-small")
    text = "What are some healthy snacks for a 5-year-old?"
    explanation = explainer.explain(text)
    confidence = explainer.get_confidence_score(text)
    print("Explanation:", explanation)
    print("Confidence:", confidence)
