import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def fine_tune_model(
    base_model="microsoft/DialoGPT-small",
    dataset_path="data",
    output_dir="models/nutrition_llm",
):
    """Fine-tunes a pretrained model on the nutrition dataset."""
    # Load the dataset
    dataset = load_dataset("json", data_files=os.path.join(dataset_path, "nutrition_dataset.jsonl"), split="train")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess the dataset
    def preprocess_function(examples):
        # Format the Q&A pairs
        inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        # Tokenize the inputs
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        # Set the labels to be the same as the input IDs
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Create the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fine_tune_model(dataset_path="../data")
