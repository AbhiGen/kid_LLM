import csv
import json
import random
from pathlib import Path

# Define the categories for the dataset
CATEGORIES = {
    "age_groups": ["toddlers (1-3 years)", "preschoolers (4-5 years)", "kids (6-12 years)", "teens (13-18 years)"],
    "meals": ["breakfast", "lunch", "dinner", "snacks"],
    "allergies": ["peanuts", "dairy", "gluten", "shellfish", "eggs"],
    "dietary_restrictions": ["vegetarian", "vegan", "pescatarian"],
    "health_tips": [
        "importance of hydration",
        "benefits of a balanced diet",
        "how to read nutrition labels",
        "healthy snack ideas",
        "the food pyramid",
    ],
    "food_safety": [
        "hand washing before meals",
        "safe food storage",
        "cooking temperatures",
        "cross-contamination",
    ],
    "picky_eating": [
        "how to introduce new foods",
        "making food fun",
        "patience with picky eaters",
        "involving kids in cooking",
    ],
}

# Define templates for generating questions and answers
TEMPLATES = [
    {
        "question": "What are some healthy {meals} for {age_groups}?",
        "answer": "For {age_groups}, a healthy {meals} could include a source of protein like lean meat or beans, a whole grain like brown rice or quinoa, and a variety of colorful vegetables. For example, a grilled chicken salad or a black bean burger on a whole wheat bun.",
    },
    {
        "question": "My child is allergic to {allergies}. What are some safe {meals} for them?",
        "answer": "For a child with a {allergies} allergy, it's crucial to avoid any foods containing that allergen. For {meals}, a safe option could be a sunflower seed butter and jelly sandwich on gluten-free bread, or a rice pasta with a vegetable-based sauce.",
    },
    {
        "question": "What are some {dietary_restrictions} {meals} for {age_groups}?",
        "answer": "A {dietary_restrictions} {meals} for {age_groups} could be a lentil soup with a side of whole-grain bread, or a tofu stir-fry with a variety of vegetables.",
    },
    {
        "question": "Can you give me some information about the {health_tips} for {age_groups}?",
        "answer": "The {health_tips} is especially important for {age_groups}. For example, it's recommended that they drink plenty of water throughout the day to stay hydrated and support their growing bodies.",
    },
    {
        "question": "What are the key points of {food_safety} when preparing food for {age_groups}?",
        "answer": "When it comes to {food_safety} for {age_groups}, it's important to always wash your hands before preparing food, cook meat to the proper temperature, and store leftovers in the refrigerator promptly.",
    },
    {
        "question": "How can I deal with {picky_eating} in {age_groups}?",
        "answer": "Dealing with {picky_eating} in {age_groups} can be challenging. One tip is to {picky_eating}, for example, by creating a colorful plate or cutting sandwiches into fun shapes. It's also important to be patient and continue offering new foods without pressure.",
    },
]

def generate_dataset(num_samples=2000):
    """Generates a dataset of nutrition-related questions and answers."""
    dataset = []
    for _ in range(num_samples):
        template = random.choice(TEMPLATES)
        question = template["question"]
        answer = template["answer"]

        for category, values in CATEGORIES.items():
            if f"{{{category}}}" in question:
                value = random.choice(values)
                question = question.replace(f"{{{category}}}", value)
                answer = answer.replace(f"{{{category}}}", value)

        dataset.append({"question": question, "answer": answer})
    return dataset

def save_dataset(dataset, output_dir="../data"):
    """Saves the dataset to JSON, CSV, and JSONL files."""
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(os.path.join(script_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_dir / "nutrition_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)

    # Save to CSV
    with open(output_dir / "nutrition_dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(dataset)

    # Save to JSONL
    with open(output_dir / "nutrition_dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset, output_dir=".")
    print(f"Successfully generated and saved a dataset with {len(dataset)} samples.")
