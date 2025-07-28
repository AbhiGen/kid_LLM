import gradio as gr
import pandas as pd
import qrcode
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from kids_nutrition_llm.src.explainability import Explainer

# Load the fine-tuned model and tokenizer
model_path = "../models/nutrition_llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the explainer
explainer = Explainer(model_path=model_path)

# Load the nutrition dataset for the nutrition facts tab
nutrition_df = pd.read_csv("../data/nutrition_dataset.csv")

# --- UI Functions ---

def chat_interface(message, history, age, gender, allergies, dietary_restrictions):
    """The main chat interface."""
    # Add demographic information to the prompt
    prompt = f"Age: {age}, Gender: {gender}, Allergies: {allergies}, Dietary Restrictions: {dietary_restrictions}\n\n{message}"
    response = chatbot(prompt, max_length=150)[0]["generated_text"]
    return response

def explanation_interface(message):
    """The explanation interface."""
    explanation = explainer.explain(message)
    confidence = explainer.get_confidence_score(message)
    return explanation, f"{confidence:.2f}"

def meal_planner_interface(age, gender, allergies, dietary_restrictions, weight, height):
    """The meal planner interface."""
    # Growth-Aware Meal Planner
    prompt = f"Create a healthy one-day meal plan for a {age}-year-old {gender} with a weight of {weight} kg and height of {height} cm. The child has {allergies} allergies and {dietary_restrictions} dietary restrictions. The meal plan should be adjusted for their growth needs."
    meal_plan = chatbot(prompt, max_length=300)[0]["generated_text"]

    # Smart Allergy Filter
    if allergies:
        allergies_list = [a.strip() for a in allergies.split(",")]
        for allergy in allergies_list:
            if allergy.lower() in meal_plan.lower():
                meal_plan += f"\n\n**Warning:** This meal plan may contain {allergy}. Please double-check the ingredients."

    # Visual Traffic-Light System (placeholder)
    meal_plan += "\n\n**Sugar:** 🟢\n**Sodium:** 🟡\n**Fat:** 🟢"

    return meal_plan

def nutrition_bingo_interface():
    """The nutrition bingo interface."""
    # Nutrition Bingo Game
    bingo_board = [
        ["Eat a fruit", "Eat a vegetable", "Drink water", "Try a new food", "Eat a whole grain"],
        ["Eat a protein", "Eat a dairy product", "Help cook a meal", "Eat a healthy snack", "Eat a green food"],
        ["Eat a red food", "Eat an orange food", "Eat a yellow food", "Eat a purple food", "Eat a blue food"],
        ["Eat a food that grows on a tree", "Eat a food that grows in the ground", "Eat a food with a peel", "Eat a food with seeds", "Eat a food that is crunchy"],
        ["Eat a food that is soft", "Eat a food that is sweet", "Eat a food that is savory", "Eat a food that is sour", "Eat a food that is bitter"],
    ]
    return pd.DataFrame(bingo_board)

def qr_code_interface(text):
    """The QR code interface."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

# --- Gradio App ---

with gr.Blocks() as app:
    gr.Markdown("# Kids' Nutrition LLM")

    with gr.Tabs():
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                with gr.Column():
                    age_input = gr.Number(label="Age")
                    gender_input = gr.Radio(["Male", "Female", "Other"], label="Gender")
                    allergies_input = gr.Textbox(label="Allergies (comma-separated)")
                    dietary_restrictions_input = gr.Textbox(
                        label="Dietary Restrictions (e.g., vegetarian, vegan)"
                    )
                with gr.Column():
                    chat_history = gr.Chatbot()
                    message_input = gr.Textbox(label="Your message")
                    submit_button = gr.Button("Send")

            submit_button.click(
                chat_interface,
                inputs=[message_input, chat_history, age_input, gender_input, allergies_input, dietary_restrictions_input],
                outputs=chat_history,
            )

        with gr.TabItem("🔍 Explanation"):
            with gr.Row():
                with gr.Column():
                    explanation_input = gr.Textbox(label="Enter a recommendation to explain")
                    explain_button = gr.Button("Explain")
                with gr.Column():
                    explanation_output = gr.JSON(label="LIME Explanation")
                    confidence_output = gr.Textbox(label="Confidence Score")

            explain_button.click(
                explanation_interface,
                inputs=[explanation_input],
                outputs=[explanation_output, confidence_output],
            )

        with gr.TabItem("📊 Nutrition Facts"):
            gr.DataFrame(nutrition_df)

        with gr.TabItem("🍽️ Meal Planner"):
            with gr.Row():
                with gr.Column():
                    mp_age_input = gr.Number(label="Age")
                    mp_gender_input = gr.Radio(["Male", "Female", "Other"], label="Gender")
                    mp_allergies_input = gr.Textbox(label="Allergies (comma-separated)")
                    mp_dietary_restrictions_input = gr.Textbox(
                        label="Dietary Restrictions (e.g., vegetarian, vegan)"
                    )
                    mp_weight_input = gr.Number(label="Weight (kg)")
                    mp_height_input = gr.Number(label="Height (cm)")
                    plan_button = gr.Button("Generate Meal Plan")
                with gr.Column():
                    meal_plan_output = gr.Textbox(label="Your Meal Plan")
                    qr_code_output = gr.Image(label="Shareable QR Code", type="pil")

            plan_button.click(
                meal_planner_interface,
                inputs=[
                    mp_age_input,
                    mp_gender_input,
                    mp_allergies_input,
                    mp_dietary_restrictions_input,
                    mp_weight_input,
                    mp_height_input,
                ],
                outputs=meal_plan_output,
            )
            meal_plan_output.change(
                lambda x: qr_code_interface(x),
                inputs=[meal_plan_output],
                outputs=qr_code_output,
            )

        with gr.TabItem("🎲 Nutrition Bingo"):
            bingo_output = gr.Dataframe(label="Nutrition Bingo Board")
            bingo_button = gr.Button("New Board")
            bingo_button.click(nutrition_bingo_interface, outputs=bingo_output)


        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                This is a Kids' Nutrition LLM, a tool to help parents and caregivers make informed decisions about their children's nutrition.
                **Disclaimer:** This is not medical advice. Always consult with a pediatrician or registered dietitian for personalized nutrition advice.
                """
            )

if __name__ == "__main__":
    app.launch()
