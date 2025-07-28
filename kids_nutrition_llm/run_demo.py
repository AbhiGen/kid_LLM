import gradio as gr
import pandas as pd
import qrcode
from io import BytesIO
from transformers import pipeline

# Load the base model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Load the nutrition dataset for the nutrition facts tab
nutrition_df = pd.read_csv("data/nutrition_dataset.csv")

# --- UI Functions ---

def chat_interface(message, history, age, gender, allergies, dietary_restrictions):
    """The main chat interface."""
    # Add demographic information to the prompt
    prompt = f"Age: {age}, Gender: {gender}, Allergies: {allergies}, Dietary Restrictions: {dietary_restrictions}\n\n{message}"
    response = chatbot(prompt, max_length=150)[0]["generated_text"]
    return response

def meal_planner_interface(age, gender, allergies, dietary_restrictions):
    """The meal planner interface."""
    # This is a placeholder for the growth-aware meal planner
    prompt = f"Create a healthy one-day meal plan for a {age}-year-old {gender} with {allergies} allergies and {dietary_restrictions} dietary restrictions."
    meal_plan = chatbot(prompt, max_length=300)[0]["generated_text"]
    return meal_plan

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
    gr.Markdown("# Kids' Nutrition LLM (Demo Mode)")

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
                    plan_button = gr.Button("Generate Meal Plan")
                with gr.Column():
                    meal_plan_output = gr.Textbox(label="Your Meal Plan")
                    qr_code_output = gr.Image(label="Shareable QR Code", type="pil")

            plan_button.click(
                meal_planner_interface,
                inputs=[mp_age_input, mp_gender_input, mp_allergies_input, mp_dietary_restrictions_input],
                outputs=meal_plan_output,
            )
            meal_plan_output.change(
                lambda x: qr_code_interface(x),
                inputs=[meal_plan_output],
                outputs=qr_code_output,
            )


        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                This is a Kids' Nutrition LLM, a tool to help parents and caregivers make informed decisions about their children's nutrition.
                **Disclaimer:** This is not medical advice. Always consult with a pediatrician or registered dietitian for personalized nutrition advice.
                """
            )

if __name__ == "__main__":
    app.launch()
