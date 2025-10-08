import os
import streamlit as st
from huggingface_hub import InferenceClient
from config import HF_TOKEN

# Initialize Hugging Face InferenceClient with Fireworks AI provider
client = InferenceClient(
    provider="fireworks-ai",
    api_key="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",
)

# Setup session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I'm TalentGenius, your AI Hiring Assistant."}
    ]

st.title("🧠 TalentGenius - AI Hiring Assistant")

st.markdown("Fill out the form below and get technical questions based on your tech stack.")

with st.form("candidate_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50)
    position = st.text_input("Desired Position(s)")
    location = st.text_input("Current Location")
    tech_stack = st.text_area("Tech Stack (comma-separated: e.g., Python, Django, MySQL)")

    submitted = st.form_submit_button("Generate Questions")

if submitted:
    if not tech_stack.strip():
        st.warning("Please enter your tech stack.")
    else:
        # Prepare prompt for chat model
        stack_list = [tech.strip() for tech in tech_stack.split(",") if tech.strip()]
        prompt = f"""Act as a technical interviewer. For each of the following technologies, generate 3 technical interview questions assessing intermediate to advanced skills:\n\n{', '.join(stack_list)}\n\nList them clearly."""

        # Prepare messages in chat format
        messages = [
            {"role": "system", "content": "You are an expert technical interviewer."},
            {"role": "user", "content": prompt}
        ]

        with st.spinner("Generating questions..."):
            try:
                completion = client.chat.completions.create(
                    model="Qwen/Qwen3-235B-A22B",
                    messages=messages
                )
                answer = completion.choices[0].message["content"]

                st.success("Here are your tailored interview questions:")
                st.markdown(f"```markdown\n{answer}\n```")

                # Save candidate data locally
                candidate_data = {
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "experience": experience,
                    "position": position,
                    "location": location,
                    "tech_stack": stack_list,
                    "generated_questions": answer
                }

                os.makedirs("data", exist_ok=True)
                with open("data/candidates.json", "a", encoding="utf-8") as f:
                    f.write(str(candidate_data) + "\n")

                st.success("✅ Thank you! We'll contact you soon with further steps.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("© 2025 TalentGenius. All rights reserved.")
