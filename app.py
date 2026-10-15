import streamlit as st
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Trustworthy Visual Question Answering",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Trustworthy Visual Question Answering")
st.caption(
    "Evidence-aware image question answering with uncertainty warnings "
    "for assistive technology."
)

st.info(
    "Research prototype: AI answers may be incorrect. "
    "Please verify important information."
)

# ---------------- LOAD MODEL ----------------
MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"

@st.cache_resource
def load_model():
    processor = ViltProcessor.from_pretrained(MODEL_NAME)
    model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Project Modules")
    st.markdown("""
    1. Image Input
    2. Question Processing
    3. Visual Question Answering
    4. Evidence Inspection
    5. Uncertainty Estimation
    6. Answer History
    """)
    st.warning(
        "The displayed model score is a confidence-like score, "
        "not a calibrated probability that the answer is correct."
    )

# ---------------- IMAGE INPUT ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Ask a Question")
        question = st.text_input(
            "Question about the image",
            placeholder="Example: What objects are visible?"
        )

        uncertainty_threshold = st.slider(
            "Low-score warning threshold",
            min_value=0.05,
            max_value=0.80,
            value=0.20,
            step=0.05
        )

        ask_button = st.button(
            "Answer Question",
            type="primary"
        )

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Analyzing the image and question..."):
                    processor, model = load_model()

                    inputs = processor(
                        images=image,
                        text=question.strip(),
                        return_tensors="pt"
                    )

                    with torch.inference_mode():
                        outputs = model(**inputs)

                    probabilities = torch.softmax(
                        outputs.logits[0], dim=-1
                    )

                    top_count = min(5, probabilities.numel())
                    scores, indices = torch.topk(
                        probabilities, k=top_count
                    )

                    answers = []
                    for score, index in zip(scores, indices):
                        answer = model.config.id2label[index.item()]
                        answers.append({
                            "answer": answer,
                            "score": float(score.item())
                        })

                    best_answer = answers[0]["answer"]
                    best_score = answers[0]["score"]

                st.subheader("Answer")
                st.success(best_answer)

                st.metric(
                    "Model score",
                    f"{best_score:.1%}"
                )

                # This threshold is a heuristic, not a validated
                # probability of correctness.
                if best_score < uncertainty_threshold:
                    st.warning(
                        "Uncertainty warning: the model's top-answer "
                        "score is low. Do not rely on this answer "
                        "without checking the image."
                    )
                else:
                    st.info(
                        "The top-answer score passed the selected "
                        "threshold, but the answer may still be wrong."
                    )

                # ---------------- EVIDENCE INSPECTION ----------------
                st.subheader("Evidence Inspection")
                st.image(
                    image,
                    caption="Original image used by the model",
                    use_container_width=True
                )

                st.write(
                    "*Question:*",
                    question.strip()
                )
                st.write(
                    "*Predicted answer:*",
                    best_answer
                )

                st.caption(
                    "This version displays the source image as context. "
                    "It does not yet generate object-level grounding "
                    "or prove which image region supports the answer."
                )

                # ---------------- TOP CANDIDATES ----------------
                st.subheader("Alternative Answers")

                for rank, item in enumerate(answers, start=1):
                    st.write(
                        f"{rank}. *{item['answer']}* — "
                        f"model score: {item['score']:.1%}"
                    )

                # ---------------- SAVE HISTORY ----------------
                st.session_state.history.insert(0, {
                    "question": question.strip(),
                    "answer": best_answer,
                    "score": best_score
                })

            except Exception as error:
                st.error(
                    "The model could not process this request. "
                    "Check your internet connection and installed "
                    "dependencies, then try again."
                )
                st.exception(error)

else:
    st.markdown("### How to use")
    st.write("1. Upload a JPG or PNG image.")
    st.write("2. Type a question about the image.")
    st.write("3. Select *Answer Question*.")
    st.write("4. Review the answer, alternative answers, and warning.")

# ---------------- HISTORY ----------------
st.divider()
st.subheader("Question History")

if st.session_state.history:
    for index, item in enumerate(st.session_state.history, start=1):
        with st.expander(
            f"{index}. {item['question']}"
        ):
            st.write("*Answer:*", item["answer"])
            st.write(
                f"*Model score:* {item['score']:.1%}"
            )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("No questions asked in this session yet.")