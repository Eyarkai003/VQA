
import io
import json
import html

import streamlit as st
import torch
import speech_recognition as sr

from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import streamlit.components.v1 as components


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Trustworthy VQA Assistant",
    page_icon="👁️",
    layout="centered"
)

torch.set_num_threads(2)

MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"

st.markdown("""
<style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 750;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #777777;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------

@st.cache_resource(show_spinner=False)
def load_vqa_model():
    processor = ViltProcessor.from_pretrained(MODEL_NAME)
    model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


# ---------------- VOICE OUTPUT ----------------

def speak_answer(text):
    safe_text = json.dumps(text)
    components.html(
        f"""
        <button onclick='
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance({safe_text});
            msg.lang = "en-IN";
            window.speechSynthesis.speak(msg);
        ' style="
            padding:12px 20px;
            border-radius:10px;
            border:0;
            background:#1769aa;
            color:white;
            font-size:16px;
            cursor:pointer;">
            🔊 Speak Answer
        </button>
        """,
        height=65
    )


# ---------------- VOICE QUESTION INPUT ----------------

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(io.BytesIO(audio_file.getvalue())) as source:
        audio_data = recognizer.record(source)

    return recognizer.recognize_google(
        audio_data,
        language="en-IN"
    )


# ---------------- AI ANSWERING ----------------

def answer_question(image, question, processor, model):
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits[0], dim=-1
        )

    count = min(3, probabilities.numel())
    scores, indices = torch.topk(probabilities, k=count)

    results = []

    for score, index in zip(scores.tolist(), indices.tolist()):
        label = model.config.id2label.get(
            index, str(index)
        )
        results.append({
            "answer": str(label),
            "score": float(score)
        })

    return results


# ---------------- MAIN INTERFACE ----------------

st.markdown(
    '<div class="main-title">👁️ Trustworthy VQA Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about an image using text or voice.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Assistive technology demo: AI answers may be incorrect. "
    "Do not rely on this prototype alone for navigation, "
    "emergency decisions, or other safety-critical tasks."
)

st.subheader("1. Capture or upload an image")

input_method = st.radio(
    "Choose image input",
    ["Use Camera", "Upload Image"],
    horizontal=True
)

image_file = None

if input_method == "Use Camera":
    image_file = st.camera_input(
        "Take a picture",
        key="camera"
    )
else:
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

image = None

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(
            image,
            caption="Image for AI analysis",
            use_container_width=True
        )
    except Exception:
        st.error("The image could not be opened. Try another image.")
        image = None


st.subheader("2. Ask a question")

question_text = st.text_input(
    "Type your question",
    placeholder="Example: What is in front of me?",
    key="typed_question"
)

st.caption("Or record your question in English.")

audio_file = st.audio_input(
    "🎙️ Record a question",
    sample_rate=16000
)

if audio_file is not None:
    if st.button("Convert voice to text"):
        with st.spinner("Recognizing your speech..."):
            try:
                recognized_text = transcribe_audio(audio_file)
                st.session_state["voice_question"] = recognized_text
                st.success("Recognized question:")
                st.write(recognized_text)
            except sr.UnknownValueError:
                st.error("Speech was unclear. Please record again.")
            except sr.RequestError:
                st.error(
                    "Speech recognition service is unavailable. "
                    "Check your internet connection and try again."
                )
            except Exception as error:
                st.error(f"Audio processing failed: {error}")


voice_question = st.session_state.get("voice_question", "")
question = question_text.strip() or voice_question.strip()

st.caption(
    "Typed question takes priority. Otherwise, the converted "
    "voice question is used."
)


st.subheader("3. Get the AI answer")

if st.button("🔍 Analyze Image and Answer", type="primary",
             use_container_width=True):

    if image is None:
        st.warning("Please capture or upload an image first.")

    elif not question:
        st.warning("Please type a question or convert a voice question.")

    else:
        try:
            with st.spinner(
                "Loading the AI model and analyzing your image..."
            ):
                processor, model = load_vqa_model()
                results = answer_question(
                    image, question, processor, model
                )

            st.markdown("### Your question")
            st.write(question)

            st.markdown("### AI answer")
            st.success(results[0]["answer"])

            st.caption(
                "This answer is generated by a pretrained AI model "
                "and may not describe the image correctly."
            )

            st.markdown("### Alternative answers")

            for number, result in enumerate(results, start=1):
                st.write(
                    f"{number}. {result['answer']} "
                    f"(model score: {result['score']:.3f})"
                )

            st.warning(
                "Uncertainty note: these scores are normalized model "
                "scores, NOT calibrated probabilities of correctness. "
                "Reliability must be evaluated separately."
            )

            st.markdown("### Listen to the answer")
            speak_answer(
                "The AI answer is " + results[0]["answer"]
            )

            st.session_state["last_answer"] = results[0]["answer"]
            st.session_state["last_question"] = question

        except Exception as error:
            st.error(
                "The AI could not process this request. "
                "Check the app logs for the technical error."
            )
            with st.expander("Technical details"):
                st.code(str(error))


# ---------------- PROJECT INFORMATION ----------------

with st.expander("About this research project"):
    st.write("""
    **Research focus:** Trustworthy Visual Question Answering
    for assistive technology.

    **Baseline model:** ViLT fine-tuned on VQAv2.

    **Current prototype features:**
    - Image capture and upload
    - Typed questions
    - Voice recording and speech recognition
    - AI answer generation
    - Alternative answers and model scores
    - Browser-based speech output
    - An explicit warning about uncertainty

    **Research limitation:** Evidence grounding and calibrated
    uncertainty estimation are not fully implemented in this
    baseline. They require additional methods and evaluation.
    """)
