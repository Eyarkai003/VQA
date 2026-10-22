
import io
import json

import streamlit as st
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Trustworthy Visual Question Answering",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ Trustworthy Visual Question Answering")
st.caption("AI-powered visual assistance for everyday objects")

# CPU inference: suitable for machines without a GPU.
torch.set_num_threads(2)


# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = ViltProcessor.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )
    model.eval()
    return processor, model


def get_answer(image, question):
    processor, model = load_model()

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = outputs.logits[0]
    probabilities = torch.softmax(scores, dim=-1)

    count = min(3, probabilities.numel())
    values, indices = torch.topk(probabilities, count)

    answers = []
    for score, index in zip(values.tolist(), indices.tolist()):
        label = model.config.id2label.get(index, str(index))
        answers.append({
            "answer": str(label),
            "score": float(score),
        })

    return answers


# ---------------- RESET ----------------
if "page_version" not in st.session_state:
    st.session_state.page_version = 0

if "result" not in st.session_state:
    st.session_state.result = None

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""


def reset_app():
    st.session_state.page_version += 1
    st.session_state.result = None
    st.session_state.voice_question = ""
    st.session_state.reset_message = True


st.button(
    "🔄 New Question / Reset",
    on_click=reset_app,
    use_container_width=True,
)

if st.session_state.pop("reset_message", False):
    st.success("Previous question, answer, and recording cleared.")

version = st.session_state.page_version


# ---------------- IMAGE INPUT ----------------
st.subheader("1. Select an image")

image_source = st.radio(
    "Choose input",
    ["Camera", "Upload image"],
    horizontal=True,
    key=f"source_{version}",
)

image = None

if image_source == "Camera":
    photo = st.camera_input(
        "Take a picture of an object",
        key=f"camera_{version}",
    )
else:
    photo = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"upload_{version}",
    )

if photo is not None:
    try:
        image = Image.open(photo).convert("RGB")
        st.image(
            image,
            caption="Image selected for analysis",
            use_container_width=True,
        )
    except Exception:
        st.error("This image could not be opened. Try another photo.")


# ---------------- QUESTION INPUT ----------------
st.subheader("2. Ask a question")

typed_question = st.text_input(
    "Type your question",
    placeholder="Example: Is there a water bottle?",
    key=f"question_{version}",
)

audio = st.audio_input(
    "Or record your question",
    key=f"audio_{version}",
)

if st.button(
    "🎤 Convert Recording to Text",
    key=f"transcribe_{version}",
):
    if audio is None:
        st.warning("Record your question first.")
    elif sr is None:
        st.error("SpeechRecognition is not installed.")
    else:
        try:
            recognizer = sr.Recognizer()
            audio_bytes = io.BytesIO(audio.getvalue())

            with sr.AudioFile(audio_bytes) as source:
                recording = recognizer.record(source)

            with st.spinner("Converting speech to text..."):
                transcript = recognizer.recognize_google(recording)

            st.session_state.voice_question = transcript
            st.success("Recognized question: " + transcript)

        except sr.UnknownValueError:
            st.warning("Speech was unclear. Please record again.")
        except Exception as exc:
            st.error(
                "Voice transcription failed. You can type "
                "your question instead."
            )
            st.caption(str(exc))

if st.session_state.voice_question:
    st.write(
        "**Recognized voice question:**",
        st.session_state.voice_question,
    )

question = (
    typed_question.strip()
    or st.session_state.voice_question.strip()
)


# ---------------- ANALYSIS ----------------
st.subheader("3. AI analysis")

if st.button(
    "🔍 Analyze Image and Answer",
    type="primary",
    use_container_width=True,
    key=f"analyze_{version}",
):
    if image is None:
        st.error("Please take a picture or upload an image.")
    elif not question:
        st.error("Please type or record a question.")
    else:
        try:
            with st.spinner(
                "Loading the AI model and analyzing the image. "
                "The first run may take several minutes..."
            ):
                answers = get_answer(image, question)

            st.session_state.result = {
                "question": question,
                "answers": answers,
                "image": image.copy(),
            }

        except Exception as exc:
            st.session_state.result = None
            st.error("The AI analysis failed.")
            st.write("Error details:")
            st.code(str(exc))
            st.info(
                "If this continues, check the Streamlit app logs. "
                "The model may have failed to download or load."
            )


# ---------------- DISPLAY RESULT ----------------
result = st.session_state.result

if result is not None:
    st.divider()
    st.subheader("4. Result")

    st.image(
        result["image"],
        caption="Analyzed image",
        use_container_width=True,
    )

    st.write("**Your question:**", result["question"])

    answers = result["answers"]

    if answers:
        answer = answers[0]["answer"]
        st.markdown("### AI answer")
        st.success(answer)

        with st.expander("View alternative model answers"):
            for item in answers:
                st.write(
                    f'{item["answer"]} '
                    f'(model score: {item["score"]:.3f})'
                )

        st.caption(
            "Model scores are not calibrated confidence estimates. "
            "Answers may be incorrect or incomplete."
        )

        # Text-to-speech runs in the browser.
        speech_text = json.dumps(answer)

        st.components.v1.html(
            f"""
            <button id="speakButton"
              style="padding:12px;font-size:16px;">
              🔊 Speak Answer
            </button>
            <script>
              const answerText = {speech_text};
              document.getElementById("speakButton")
                .addEventListener("click", () => {{
                  if (!("speechSynthesis" in window)) {{
                    alert("Speech is not supported by this browser.");
                    return;
                  }}
                  window.speechSynthesis.cancel();
                  const utterance =
                    new SpeechSynthesisUtterance(answerText);
                  window.speechSynthesis.speak(utterance);
                }});
            </script>
            """,
            height=60,
        )

st.divider()
st.caption(
    "Research prototype. Verify important answers independently. "
    "Do not rely on this system alone for navigation or safety."
)
