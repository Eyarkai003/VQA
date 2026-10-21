

import streamlit as st
import streamlit.components.v1 as components
import torch
import speech_recognition as sr

from PIL import Image
from transformers import (
    ViltProcessor,
    ViltForQuestionAnswering,
    OwlViTProcessor,
    OwlViTForObjectDetection,
)

# ---------------- SETTINGS ----------------

st.set_page_config(
    page_title="Trustworthy VQA Assistant",
    page_icon="👁️",
    layout="centered",
)

torch.set_num_threads(2)

VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
DETECTOR_MODEL = "google/owlvit-base-patch32"

st.title("👁️ Trustworthy VQA Assistant")
st.write(
    "Ask questions about images and detect everyday objects."
)

st.info(
    "This is a research prototype. AI answers and detections "
    "may be incorrect. Do not rely on them for safety-critical decisions."
)


# ---------------- RESET ----------------

if st.button("🔄 New Question / Reset", use_container_width=True):
    for key in [
        "question_text",
        "voice_question",
        "last_answer",
        "last_question",
        "last_objects",
        "voice_audio",
        "uploaded_image",
        "camera_image",
        "image_source",
    ]:
        st.session_state.pop(key, None)

    st.rerun()


# ---------------- LOAD MODELS ----------------

@st.cache_resource(show_spinner=False)
def load_vqa():
    processor = ViltProcessor.from_pretrained(VQA_MODEL)
    model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_detector():
    processor = OwlViTProcessor.from_pretrained(DETECTOR_MODEL)
    model = OwlViTForObjectDetection.from_pretrained(DETECTOR_MODEL)
    model.eval()
    return processor, model


# ---------------- IMAGE INPUT ----------------

st.header("1. Choose an image")

image_source = st.radio(
    "Image input",
    ["Camera", "Upload"],
    horizontal=True,
    key="image_source",
)

image_file = None

if image_source == "Camera":
    image_file = st.camera_input(
        "Take a picture",
        key="camera_image",
    )
else:
    image_file = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploaded_image",
    )

image = None

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(
            image,
            caption="Selected image",
            use_container_width=True,
        )
    except Exception:
        st.error("Could not read this image. Please try another.")
        image = None


# ---------------- QUESTION INPUT ----------------

st.header("2. Ask your question")

question_text = st.text_input(
    "Type a question",
    placeholder="Example: What is in this picture?",
    key="question_text",
)

audio_file = st.audio_input(
    "🎙️ Record your question",
    key="voice_audio",
)

if st.button("Convert voice to text"):
    if audio_file is None:
        st.warning("Record a question first.")
    else:
        try:
            recognizer = sr.Recognizer()

            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)

            with st.spinner("Recognizing speech..."):
                recognized = recognizer.recognize_google(
                    audio_data,
                    language="en-IN",
                )

            st.session_state["voice_question"] = recognized
            st.success("Recognized question")
            st.write(recognized)

        except sr.UnknownValueError:
            st.error("Speech was unclear. Please record again.")
        except sr.RequestError:
            st.error(
                "Speech recognition service is unavailable. "
                "Check your internet connection."
            )
        except Exception as exc:
            st.error(f"Audio error: {exc}")

voice_question = st.session_state.get("voice_question", "")
question = question_text.strip() or voice_question.strip()


# ---------------- OBJECT DETECTION ----------------

def detect_objects(image):
    processor, model = load_detector()

    labels = [
        "water bottle",
        "bag",
        "handbag",
        "backpack",
        "cup",
        "glass",
        "chair",
        "table",
        "book",
        "mobile phone",
        "keys",
        "shoe",
        "person",
        "door",
        "laptop",
        "bowl",
        "plate",
        "umbrella",
        "wallet",
        "remote control",
        "pen",
        "watch",
        "bottle",
        "box",
        "sofa",
        "bed",
        "fan",
        "television",
        "food",
        "clothing",
    ]

    inputs = processor(
        text=[labels],
        images=image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.10,
        text_labels=[labels],
    )[0]

    objects = []

    for score, label, box in zip(
        results["scores"],
        results["text_labels"],
        results["boxes"],
    ):
        objects.append({
            "label": str(label),
            "score": float(score),
            "box": [round(float(x)) for x in box.tolist()],
        })

    return objects


# ---------------- VQA ANSWERING ----------------

def answer_question(image, question):
    processor, model = load_vqa()

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1)

    count = min(3, probs.numel())
    scores, indices = torch.topk(probs, k=count)

    answers = []

    for score, index in zip(scores.tolist(), indices.tolist()):
        answers.append({
            "answer": model.config.id2label.get(index, str(index)),
            "score": float(score),
        })

    return answers


# ---------------- RUN ANALYSIS ----------------

st.header("3. Analyze")

if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
    if image is None:
        st.warning("Capture or upload an image first.")
    elif not question:
        st.warning("Type a question or convert a voice recording first.")
    else:
        try:
            with st.spinner("Loading AI models and analyzing..."):
                answers = answer_question(image, question)
                objects = detect_objects(image)

            st.session_state["last_answer"] = answers[0]["answer"]
            st.session_state["last_question"] = question
            st.session_state["last_objects"] = objects

        except Exception as exc:
            st.error(
                "Analysis failed. Check the app logs for details."
            )
            st.code(str(exc))

if "last_answer" in st.session_state:
    st.subheader("AI Answer")
    st.success(st.session_state["last_answer"])

    st.write("**Question:**", st.session_state["last_question"])

    st.caption(
        "Model scores are not calibrated probabilities of correctness."
    )

    st.subheader("Alternative answers")

    for item in answers if "answers" in locals() else []:
        st.write(
            f"{item['answer']} — model score {item['score']:.3f}"
        )

    answer_text = st.session_state["last_answer"]

    st.subheader("Listen to the answer")
    components.html(
        f"""
        <button id="speakBtn" style="
            padding:12px 18px;
            border:0;
            border-radius:9px;
            background:#1769aa;
            color:white;
            font-size:16px;">
            🔊 Speak Answer
        </button>
        <script>
        document.getElementById("speakBtn").onclick = function() {{
            window.speechSynthesis.cancel();
            const message = new SpeechSynthesisUtterance(
                {__import__('json').dumps(answer_text)}
            );
            message.lang = "en-IN";
            window.speechSynthesis.speak(message);
        }};
        </script>
        """,
        height=65,
    )

    st.subheader("Detected everyday objects")

    objects = st.session_state.get("last_objects", [])

    if objects:
        seen = set()
        for obj in objects:
            label = obj["label"]
            if label not in seen:
                st.write(
                    f"• {label} "
                    f"(detection score: {obj['score']:.2f})"
                )
                seen.add(label)
    else:
        st.write(
            "No objects were detected from the selected label list. "
            "This does not prove the image contains no objects."
        )

    st.warning(
        "Check important answers visually. Object detection can miss "
        "items or incorrectly label similar objects."
    )
