

import io
import json
import re

import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import (
    ViltProcessor,
    ViltForQuestionAnswering,
    OwlViTProcessor,
    OwlViTForObjectDetection,
)

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Trustworthy Visual Question Answering",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ Trustworthy Visual Question Answering")
st.caption(
    "Evidence-guided visual assistance for everyday objects"
)

st.info(
    "Take a photo, ask a question by typing or voice, "
    "and receive an AI-generated answer."
)

# CPU mode helps compatibility with laptops without a GPU.
DEVICE = "cpu"

# Several phrases help OWL-ViT recognize different bottle types.
# They are prompts, not a guarantee of perfect detection.
PROMPTS = [
    ("water bottle", "a photo of a water bottle"),
    ("water bottle", "a photo of a plastic drinking water bottle"),
    ("water bottle", "a photo of a reusable water bottle"),
    ("water bottle", "a photo of a transparent water bottle"),
    ("water bottle", "a photo of a bottle filled with water"),
    ("bottle", "a photo of a bottle"),
    ("bag", "a photo of a bag"),
    ("bag", "a photo of a handbag"),
    ("bag", "a photo of a backpack"),
    ("cup", "a photo of a drinking cup"),
    ("glass", "a photo of a drinking glass"),
    ("mobile phone", "a photo of a mobile phone"),
    ("laptop", "a photo of a laptop computer"),
    ("book", "a photo of a book"),
    ("chair", "a photo of a chair"),
    ("table", "a photo of a table"),
    ("person", "a photo of a person"),
    ("keys", "a photo of keys"),
    ("shoe", "a photo of a shoe"),
    ("umbrella", "a photo of an umbrella"),
    ("wallet", "a photo of a wallet"),
    ("pen", "a photo of a pen"),
    ("remote control", "a photo of a television remote control"),
    ("plate", "a photo of a plate"),
    ("bowl", "a photo of a bowl"),
    ("box", "a photo of a box"),
    ("sofa", "a photo of a sofa"),
    ("bed", "a photo of a bed"),
    ("fan", "a photo of an electric fan"),
    ("television", "a photo of a television"),
]

LABELS = [item[0] for item in PROMPTS]
TEXT_PROMPTS = [item[1] for item in PROMPTS]


# ---------------- LOAD MODELS ON DEMAND ----------------
@st.cache_resource(show_spinner=False)
def load_vqa_model():
    processor = ViltProcessor.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_detector():
    processor = OwlViTProcessor.from_pretrained(
        "google/owlvit-base-patch32"
    )
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    )
    model.eval()
    return processor, model


# ---------------- RESET ----------------
if "ui_version" not in st.session_state:
    st.session_state.ui_version = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def reset_app():
    # Changing widget keys creates fresh media/input widgets.
    st.session_state.ui_version += 1
    st.session_state.last_result = None
    st.session_state.voice_question = ""
    st.session_state.reset_notice = True


if st.button(
    "🔄 New Question / Reset",
    on_click=reset_app,
    use_container_width=True,
):
    pass

if st.session_state.pop("reset_notice", False):
    st.success(
        "Reset complete. Previous answers cleared. "
        "Capture or upload a new image and ask again."
    )

version = st.session_state.ui_version


# ---------------- IMAGE INPUT ----------------
st.subheader("1. Choose an image")

source = st.radio(
    "Image source",
    ["Camera", "Upload image"],
    horizontal=True,
    key=f"source_{version}",
)

image = None

if source == "Camera":
    camera_file = st.camera_input(
        "Take a picture",
        key=f"camera_{version}",
    )
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
else:
    uploaded_file = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"upload_{version}",
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    st.image(image, caption="Current image", use_container_width=True)


# ---------------- QUESTION INPUT ----------------
st.subheader("2. Ask a question")

question = st.text_input(
    "Type your question",
    placeholder="Example: Is there a water bottle in this picture?",
    key=f"question_{version}",
)

st.caption("Or record your question below.")

audio = st.audio_input(
    "Record your question",
    key=f"audio_{version}",
)

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

if audio is not None:
    if sr is None:
        st.warning(
            "Voice transcription is unavailable because "
            "SpeechRecognition is not installed."
        )
    else:
        try:
            recognizer = sr.Recognizer()
            audio_bytes = io.BytesIO(audio.getvalue())

            with sr.AudioFile(audio_bytes) as audio_source:
                audio_data = recognizer.record(audio_source)

            with st.spinner("Transcribing your voice..."):
                transcript = recognizer.recognize_google(
                    audio_data
                )

            st.session_state.voice_question = transcript
            st.success("Recognized question: " + transcript)

        except sr.UnknownValueError:
            st.warning(
                "I could not understand that recording. "
                "Please record again or type your question."
            )
        except Exception as exc:
            st.warning(
                "Voice transcription failed. You can still "
                "type your question. Details: " + str(exc)
            )

voice_question = st.session_state.voice_question

if voice_question:
    st.write("**Voice question:**", voice_question)

effective_question = question.strip() or voice_question.strip()

# ---------------- DETECT OBJECTS ----------------
def detect_objects(pil_image):
    processor, model = load_detector()

    inputs = processor(
        text=[TEXT_PROMPTS],
        images=pil_image,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [[pil_image.height, pil_image.width]]
    )

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.10,
    )

    found = []

    for score, label_id, box in zip(
        results[0]["scores"].tolist(),
        results[0]["labels"].tolist(),
        results[0]["boxes"].tolist(),
    ):
        if label_id < 0 or label_id >= len(LABELS):
            continue

        label = LABELS[label_id]

        # Keep only reasonably confident predictions.
        # These scores are not calibrated probabilities.
        if score < 0.15:
            continue

        found.append({
            "label": label,
            "score": float(score),
            "box": box,
        })

    # Suppress duplicate prompts for the same object when
    # their bounding boxes substantially overlap.
    found.sort(key=lambda item: item["score"], reverse=True)
    unique = []

    for item in found:
        x1, y1, x2, y2 = item["box"]
        duplicate = False

        for previous in unique:
            if item["label"] != previous["label"]:
                continue

            a1, b1, a2, b2 = previous["box"]
            ix1 = max(x1, a1)
            iy1 = max(y1, b1)
            ix2 = min(x2, a2)
            iy2 = min(y2, b2)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = max(0, x2 - x1) * max(0, y2 - y1)
            area2 = max(0, a2 - a1) * max(0, b2 - b1)
            union = area1 + area2 - intersection

            if union > 0 and intersection / union > 0.50:
                duplicate = True
                break

        if not duplicate:
            unique.append(item)

    return unique


# ---------------- VQA ANSWERING ----------------
def answer_question(pil_image, user_question):
    processor, model = load_vqa_model()

    inputs = processor(
        images=pil_image,
        text=user_question,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = outputs.logits.softmax(dim=-1)[0]
    top = torch.topk(
        probabilities,
        k=min(3, probabilities.numel()),
    )

    answers = []

    for score, index in zip(
        top.values.tolist(),
        top.indices.tolist(),
    ):
        label = model.config.id2label.get(
            index, str(index)
        )
        answers.append({
            "answer": str(label),
            "score": float(score),
        })

    return answers


# ---------------- ANALYZE ----------------
st.subheader("3. Analyze")

if st.button(
    "🔍 Analyze Image and Answer",
    type="primary",
    use_container_width=True,
):
    if image is None:
        st.error("Please take a picture or upload an image.")
    elif not effective_question:
        st.error("Please type or record a question.")
    else:
        try:
            with st.spinner(
                "Loading AI models and analyzing your image..."
            ):
                detected = detect_objects(image)
                answers = answer_question(
                    image, effective_question
                )

            # Store the complete result so it remains available
            # after Streamlit reruns the script.
            st.session_state.last_result = {
                "question": effective_question,
                "objects": detected,
                "answers": answers,
                "image": image.copy(),
            }

        except Exception as exc:
            st.error(
                "Analysis failed. Check the app logs for "
                "model-loading or memory errors."
            )
            st.exception(exc)


# ---------------- DISPLAY SAVED RESULT ----------------
result = st.session_state.last_result

if result is not None:
    st.divider()
    st.subheader("4. Results")

    st.write("**Question analyzed:**", result["question"])

    detected = result["objects"]
    bottle_found = any(
        item["label"] == "water bottle"
        for item in detected
    )

    # Draw boxes on a copy of the image.
    annotated = result["image"].copy()
    draw = ImageDraw.Draw(annotated)

    for item in detected:
        x1, y1, x2, y2 = item["box"]
        label_text = (
            f'{item["label"]} ({item["score"]:.2f})'
        )
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )
        draw.text(
            (x1, max(0, y1 - 16)),
            label_text,
            fill="red",
        )

    st.image(
        annotated,
        caption="Detected objects and bounding boxes",
        use_container_width=True,
    )

    st.markdown("**Detected objects**")

    if detected:
        # Show the best score for each object label.
        best_scores = {}
        for item in detected:
            best_scores[item["label"]] = max(
                item["score"],
                best_scores.get(item["label"], 0),
            )

        for label, score in sorted(
            best_scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        ):
            st.write(f"- {label}: score {score:.2f}")
    else:
        st.warning(
            "No objects passed the detection threshold. "
            "This does not prove that the image contains no objects."
        )

    if bottle_found:
        st.success("A water bottle was detected.")
    else:
        st.warning(
            "A water bottle was not confidently detected. "
            "Try a clearer, well-lit image with the whole bottle visible."
        )

    st.markdown("**AI answer**")

    answers = result["answers"]
    best_answer = answers[0]["answer"] if answers else (
        "I could not generate an answer."
    )

    st.write(best_answer)

    with st.expander("Alternative model answers"):
        for item in answers:
            st.write(
                f'{item["answer"]} — model score '
                f'{item["score"]:.3f}'
            )

    st.caption(
        "Model scores are not calibrated confidence estimates. "
        "Object detection and answers can be incorrect. "
        "Verify important information before relying on it."
    )

    # Browser speech synthesis. The user can replay the answer.
    if best_answer:
        speech_text = json.dumps(best_answer)
        st.components.v1.html(
            f"""
            <button id="speakAnswer"
              style="padding:10px 16px; font-size:16px;">
              🔊 Speak answer
            </button>
            <script>
              const answerText = {speech_text};
              document.getElementById("speakAnswer")
                .addEventListener("click", function() {{
                  if ("speechSynthesis" in window) {{
                    window.speechSynthesis.cancel();
                    const utterance =
                      new SpeechSynthesisUtterance(answerText);
                    window.speechSynthesis.speak(utterance);
                  }}
                }});
            </script>
            """,
            height=60,
        )

st.divider()
st.caption(
    "Research prototype for assistive technology. "
    "It may miss objects and should not be used as the sole "
    "source of information for navigation or safety-critical decisions."
)
