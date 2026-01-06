## hardik jangir: 22b3901
## archishman : 22b2405

## the video link for the demo:  https://drive.google.com/file/d/171dNHXwYJxIV4NB6TBJpw3M1FCYS_05x/view?usp=sharing


## the video link for the code explaination:   https://drive.google.com/file/d/1dKJFB9svGhfQk4gRgzV9NYpx6hybUYcC/view?usp=sharing

import speech_recognition as sr
import threading
import queue
import time
from gtts import gTTS
from playsound import playsound
import os
import cv2
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine
from datetime import datetime
from huggingface_hub import InferenceClient  # Hugging Face for LLM
import pywhatkit

# ---------------- Global Config ----------------
TRUSTED_EMBEDDINGS_FILE = "trusted_embeddings.pkl"
trusted_embeddings = pickle.load(open(TRUSTED_EMBEDDINGS_FILE, "rb"))

recognizer = sr.Recognizer()
mic = sr.Microphone()

# States
state = "S0"
current_user = None
intruder_level = 0

# Commands
ON_COMMANDS = ["guard my room", "start guard", "start",
               "it is the time"]
OFF_COMMANDS = ["do not guard my room", "stop guard",
                "stop", "it is not the time"]

hf_token = "************"

# WhatsApp settings
ENABLE_WHATSAPP_ALERT = True  # <-- Set True/False here
WHATSAPP_TO_NUMBER = "+91xxxxxxxxxx"  # Replace with your number with country code, removing my numbers

# ---------------- TTS Queue ----------------
tts_queue = queue.Queue()

def tts_worker():
    while True:
        msg = tts_queue.get()
        try:
            tts = gTTS(text=msg, lang='en')
            tmp_file = "temp_tts.mp3"
            tts.save(tmp_file)
            playsound(tmp_file)
            os.remove(tmp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(msg):
    print(f"Speak: {msg}")
    
    tts_queue.put(msg)

# ---------------- Face Recognition ----------------
def match_face_input(input_img_path, show_bbox=False, threshold=0.38):
    try:
        faces = DeepFace.extract_faces(
            img_path=input_img_path,
            detector_backend="retinaface",
            enforce_detection=True
        )
    except Exception as e:
        print(f"❌ No face detected: {e}")
        return None, None, 1.0

    best_score = 1.0
    best_user = None

    for face in faces:
        bbox = face["facial_area"]
        (x, y, w, h) = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])

        if show_bbox:
            img = cv2.imread(input_img_path)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face Detection", img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        img = cv2.imread(input_img_path)
        cropped = img[y:y+h, x:x+w]
        cropped_rgb_path = "temp_cropped_face.jpg"
        cv2.imwrite(cropped_rgb_path, cropped)

        embedding_new = DeepFace.represent(
            img_path=cropped_rgb_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        for user, ref_emb in trusted_embeddings.items():
            score = cosine(embedding_new, ref_emb)
            print(f"[Cosine Score] {user} vs input image: {score:.4f}")
            if score < best_score:
                best_score = score 
                
                best_user = user

        os.remove(cropped_rgb_path)

    print(f"\n✅ Best match: {best_user} with score: {best_score:.4f}")
    if best_score < threshold:
        return True, best_user, best_score
    else:
        return False, None, best_score

def capture_face_snapshot(temp_image="snapshot.jpg"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None
    cv2.imwrite(temp_image, frame)
    return temp_image, frame

def save_flagged_user(frame, folder="flagged_unrecognized_guys"):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"user_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

# ---------------- WhatsApp ----------------
def send_whatsapp_alert(image_path, caption, to_number=WHATSAPP_TO_NUMBER):
    if not ENABLE_WHATSAPP_ALERT:
        return
    try:
        print(f"Sending WhatsApp alert to {to_number} ...")
        pywhatkit.sendwhats_image(to_number, image_path, caption, wait_time=40, tab_close=True, close_time=5)
        print("WhatsApp alert sent.")
    except Exception as e:
        print(f"❌ WhatsApp alert error: {e}")

# ---------------- LLM ----------------
def get_llm_response(level, user_text):
    client = InferenceClient(token=hf_token, model="mistralai/Mistral-7B-Instruct-v0.3")
    prompts = {
        1: "Level 1: Politely ask who they are. Keep it short and natural.",
        2: "Level 2: Firmly ask them to leave. Keep it short and natural.",
        3: "Level 3: Give a stern warning or alarm. Keep it short, spoken-style."
    }
    prompt = f"{prompts.get(level, prompts[3])} User said: '{user_text}'"

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a security guard AI responding to intruders."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ---------------- State Machine ----------------
def guard_state_control(text):
    global state, current_user, intruder_level

    text = text.lower().strip()
    print(f"🎤 Recognized text: '{text}' | Current State: {state}")

    if state == "S0":
        if any(cmd in text for cmd in ON_COMMANDS):
            print("🔹 Voice ON detected → S1")
            state = "S1"
            temp_img, frame = capture_face_snapshot()
            if temp_img:
                match, user, score = match_face_input(temp_img)
                if match:
                    current_user = user
                    state = "S2"
                    intruder_level = 0
                    speak(f"Welcome {user}, guard activated.")
                else:
                    path = save_flagged_user(frame)
                    speak("Intruder detected! Access denied.")
                    if ENABLE_WHATSAPP_ALERT:
                        send_whatsapp_alert(path, "Unrecognized person detected! System in danger, please come fast.")
                    state = "S0"
                os.remove(temp_img)

    elif state == "S2":
        if any(cmd in text for cmd in OFF_COMMANDS):
            print("🔹 Voice OFF detected → S3")
            state = "S3"
            temp_img, frame = capture_face_snapshot()
            if temp_img:
                match, user, score = match_face_input(temp_img)
                if match and user == current_user:
                    speak("Guard deactivated. Goodbye.")
                    state = "S0"
                    current_user = None
                    intruder_level = 0
                else:
                    path = save_flagged_user(frame)
                    speak("Unrecognized user. Access denied.")
                    if ENABLE_WHATSAPP_ALERT:
                        send_whatsapp_alert(path, "Unrecognized person detected! System in danger, please come fast.")
                    state = "S2"
                os.remove(temp_img)
        else:
            intruder_level = min(intruder_level + 1, 3)
            try:
                llm_response = get_llm_response(intruder_level, text)
                speak(llm_response)
                if intruder_level == 3:
                    temp_img, frame = capture_face_snapshot()
                    if temp_img:
                        path = save_flagged_user(frame)
                        if ENABLE_WHATSAPP_ALERT:
                            send_whatsapp_alert(path, "Unrecognized person detected! System in danger, please come fast.")
                        os.remove(temp_img)
            except Exception as e:
                print(f"LLM Error: {e}")
                fallback = {1: "Who are you?", 2: "Please leave now.", 3: "WARNING: Intruder alert! Leave immediately!"}
                speak(fallback[intruder_level])

# ---------------- Callback ----------------
def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        guard_state_control(text)
    except sr.UnknownValueError:
        print("❌ Could not understand speech")
    except sr.RequestError as e:
        print(f"⚠️ API Error: {e}")

# ---------------- Main ----------------
if __name__ == "__main__":
    with mic as source:
        print("🎧 Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=3)
        print(f"✅ Calibration done. Energy threshold: {recognizer.energy_threshold}")

    stop_listening = recognizer.listen_in_background(mic, callback)
    print("🟢 Guard monitoring started.")

    try:
        while True:
            print(f"🛡 Current State: {state} | Intruder Level: {intruder_level}")
            time.sleep(2)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        print("🛑 Stopping guard monitoring...")
