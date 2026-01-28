import requests
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from gtts import gTTS
from playsound import playsound

# =============================
# رابط ESP32-CAM
# =============================
CAMERA_URL = "http://192.168.12.90/capture"

# =============================
# ترجمة الأشياء
# =============================
translate = {
    "person": "شخص",
    "chair": "كرسي",
    "table": "طاولة",
    "bottle": "زجاجة",
    "cup": "كوب",
    "cell phone": "هاتف",
    "laptop": "حاسوب محمول",
    "tv": "تلفاز",
    "book": "كتاب",
    "backpack": "حقيبة",
    "door": "باب",
    "window": "نافذة",
    "computer": "حاسوب",
    "keyboard": "لوحة مفاتيح",
    "bench": "مقعد",
    "trash can": "سلة مهملات"
}

# =============================
# نطق عربي (Google)
# =============================
def speak(text):
    tts = gTTS(text=text, lang="ar")
    tts.save("voice.mp3")
    playsound("voice.mp3")
    os.remove("voice.mp3")

# =============================
# تحميل YOLO
# =============================
model = YOLO("yolov8n.pt")

last_sentence = ""

print("🟢 النظام يعمل")
speak("النظام يعمل")

# =============================
# تحليل الاتجاه
# =============================
def get_direction(x_center, width):
    if x_center < width / 3:
        return "على اليسار"
    elif x_center > 2 * width / 3:
        return "على اليمين"
    else:
        return "أمامك"

# =============================
# قريب / بعيد
# =============================
def get_distance(box_area, frame_area):
    ratio = box_area / frame_area
    if ratio > 0.15:
        return "قريب"
    else:
        return "بعيد"

# =============================
# الحلقة الرئيسية
# =============================
while True:
    try:
        r = requests.get(CAMERA_URL, timeout=5)
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)

        h, w, _ = img.shape
        frame_area = h * w

        results = model(img, verbose=False)
        descriptions = []

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                eng = model.names[cls_id]
                ar = translate.get(eng, eng)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) // 2
                area = (x2 - x1) * (y2 - y1)

                direction = get_direction(x_center, w)
                distance = get_distance(area, frame_area)

                desc = f"{ar} {distance} {direction}"
                descriptions.append(desc)

        if descriptions:
            sentence = "أمامك " + " و ".join(descriptions)
        else:
            sentence = "لا أرى شيء واضح"

        # 🔁 لا تكرر الكلام
        if sentence != last_sentence:
            print(sentence)
            speak(sentence)
            last_sentence = sentence

        cv2.imshow("ESP32-CAM", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(3)

    except Exception as e:
        print("❌ خطأ:", e)
        speak("حدث خطأ في الاتصال")
        time.sleep(5)

cv2.destroyAllWindows()
