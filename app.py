"""
DR Detection - Flask Microservice
==================================
รับรูปภาพ → ONNX inference → ส่งผลลัพธ์กลับ JSON
"""

import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # อนุญาต Next.js เรียกข้าม origin

# ─── CONFIG ───────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model_dr.onnx")
IMG_SIZE   = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DR grade mapping
GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

RISK_LEVELS = [
    "ไม่พบความเสี่ยง",
    "ความเสี่ยงต่ำ",
    "ความเสี่ยงปานกลาง",
    "ความเสี่ยงสูง",
    "ความเสี่ยงสูง",
]

URGENCY_LEVELS = [
    "ไม่เร่งด่วน",
    "ควรตรวจติดตาม",
    "ควรพบแพทย์เร็ว",
    "เร่งด่วนมาก",
    "เร่งด่วนมาก",
]

FINDINGS_MAP = {
    0: ["จอประสาทตามีลักษณะปกติ", "ไม่พบจุดเลือดออกหรือสิ่งผิดปกติ", "หลอดเลือดมีขนาดและรูปร่างปกติ"],
    1: ["พบ Microaneurysms เล็กน้อย", "ยังไม่พบการรั่วซึมของสารน้ำ", "จอประสาทตาส่วนใหญ่ยังปกติ"],
    2: ["พบ Microaneurysms และ Hemorrhages", "อาจพบ Hard Exudates หรือ Cotton Wool Spots", "มีการเปลี่ยนแปลงของหลอดเลือดบางส่วน"],
    3: ["พบเลือดออกในจอประสาทตาหลายจุด", "พบ Venous Beading หรือ IRMA", "มีความเสี่ยงสูงที่จะเกิด Proliferative DR"],
    4: ["พบการสร้างหลอดเลือดใหม่ผิดปกติ (Neovascularization)", "อาจพบ Vitreous Hemorrhage หรือ Fibrous Proliferation", "มีความเสี่ยงสูงต่อการสูญเสียการมองเห็น"],
}

RECOMMENDATIONS_MAP = {
    0: ["ตรวจคัดกรองตาปีละ 1 ครั้ง", "ควบคุมระดับน้ำตาลในเลือดให้อยู่ในเกณฑ์ปกติ", "รักษาความดันโลหิตและไขมันในเลือดให้เหมาะสม"],
    1: ["นัดติดตามผลทุก 6-12 เดือน", "ควบคุมระดับน้ำตาลในเลือดอย่างเคร่งครัด", "งดสูบบุหรี่และลดการดื่มแอลกอฮอล์"],
    2: ["พบจักษุแพทย์ภายใน 3-6 เดือน", "ควบคุมระดับน้ำตาล HbA1c ให้ต่ำกว่า 7%", "อาจพิจารณาการรักษาด้วย Laser หากจำเป็น"],
    3: ["พบจักษุแพทย์ภายใน 1 เดือน", "อาจต้องการการรักษาด้วย Pan-retinal Photocoagulation", "ควบคุมปัจจัยเสี่ยงทุกด้านอย่างเร่งด่วน"],
    4: ["พบจักษุแพทย์โดยเร็วที่สุด", "อาจต้องการการรักษาด้วย Vitrectomy หรือ Anti-VEGF", "ห้ามชะลอการรักษา เสี่ยงต่อการสูญเสียการมองเห็นถาวร"],
}

DESCRIPTIONS_MAP = {
    0: "ผลการตรวจไม่พบสัญญาณของโรคเบาหวานขึ้นตา จอประสาทตามีลักษณะปกติ แนะนำให้ตรวจคัดกรองต่อเนื่องปีละครั้ง",
    1: "พบสัญญาณเริ่มต้นของโรคเบาหวานขึ้นตาระดับเล็กน้อย (Mild NPDR) ควรติดตามอาการและควบคุมระดับน้ำตาลในเลือดอย่างสม่ำเสมอ",
    2: "พบการเปลี่ยนแปลงของจอประสาทตาในระดับปานกลาง (Moderate NPDR) ควรพบจักษุแพทย์เพื่อประเมินและวางแผนการรักษา",
    3: "พบการเปลี่ยนแปลงรุนแรงของจอประสาทตา (Severe NPDR) มีความเสี่ยงสูงที่จะพัฒนาเป็น Proliferative DR ต้องการการดูแลโดยจักษุแพทย์โดยเร็ว",
    4: "พบ Proliferative DR ซึ่งเป็นระยะรุนแรงของโรคเบาหวานขึ้นตา มีความเสี่ยงสูงต่อการสูญเสียการมองเห็น ต้องการการรักษาเร่งด่วน",
}

# ─── โหลด ONNX model ───────────────────────────
print(f"Loading model from: {MODEL_PATH}")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("✅ Model loaded")


# ─── HELPER ─────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    return arr[np.newaxis, ...]            # add batch dim


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ─── ROUTES ─────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "กรุณาอัปโหลดภาพถ่ายจอประสาทตา"}), 400

    file = request.files["image"]
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        return jsonify({"error": "รองรับเฉพาะไฟล์ภาพ JPEG, PNG และ WebP"}), 400

    image_bytes = file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "ขนาดไฟล์ต้องไม่เกิน 10 MB"}), 400

    try:
        tensor = preprocess(image_bytes)
        logits = sess.run([output_name], {input_name: tensor})[0][0]
        probs  = softmax(logits)
        grade_idx   = int(probs.argmax())
        confidence  = float(probs[grade_idx]) * 100

        result = {
            "riskLevel":       RISK_LEVELS[grade_idx],
            "confidence":      round(confidence, 1),
            "grade":           GRADE_NAMES[grade_idx],
            "findings":        FINDINGS_MAP[grade_idx],
            "description":     DESCRIPTIONS_MAP[grade_idx],
            "recommendations": RECOMMENDATIONS_MAP[grade_idx],
            "urgency":         URGENCY_LEVELS[grade_idx],
        }
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
