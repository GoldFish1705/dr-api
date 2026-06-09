"""
DR Detection - Flask Microservice
==================================
รับรูปภาพ → ตรวจว่าเป็น fundus ไหม → ONNX inference → ส่งผลลัพธ์ JSON
เพิ่ม /gradcam endpoint สำหรับ Occlusion-based heatmap
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.ndimage import gaussian_filter

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────
MODEL_PATH      = os.environ.get("MODEL_PATH", "model_dr.onnx")
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "fundus_classifier.onnx")
IMG_SIZE        = 512
IMG_SIZE_CLS    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FUNDUS_THRESHOLD = 0.5

GRADE_NAMES    = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
RISK_LEVELS    = ["ไม่พบความเสี่ยง", "ความเสี่ยงต่ำ", "ความเสี่ยงปานกลาง", "ความเสี่ยงสูง", "ความเสี่ยงสูง"]
URGENCY_LEVELS = ["ไม่เร่งด่วน", "ควรตรวจติดตาม", "ควรพบแพทย์เร็ว", "เร่งด่วนมาก", "เร่งด่วนมาก"]

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

# ─── โหลด ONNX models ──────────────────────────
print(f"Loading DR model: {MODEL_PATH}")
sess_dr   = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
dr_input  = sess_dr.get_inputs()[0].name
dr_output = sess_dr.get_outputs()[0].name
print("✅ DR model loaded")

print(f"Loading fundus classifier: {CLASSIFIER_PATH}")
sess_cls   = ort.InferenceSession(CLASSIFIER_PATH, providers=["CPUExecutionProvider"])
cls_input  = sess_cls.get_inputs()[0].name
cls_output = sess_cls.get_outputs()[0].name
print("✅ Fundus classifier loaded")


# ─── HELPERS ────────────────────────────────────
def preprocess(image_bytes: bytes, size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def is_fundus(image_bytes: bytes) -> tuple[bool, float]:
    tensor = preprocess(image_bytes, IMG_SIZE_CLS)
    logit  = sess_cls.run([cls_output], {cls_input: tensor})[0][0][0]
    prob   = float(1 / (1 + np.exp(-logit)))
    return prob > FUNDUS_THRESHOLD, prob

def compute_occlusion_heatmap(image_bytes: bytes, target_class: int,
                               patch_size: int = 64, stride: int = 32) -> np.ndarray:
    """Occlusion sensitivity map — ปิดบังส่วนต่างๆ แล้วดูว่า confidence ลดลงแค่ไหน"""
    tensor = preprocess(image_bytes, IMG_SIZE)  # (1, 3, 512, 512)

    # baseline confidence
    base_logits = sess_dr.run([dr_output], {dr_input: tensor})[0][0]
    base_prob   = softmax(base_logits)[target_class]

    h, w   = IMG_SIZE, IMG_SIZE
    heatmap = np.zeros((h, w), dtype=np.float32)
    count   = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = tensor.copy()
            occluded[0, :, y:y+patch_size, x:x+patch_size] = 0.0  # ปิดด้วย black

            logits = sess_dr.run([dr_output], {dr_input: occluded})[0][0]
            prob   = softmax(logits)[target_class]

            importance = base_prob - prob  # ยิ่งตกมาก ยิ่งสำคัญ
            heatmap[y:y+patch_size, x:x+patch_size] += importance
            count[y:y+patch_size, x:x+patch_size]   += 1

    count = np.maximum(count, 1)
    heatmap /= count
    heatmap  = gaussian_filter(heatmap, sigma=patch_size // 4)

    # normalize 0-1
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-6:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    return heatmap

def heatmap_to_overlay(image_bytes: bytes, heatmap: np.ndarray) -> str:
    """แปลง heatmap เป็น base64 image ที่ทับบนรูปต้นฉบับ"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img, dtype=np.float32)

    # colormap jet: blue→green→yellow→red
    h = heatmap
    r = np.clip(1.5 - np.abs(h * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(h * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(h * 4 - 1), 0, 1)
    colormap = np.stack([r, g, b], axis=-1) * 255

    alpha   = 0.5
    overlay = (img_arr * (1 - alpha) + colormap * alpha).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
        fundus_ok, fundus_prob = is_fundus(image_bytes)
        if not fundus_ok:
            return jsonify({
                "error": "ไม่พบภาพถ่ายจอประสาทตา กรุณาอัปโหลดภาพถ่ายจากอุปกรณ์ตรวจจอประสาทตาเท่านั้น",
                "fundus_confidence": round(fundus_prob * 100, 1),
            }), 400

        tensor     = preprocess(image_bytes, IMG_SIZE)
        logits     = sess_dr.run([dr_output], {dr_input: tensor})[0][0]
        probs      = softmax(logits)
        grade_idx  = int(probs.argmax())
        confidence = float(probs[grade_idx]) * 100

        result = {
            "riskLevel":        RISK_LEVELS[grade_idx],
            "confidence":       round(confidence, 1),
            "grade":            GRADE_NAMES[grade_idx],
            "findings":         FINDINGS_MAP[grade_idx],
            "description":      DESCRIPTIONS_MAP[grade_idx],
            "recommendations":  RECOMMENDATIONS_MAP[grade_idx],
            "urgency":          URGENCY_LEVELS[grade_idx],
            "fundusConfidence": round(fundus_prob * 100, 1),
            "gradeIdx":         grade_idx,
        }
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"}), 500


@app.route("/gradcam", methods=["POST"])
def gradcam():
    """คำนวณ occlusion heatmap แล้วส่งกลับเป็น base64 image"""
    if "image" not in request.files:
        return jsonify({"error": "กรุณาอัปโหลดภาพ"}), 400

    file        = request.files["image"]
    image_bytes = file.read()
    grade_idx   = int(request.form.get("gradeIdx", 0))

    try:
        heatmap    = compute_occlusion_heatmap(image_bytes, grade_idx)
        overlay_b64 = heatmap_to_overlay(image_bytes, heatmap)
        return jsonify({"success": True, "heatmap": overlay_b64})
    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาด: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)