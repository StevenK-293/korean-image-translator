import os
import re
import cv2
import torch
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from transformers import MarianMTModel, MarianTokenizer
from sklearn.cluster import DBSCAN

IMAGE_ROOT = "korean_images"
OUTPUT_ROOT = "translated_images"
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp")
FONT_PATH = "arial.ttf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_CONF = 0.45
PADDING = 12

# print("Loading OCR...")
reader = easyocr.Reader(["ko"], gpu=DEVICE == "cuda")

# print("Loading translation model...")
MODEL_NAME = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def clean_korean(text):
    return re.sub(r"[^가-힣0-9\s\?\!\.\,]", "", text).strip()

def translate_batch(texts):
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        outputs = model.generate(**inputs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def auto_font(draw, text, box_w, box_h):
    low, high = 8, 64
    best = low

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(FONT_PATH, mid)
        bbox = draw.multiline_textbbox((0, 0), text, font=font)

        if bbox[2] <= box_w and bbox[3] <= box_h:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return ImageFont.truetype(FONT_PATH, best)

def merge_boxes(results):
    centers, texts, boxes = [], [], []

    for bbox, text, conf in results:
        if conf < MIN_CONF:
            continue

        text = clean_korean(text)
        if not text:
            continue

        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])

        centers.append([cx, cy])
        texts.append(text)
        boxes.append(bbox)

    if not centers:
        return []

    labels = DBSCAN(eps=60, min_samples=1).fit(centers).labels_

    merged = {}
    for i, label in enumerate(labels):
        merged.setdefault(label, []).append((texts[i], boxes[i]))

    bubbles = []
    for group in merged.values():
        full_text = " ".join(t for t, _ in group)
        all_pts = [pt for _, b in group for pt in b]
        bubbles.append((full_text, all_pts))

    return bubbles


def process_image(path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ocr = reader.readtext(rgb, detail=1)
    bubbles = merge_boxes(ocr)

    texts = [b[0] for b in bubbles]
    translations = translate_batch(texts) if texts else []

    for (ko_text, pts), en_text in zip(bubbles, translations):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x1, x2 = int(min(xs)) - PADDING, int(max(xs)) + PADDING
        y1, y2 = int(min(ys)) - PADDING, int(max(ys)) + PADDING

        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        #  erase original text
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

        # drraw english text
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        box_w = x2 - x1
        box_h = y2 - y1

        font = auto_font(draw, en_text, box_w, box_h)

        tb = draw.multiline_textbbox((0, 0), en_text, font=font)
        tx = x1 + (box_w - tb[2]) // 2
        ty = y1 + (box_h - tb[3]) // 2

        draw.multiline_text(
            (tx, ty),
            en_text,
            fill=(0, 0, 0),
            font=font,
            align="center"
        )

        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    files = [
        f for f in sorted(os.listdir(IMAGE_ROOT))
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    if not files:
        print("no images found in korean_images folder")
        return

    print(f"\nFound {len(files)} images\n")

    for file in files:
        src = os.path.join(IMAGE_ROOT, file)
        dst = os.path.join(
            OUTPUT_ROOT,
            os.path.splitext(file)[0] + "_en.jpg"
        )

        print(f"translating: {file}")
        try:
            result = process_image(src)
            result.save(dst)
            print(f"saved → {dst}\n")
        except Exception as e:
            print(f"Failed!!! {file}: {e}\n")

if __name__ == "__main__":
    main()
