this is still w.i.p i want to add more features

## Features
- OCR using:
  - Tesseract
  - EasyOCR
  - Combined OCR mode
- Translation engines:
  - Google Translate (Default)
  - Microsoft Translator
  - DeepL
  - MarianMT (offline)
  - OpenAI GPT
  - Google Gemini
  - Anthropic Claude
- Speech bubble detection and auto-fill
- Text removal (inpaint / blur / fill)
- Font selection, outlines, background fill
- Vertical (webtoon-style) image export
- Batch processing

---

## Requirements

- Python **3.9+**
- Tesseract OCR (system install)
- See `requirements.txt` for Python dependencies

---

## Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
````

### 2. Install Tesseract OCR

**Windows**

* Install from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* Default path used by the app:

  ```
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
---

## API Keys (Optional)

The app supports multiple translation APIs.

Supported:

* OpenAI
* Google Gemini
* Anthropic Claude
* DeepL
* Microsoft Translator

Keys are stored locally in:

```
scanlation_settings.json
```


