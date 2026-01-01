import os
import re
import cv2
import torch
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox, colorchooser, simpledialog
import threading
import queue
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageFilter, ImageEnhance, ImageOps
import easyocr
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator, MicrosoftTranslator, DeeplTranslator
import glob
import matplotlib.font_manager
from difflib import SequenceMatcher
import openai
from openai import OpenAI
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from anthropic import Anthropic
import requests
from io import BytesIO
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SettingsManager:
    def __init__(self):
        self.settings_file = "translation_settings.json"
        self.default_settings = {
            "api_keys": {
                "openai": "",
                "gemini": "",
                "claude": "",
                "deepl": "",
                "microsoft": ""
            },
            "ocr": {
                "method": "combined",
                "preprocess": "adaptive",
                "confidence": 0.3,
                "remove_original": True
            },
            "translation": {
                "method": "google",
                "target_lang": "en",
                "context_aware": True
            },
            "text": {
                "font": "Arial",
                "size": 24,
                "color": "#FFFFFF",
                "outline_color": "#000000",
                "outline_width": 2,
                "bg_fill": True,
                "bg_color": "#000000",
                "bg_opacity": 0.5,
                "alignment": "center",
                "line_spacing": 1.2,
                "bubble_detect": True,
                "bubble_fill": True,
                "bubble_color": "#FFFFFF",
                "bubble_opacity": 0.9
            },
            "image": {
                "auto_adjust": True,
                "contrast": 1.0,
                "brightness": 1.0,
                "sharpness": 1.0,
                "despeckle": True
            },
            "ui": {
                "theme": "dark",
                "font_size": 11,
                "show_grid": False,
                "vertical_mode": False,
                "auto_scroll": True
            }
        }
        self.load_settings()
    
    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    self.settings = self.merge_settings(loaded)
            except:
                self.settings = self.default_settings.copy()
        else:
            self.settings = self.default_settings.copy()
    
    def merge_settings(self, loaded):
        merged = self.default_settings.copy()
        for category in merged:
            if category in loaded:
                merged[category].update(loaded[category])
        return merged
    
    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)

class TranslationManager:
    def __init__(self, settings_manager):
        self.settings = settings_manager
        self.models = {}
        self.clients = {}
        self.loading_models()
    
    def loading_models(self):
        try:
            #  will add more models later
            self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
            self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ko-en").to(DEVICE)
            self.model.eval()
            self.models["marian"] = True
        except:
            self.models["marian"] = False
        
        api_keys = self.settings.settings["api_keys"]
        if api_keys["openai"]:
            try:
                self.clients["openai"] = OpenAI(api_key=api_keys["openai"])
                self.models["openai"] = True
            except:
                self.models["openai"] = False
        
        if api_keys["gemini"] and genai:
            try:
                genai.configure(api_key=api_keys["gemini"])
                self.models["gemini"] = genai.GenerativeModel('gemini-pro')
            except:
                self.models["gemini"] = False
        
        if api_keys["claude"]:
            try:
                self.clients["claude"] = Anthropic(api_key=api_keys["claude"])
                self.models["claude"] = True
            except:
                self.models["claude"] = False
    
    def translate(self, text, method="google"):
        if not text.strip():
            return ""
        
        try:
            if method == "google":
                return GoogleTranslator(source='ko', target='en').translate(text)
            elif method == "microsoft":
                api_key = self.settings.settings["api_keys"]["microsoft"]
                if api_key:
                    return MicrosoftTranslator(api_key=api_key, source='ko', target='en').translate(text)
            elif method == "deepl":
                api_key = self.settings.settings["api_keys"]["deepl"]
                if api_key:
                    return DeeplTranslator(api_key=api_key, source='ko', target='en').translate(text)
            elif method == "marian" and self.models.get("marian"):
                return self.marian_translate(text)
            elif method == "openai" and self.models.get("openai"):
                return self.openai_translate(text)
            elif method == "gemini" and self.models.get("gemini"):
                return self.gemini_translate(text)
            elif method == "claude" and self.models.get("claude"):
                return self.claude_translate(text)
        except Exception as e:
            print(f"Translation error ({method}): {e}")
        
        return GoogleTranslator(source='ko', target='en').translate(text)
    
    def marian_translate(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def openai_translate(self, text):
        response = self.clients["openai"].chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Translate Korean to English accurately."},
                     {"role": "user", "content": text}],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def gemini_translate(self, text):
        response = self.models["gemini"].generate_content(f"Translate from Korean to English: {text}")
        return response.text
    
    def claude_translate(self, text):
        response = self.clients["claude"].messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": f"Translate this Korean text to English: {text}"}]
        )
        return response.content[0].text

class OCRManager:
    def __init__(self, settings_manager):
        self.settings = settings_manager
        self.readers = {}
        self.initialize_readers()
    
    def initialize_readers(self):
        try:
            self.readers["easyocr"] = easyocr.Reader(["ko"], gpu=DEVICE == "cuda")
        except:
            self.readers["easyocr"] = None
    
    def preprocess_image(self, image, method="adaptive"):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'adaptive':
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return result
        elif method == 'enhanced':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(gray)
        elif method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
            return edges
        else:
            return gray
    
    def extract_text(self, image, method="combined", preprocess="adaptive", confidence=0.3):
        if method == "tesseract":
            return self.tesseract_ocr(image, preprocess)
        elif method == "easyocr" and self.readers.get("easyocr"):
            return self.easyocr_ocr(image, preprocess, confidence)
        elif method == "combined":
            return self.combined_ocr(image, preprocess, confidence)
    
    def tesseract_ocr(self, image, preprocess):
        processed = self.preprocess_image(image, preprocess)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(processed)
        config = '--oem 3 --psm 6 -l kor+eng'
        text = pytesseract.image_to_string(pil_image, config=config)
        return self.clean_text(text)
    
    def easyocr_ocr(self, image, preprocess, confidence):
        processed = self.preprocess_image(image, preprocess)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        results = self.readers["easyocr"].readtext(processed, detail=0, paragraph=True)
        filtered = [r for r in results if len(r.strip()) > 0]
        text = " ".join(filtered)
        return self.clean_text(text)
    
    def combined_ocr(self, image, preprocess, confidence):
        tesseract_text = self.tesseract_ocr(image, preprocess)
        easyocr_text = ""
        if self.readers.get("easyocr"):
            easyocr_text = self.easyocr_ocr(image, preprocess, confidence)
        
        if len(tesseract_text) > len(easyocr_text) * 1.5:
            return tesseract_text
        elif len(easyocr_text) > len(tesseract_text) * 1.5:
            return easyocr_text
        
        combined = f"{tesseract_text}\n{easyocr_text}"
        return self.clean_text(combined)
    
    def clean_text(self, text):
        text = re.sub(r'[^\uAC00-\uD7A30-9a-zA-Z\s\.,!?\-\'\"\(\)\[\]:;]', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class SpeechBubbleDetector:
    @staticmethod
    def detect_bubbles(image, method="contour"):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        bubbles = []
        
        if method == "contour":
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3,3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:
                                bubbles.append((x, y, x+w, y+h))
        
        elif method == "color":
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    bubbles.append((x, y, x+w, y+h))
        
        return bubbles
    
    @staticmethod
    def fill_bubble(image, bbox, color="#FFFFFF", opacity=0.9):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
        
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        a = int(opacity * 255)
        
        overlay = Image.new('RGBA', image.size, (r, g, b, a))
        mask_img = Image.fromarray(mask, 'L')
        result = Image.composite(overlay, image.convert('RGBA'), mask_img)
        
        return result.convert('RGB')

class TextRemover:
    @staticmethod
    def remove_text(image, bbox, method="inpaint"):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        width, height = image.size
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width-1))
        y1 = max(0, min(y1, height-1))
        y2 = max(0, min(y2, height-1))
        
        if x1 >= x2 or y1 >= y2:
            return image
        
        img_array = np.array(image)
        
        if method == "inpaint":
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            result = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(result)
        
        elif method == "blur":
            cropped = image.crop((x1, y1, x2, y2))
            blurred = cropped.filter(ImageFilter.GaussianBlur(radius=15))
            image.paste(blurred, (x1, y1))
            return image
        
        elif method == "fill":
            draw = ImageDraw.Draw(image)
            avg_color = TextRemover.get_average_color(image, (x1, y1, x2, y2))
            draw.rectangle([x1, y1, x2, y2], fill=avg_color)
            return image
        
        return image
    
    @staticmethod
    def get_average_color(image, bbox):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        np_img = np.array(cropped)
        avg_color = np.mean(np_img, axis=(0, 1)).astype(int)
        return tuple(avg_color)

class ImageEnhancer:
    @staticmethod
    def enhance(image, settings):
        enhanced = image.copy()
        
        if settings["auto_adjust"]:
            enhanced = ImageEnhancer.auto_contrast(enhanced)
        
        if settings["contrast"] != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(settings["contrast"])
        
        if settings["brightness"] != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(settings["brightness"])
        
        if settings["sharpness"] != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(settings["sharpness"])
        
        if settings["despeckle"]:
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        return enhanced
    
    @staticmethod
    def auto_contrast(image):
        return ImageOps.autocontrast(image, cutoff=2)

class TranslationsToolkit:
    def __init__(self):
        self.redraw_history = []
    
    @staticmethod
    def create_vertical_strip(images, spacing=20):
        if not images:
            return None
        
        widths = [img.width for img in images]
        max_width = max(widths)
        total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
        
        result = Image.new('RGB', (max_width, total_height), color='white')
        
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2
            result.paste(img, (x_offset, y_offset))
            y_offset += img.height + spacing
        
        return result
    
    @staticmethod
    def apply_manga_effects(image):
        result = image.copy()
        
        result = ImageEnhance.Contrast(result).enhance(1.2)
        result = ImageEnhance.Sharpness(result).enhance(1.1)
        
        return result
    
    @staticmethod
    def clean_scan(image):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        result = cv2.bitwise_not(cleaned)
        return Image.fromarray(result)

class ModernTkinterTheme:
    @staticmethod
    def setup_theme(master):
        bg_color = "#0f172a"
        sidebar_bg = "#1e293b"
        panel_bg = "#334155"
        button_bg = "#475569"
        button_hover = "#64748b"
        text_color = "#f8fafc"
        accent_color = "#3b82f6"
        success_color = "#10b981"
        warning_color = "#f59e0b"
        error_color = "#ef4444"
        
        master.configure(bg=bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=text_color, font=('Segoe UI', 10))
        style.configure('TButton', background=button_bg, foreground=text_color, borderwidth=0, 
                       relief='flat', font=('Segoe UI', 10))
        style.map('TButton', background=[('active', button_hover)])
        style.configure('TCheckbutton', background=bg_color, foreground=text_color)
        style.configure('TRadiobutton', background=bg_color, foreground=text_color)
        style.configure('TEntry', fieldbackground=panel_bg, foreground=text_color, borderwidth=1)
        style.configure('TCombobox', fieldbackground=panel_bg, foreground=text_color)
        style.configure('TNotebook', background=bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', background=sidebar_bg, foreground=text_color, padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', accent_color)])
        style.configure('Horizontal.TScale', background=bg_color)
        
        return {
            'bg_color': bg_color,
            'sidebar_bg': sidebar_bg,
            'panel_bg': panel_bg,
            'button_bg': button_bg,
            'button_hover': button_hover,
            'text_color': text_color,
            'accent_color': accent_color,
            'success_color': success_color,
            'warning_color': warning_color,
            'error_color': error_color
        }

class Translations_ToolKit:
    def __init__(self, master):
        self.master = master
        self.master.title("Translation Toolkit")
        self.master.geometry("1800x1000")
        
        self.colors = ModernTkinterTheme.setup_theme(master)
        self.settings = SettingsManager()
        self.translator = TranslationManager(self.settings)
        self.ocr = OCRManager(self.settings)
        self.text_remover = TextRemover()
        self.bubble_detector = SpeechBubbleDetector()
        self.image_enhancer = ImageEnhancer()
        self.toolkit = TranslationsToolkit()
        
        self.setup_variables()
        self.create_gui()
        self.load_system_fonts()
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_variables(self):
        self.images = []
        self.image_paths = []
        self.current_index = 0
        self.original_image = None
        self.edited_image = None
        self.bbox = None
        self.selection_rect = None
        self.is_locked = False
        self.scale_factor = 1.0
        self.fonts = []
        self.current_font = self.settings.settings["text"]["font"]
        self.font_size = self.settings.settings["text"]["size"]
        self.font_color = self.settings.settings["text"]["color"]
        self.outline_color = self.settings.settings["text"]["outline_color"]
        self.outline_width = self.settings.settings["text"]["outline_width"]
        self.bg_fill = self.settings.settings["text"]["bg_fill"]
        self.bg_color = self.settings.settings["text"]["bg_color"]
        self.bg_opacity = self.settings.settings["text"]["bg_opacity"]
        self.text_align = self.settings.settings["text"]["alignment"]
        self.line_spacing = self.settings.settings["text"]["line_spacing"]
        self.bubble_detect = self.settings.settings["text"]["bubble_detect"]
        self.bubble_fill = self.settings.settings["text"]["bubble_fill"]
        self.bubble_color = self.settings.settings["text"]["bubble_color"]
        self.bubble_opacity = self.settings.settings["text"]["bubble_opacity"]
        self.work_history = []
        self.text_cache = {}
        self.detected_bubbles = []
        self.current_bubble_index = 0
        self.vertical_mode = self.settings.settings["ui"]["vertical_mode"]
        self.vertical_image = None
        self.vertical_offset = 0
        self.auto_scroll = self.settings.settings["ui"]["auto_scroll"]
    
    def create_gui(self):
        self.create_menu()
        self.create_main_panels()
        self.create_status_bar()
    
    def create_menu(self):
        menubar = tk.Menu(self.master, bg=self.colors['sidebar_bg'], fg=self.colors['text_color'])
        self.master.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['sidebar_bg'], fg=self.colors['text_color'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.open_folder, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Image(s)", command=self.open_images)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current", command=self.save_current_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save & Next", command=self.save_and_next)
        file_menu.add_command(label="Save All", command=self.batch_save_all)
        file_menu.add_separator()
        file_menu.add_command(label="Export Webtoon", command=self.export_webtoon)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        edit_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['sidebar_bg'], fg=self.colors['text_color'])
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo_action, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo_action, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Enhance Image", command=self.enhance_current_image)
        edit_menu.add_command(label="Clean Scan", command=self.clean_current_image)
        edit_menu.add_separator()
        edit_menu.add_command(label="Toggle Vertical Mode", command=self.toggle_vertical_mode, accelerator="Ctrl+V")
        
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['sidebar_bg'], fg=self.colors['text_color'])
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Detect Bubbles", command=self.detect_bubbles)
        tools_menu.add_command(label="Fill All Bubbles", command=self.fill_all_bubbles)
        tools_menu.add_command(label="Auto Translate All", command=self.auto_translate_page)
        tools_menu.add_separator()
        tools_menu.add_command(label="Apply Manga Effects", command=self.apply_manga_effects)
        
        self.master.bind('<Control-o>', lambda e: self.open_folder())
        self.master.bind('<Control-s>', lambda e: self.save_current_image())
        self.master.bind('<Control-z>', lambda e: self.undo_action())
        self.master.bind('<Control-y>', lambda e: self.redo_action())
        self.master.bind('<Control-v>', lambda e: self.toggle_vertical_mode())
    
    def create_main_panels(self):
        main_container = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_sidebar(main_container)
        self.create_center_panel(main_container)
        self.create_right_panel(main_container)
    
    def create_sidebar(self, parent):
        sidebar = ttk.Frame(parent, width=350)
        parent.add(sidebar, weight=1)
        
        notebook = ttk.Notebook(sidebar)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_image_tab(notebook)
        self.create_ocr_tab(notebook)
        self.create_translation_tab(notebook)
        self.create_text_tab(notebook)
        self.create_tools_tab(notebook)
        self.create_effects_tab(notebook)
    
    def create_image_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📁 Images")
        
        ttk.Button(tab, text="📂 Open Folder", command=self.open_folder).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="🖼️ Open Image(s)", command=self.open_images).pack(fill=tk.X, pady=5, padx=5)
        
        list_frame = ttk.Frame(tab)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox = tk.Listbox(list_frame, bg=self.colors['panel_bg'], fg=self.colors['text_color'], 
                                       selectbackground=self.colors['accent_color'], yscrollcommand=scrollbar.set,
                                       font=('Segoe UI', 10))
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        nav_frame = ttk.Frame(tab)
        nav_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="🗑️", command=self.remove_image, width=3).pack(side=tk.RIGHT)
        
        ttk.Label(tab, text="Quick Actions:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        ttk.Button(tab, text="🚀 Auto Process Page", command=self.auto_process_page).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="⚡ Batch Process All", command=self.batch_process_all).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="📏 Toggle Vertical Mode", command=self.toggle_vertical_mode).pack(fill=tk.X, pady=10, padx=5)
    
    def create_ocr_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="🔍 OCR")
        
        ttk.Label(tab, text="OCR Engine:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.ocr_method_var = tk.StringVar(value=self.settings.settings["ocr"]["method"])
        methods = [("Combined", "combined"), ("Tesseract", "tesseract"), ("EasyOCR", "easyocr")]
        for text, value in methods:
            ttk.Radiobutton(tab, text=text, variable=self.ocr_method_var, value=value).pack(anchor=tk.W, padx=20)
        
        ttk.Label(tab, text="Preprocessing:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        self.preprocess_var = tk.StringVar(value=self.settings.settings["ocr"]["preprocess"])
        preproc_options = [("Adaptive", "adaptive"), ("Otsu", "otsu"), ("Enhanced", "enhanced"), 
                          ("Canny", "canny"), ("None", "none")]
        for text, value in preproc_options:
            ttk.Radiobutton(tab, text=text, variable=self.preprocess_var, value=value).pack(anchor=tk.W, padx=20)
        
        ttk.Label(tab, text="Confidence Threshold:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        self.conf_scale = ttk.Scale(tab, from_=0.1, to=1.0, value=self.settings.settings["ocr"]["confidence"])
        self.conf_scale.pack(fill=tk.X, padx=5)
        
        self.remove_original_var = tk.BooleanVar(value=self.settings.settings["ocr"]["remove_original"])
        ttk.Checkbutton(tab, text="Remove Original Text", variable=self.remove_original_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Button(tab, text="🔍 Scan Selection", command=self.scan_selection).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="📄 Scan Full Page", command=self.scan_full_page).pack(fill=tk.X, pady=2, padx=5)
    
    def create_translation_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="🌐 Translation")
        
        ttk.Label(tab, text="Translation Service:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.trans_method_var = tk.StringVar(value=self.settings.settings["translation"]["method"])
        
        services = [
            ("Google Translate", "google"),
            ("Microsoft", "microsoft"),
            ("DeepL", "deepl"),
            ("MarianMT", "marian"),
            ("OpenAI GPT", "openai"),
            ("Google Gemini", "gemini"),
            ("Claude AI", "claude")
        ]
        
        for text, value in services:
            ttk.Radiobutton(tab, text=text, variable=self.trans_method_var, value=value).pack(anchor=tk.W, padx=20)
        
        self.context_aware_var = tk.BooleanVar(value=self.settings.settings["translation"]["context_aware"])
        ttk.Checkbutton(tab, text="Context-Aware Translation", variable=self.context_aware_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Button(tab, text="🌐 Translate Selection", command=self.translate_selection).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="🔄 Auto Translate All", command=self.auto_translate_page).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="⚙️ Configure API Keys", command=self.open_settings).pack(fill=tk.X, pady=10, padx=5)
    
    def create_text_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="✏️ Text & Bubbles")
        
        notebook_inner = ttk.Notebook(tab)
        notebook_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_text_settings_tab(notebook_inner)
        self.create_bubble_settings_tab(notebook_inner)
        self.create_apply_tab(notebook_inner)
    
    def create_text_settings_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Text Settings")
        
        ttk.Label(tab, text="Font:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.font_combo = ttk.Combobox(tab, state="readonly", font=('Segoe UI', 10))
        self.font_combo.pack(fill=tk.X, padx=5)
        self.font_combo.bind("<<ComboboxSelected>>", self.on_font_select)
        
        ttk.Label(tab, text="Size:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        size_frame = ttk.Frame(tab)
        size_frame.pack(fill=tk.X, padx=5)
        ttk.Button(size_frame, text="−", width=3, command=lambda: self.adjust_font_size(-1)).pack(side=tk.LEFT)
        self.size_label = ttk.Label(size_frame, text=str(self.font_size), width=6)
        self.size_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(size_frame, text="+", width=3, command=lambda: self.adjust_font_size(1)).pack(side=tk.LEFT)
        
        ttk.Label(tab, text="Colors:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        color_frame = ttk.Frame(tab)
        color_frame.pack(fill=tk.X, padx=5)
        ttk.Button(color_frame, text="Text", command=self.choose_font_color, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="Outline", command=self.choose_outline_color, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="Background", command=self.choose_bg_color, width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(tab, text="Outline Width:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.outline_scale = ttk.Scale(tab, from_=0, to=5, value=self.outline_width)
        self.outline_scale.pack(fill=tk.X, padx=5)
        self.outline_scale.bind("<ButtonRelease-1>", lambda e: self.update_outline_width())
        
        ttk.Label(tab, text="Alignment:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.align_var = tk.StringVar(value=self.text_align)
        align_frame = ttk.Frame(tab)
        align_frame.pack(fill=tk.X, padx=5)
        aligns = [("Left", "left"), ("Center", "center"), ("Right", "right")]
        for text, value in aligns:
            ttk.Radiobutton(align_frame, text=text, variable=self.align_var, value=value).pack(side=tk.LEFT)
        
        self.bg_fill_var = tk.BooleanVar(value=self.bg_fill)
        ttk.Checkbutton(tab, text="Show Background Fill", variable=self.bg_fill_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Label(tab, text="Background Opacity:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.bg_opacity_scale = ttk.Scale(tab, from_=0, to=1, value=self.bg_opacity)
        self.bg_opacity_scale.pack(fill=tk.X, padx=5)
        self.bg_opacity_scale.bind("<ButtonRelease-1>", lambda e: self.update_bg_opacity())
    
    def create_bubble_settings_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Bubble Settings")
        
        ttk.Label(tab, text="Bubble Detection:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.bubble_method_var = tk.StringVar(value="contour")
        methods = [("Contour", "contour"), ("Color", "color")]
        for text, value in methods:
            ttk.Radiobutton(tab, text=text, variable=self.bubble_method_var, value=value).pack(anchor=tk.W, padx=20)
        
        ttk.Label(tab, text="Bubble Color:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        color_frame = ttk.Frame(tab)
        color_frame.pack(fill=tk.X, padx=5)
        self.bubble_color_btn = ttk.Button(color_frame, text="Choose Color", command=self.choose_bubble_color, width=12)
        self.bubble_color_btn.pack(side=tk.LEFT)
        self.bubble_color_label = ttk.Label(color_frame, text="█", foreground=self.bubble_color)
        self.bubble_color_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(tab, text="Bubble Opacity:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        self.bubble_opacity_scale = ttk.Scale(tab, from_=0, to=1, value=self.bubble_opacity)
        self.bubble_opacity_scale.pack(fill=tk.X, padx=5)
        
        self.bubble_detect_var = tk.BooleanVar(value=self.bubble_detect)
        ttk.Checkbutton(tab, text="Auto-detect Bubbles", variable=self.bubble_detect_var).pack(anchor=tk.W, pady=5, padx=5)
        
        self.bubble_fill_var = tk.BooleanVar(value=self.bubble_fill)
        ttk.Checkbutton(tab, text="Auto-fill Bubbles", variable=self.bubble_fill_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Button(tab, text="🔍 Detect Bubbles", command=self.detect_bubbles).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="🖌️ Fill All Bubbles", command=self.fill_all_bubbles).pack(fill=tk.X, pady=2, padx=5)
    
    def create_apply_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Apply")
        
        ttk.Label(tab, text="Apply Translation:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        
        ttk.Button(tab, text="🔄 Apply to Selection", command=self.apply_edited_text).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="📝 Apply & Next Bubble", command=self.apply_and_next_bubble).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="💾 Apply & Save Image", command=self.apply_and_save).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="🚀 Apply to All Bubbles", command=self.apply_to_all_bubbles).pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(tab, text="Navigation:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        nav_frame = ttk.Frame(tab)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(nav_frame, text="◀ Prev Bubble", command=self.prev_bubble).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next Bubble ▶", command=self.next_bubble).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(tab, text="Current Bubble:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        self.current_bubble_label = ttk.Label(tab, text="0/0", font=('Segoe UI', 10, 'bold'))
        self.current_bubble_label.pack(anchor=tk.W, padx=5)
    
    def create_tools_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="🛠️ Tools")
        
        ttk.Label(tab, text="Translation Tools:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        
        ttk.Button(tab, text="🧹 Clean Scan", command=self.clean_current_image).pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(tab, text="🎨 Manga Effects", command=self.apply_manga_effects).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="📏 Create Vertical Strip", command=self.create_vertical_strip).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="🔄 Auto Process Page", command=self.auto_process_page).pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(tab, text="Batch Processing:").pack(anchor=tk.W, pady=(10, 0), padx=5)
        ttk.Button(tab, text="⚡ Process All Pages", command=self.batch_process_all).pack(fill=tk.X, pady=2, padx=5)
        ttk.Button(tab, text="📤 Export Webtoon", command=self.export_webtoon).pack(fill=tk.X, pady=2, padx=5)
    
    def create_effects_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="✨ Effects")
        
        self.auto_adjust_var = tk.BooleanVar(value=self.settings.settings["image"]["auto_adjust"])
        ttk.Checkbutton(tab, text="Auto-Adjust Image", variable=self.auto_adjust_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Label(tab, text="Contrast:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.contrast_scale = ttk.Scale(tab, from_=0.5, to=2.0, value=self.settings.settings["image"]["contrast"])
        self.contrast_scale.pack(fill=tk.X, padx=5)
        
        ttk.Label(tab, text="Brightness:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.brightness_scale = ttk.Scale(tab, from_=0.5, to=2.0, value=self.settings.settings["image"]["brightness"])
        self.brightness_scale.pack(fill=tk.X, padx=5)
        
        ttk.Label(tab, text="Sharpness:").pack(anchor=tk.W, pady=(5, 0), padx=5)
        self.sharpness_scale = ttk.Scale(tab, from_=0.5, to=2.0, value=self.settings.settings["image"]["sharpness"])
        self.sharpness_scale.pack(fill=tk.X, padx=5)
        
        self.despeckle_var = tk.BooleanVar(value=self.settings.settings["image"]["despeckle"])
        ttk.Checkbutton(tab, text="Remove Noise", variable=self.despeckle_var).pack(anchor=tk.W, pady=5, padx=5)
        
        ttk.Button(tab, text="Apply Effects", command=self.apply_image_effects).pack(fill=tk.X, pady=10, padx=5)
        ttk.Button(tab, text="Reset Effects", command=self.reset_image_effects).pack(fill=tk.X, pady=2, padx=5)
    
    def create_center_panel(self, parent):
        center_frame = ttk.Frame(parent)
        parent.add(center_frame, weight=3)
        
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg=self.colors['panel_bg'], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        bottom_bar = ttk.Frame(center_frame)
        bottom_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.coord_label = ttk.Label(bottom_bar, text="X: 0, Y: 0, Scale: 100%")
        self.coord_label.pack(side=tk.LEFT)
        
        ttk.Button(bottom_bar, text="↺ Reset View", command=self.reset_view).pack(side=tk.RIGHT, padx=2)
    
    def create_right_panel(self, parent):
        right_frame = ttk.Frame(parent, width=400)
        parent.add(right_frame, weight=1)
        
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_preview_tab(notebook)
        self.create_text_edit_tab(notebook)
        self.create_history_tab(notebook)
        self.create_bubble_list_tab(notebook)
    
    def create_preview_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="👁️ Preview")
        
        ttk.Label(tab, text="Original Text (Korean):").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.ocr_preview = scrolledtext.ScrolledText(tab, height=6, bg=self.colors['panel_bg'], 
                                                    fg=self.colors['text_color'], wrap=tk.WORD,
                                                    font=('Segoe UI', 10))
        self.ocr_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(tab, text="Translated Text (English):").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.trans_preview = scrolledtext.ScrolledText(tab, height=6, bg=self.colors['panel_bg'], 
                                                      fg=self.colors['success_color'], wrap=tk.WORD,
                                                      font=('Segoe UI', 10))
        self.trans_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_frame = ttk.Frame(tab)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(info_frame, text="Text Length:").pack(side=tk.LEFT)
        self.text_length_label = ttk.Label(info_frame, text="0 chars")
        self.text_length_label.pack(side=tk.LEFT, padx=10)
    
    def create_text_edit_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📝 Editor")
        
        ttk.Label(tab, text="Edit Translation Text:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.text_editor = scrolledtext.ScrolledText(tab, height=12, bg=self.colors['panel_bg'], 
                                                    fg=self.colors['text_color'], wrap=tk.WORD,
                                                    font=('Segoe UI', 10))
        self.text_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="💾 Apply to Image", command=self.apply_edited_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_editor).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📋 Copy", command=self.copy_to_clipboard).pack(side=tk.RIGHT, padx=2)
    
    def create_history_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📜 History")
        
        self.history_listbox = tk.Listbox(tab, bg=self.colors['panel_bg'], fg=self.colors['text_color'],
                                         selectbackground=self.colors['accent_color'], height=10,
                                         font=('Segoe UI', 9))
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        history_buttons = ttk.Frame(tab)
        history_buttons.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(history_buttons, text="Restore", command=self.restore_from_history).pack(side=tk.LEFT)
        ttk.Button(history_buttons, text="Clear", command=self.clear_history).pack(side=tk.RIGHT)
    
    def create_bubble_list_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="💭 Bubbles")
        
        ttk.Label(tab, text="Detected Speech Bubbles:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.bubble_listbox = tk.Listbox(tab, bg=self.colors['panel_bg'], fg=self.colors['text_color'],
                                        selectbackground=self.colors['accent_color'], height=10,
                                        font=('Segoe UI', 9))
        self.bubble_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        bubble_buttons = ttk.Frame(tab)
        bubble_buttons.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bubble_buttons, text="Select", command=self.select_bubble).pack(side=tk.LEFT)
        ttk.Button(bubble_buttons, text="Fill", command=self.fill_selected_bubble).pack(side=tk.LEFT)
        ttk.Button(bubble_buttons, text="Clear List", command=self.clear_bubble_list).pack(side=tk.RIGHT)
    
    def create_status_bar(self):
        self.status_frame = ttk.Frame(self.master)
        self.status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate', length=150)
        self.progress.pack(side=tk.RIGHT)
    
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if folder:
            self.load_folder_images(folder)
    
    def load_folder_images(self, folder):
        self.images.clear()
        self.image_paths.clear()
        self.image_listbox.delete(0, tk.END)
        
        for ext in SUPPORTED_EXT:
            pattern = os.path.join(folder, f"*{ext}")
            files = glob.glob(pattern)
            self.image_paths.extend(sorted(files))
        
        if self.image_paths:
            for path in self.image_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    self.images.append(img)
                    self.image_listbox.insert(tk.END, os.path.basename(path))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            self.current_index = 0
            self.image_listbox.selection_set(0)
            self.display_current_image()
            self.update_status(f"Loaded {len(self.images)} images")
    
    def open_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )
        
        if files:
            for path in files:
                if path not in self.image_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        self.images.append(img)
                        self.image_paths.append(path)
                        self.image_listbox.insert(tk.END, os.path.basename(path))
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
            
            if self.images:
                self.current_index = len(self.images) - 1
                self.image_listbox.selection_clear(0, tk.END)
                self.image_listbox.selection_set(self.current_index)
                self.display_current_image()
    
    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.display_current_image()
    
    def display_current_image(self):
        if not self.images or self.current_index >= len(self.images):
            return
        
        self.original_image = self.images[self.current_index].copy()
        self.edited_image = self.original_image.copy()
        self.work_history = [self.edited_image.copy()]
        
        if self.vertical_mode and len(self.images) > 1:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_history_list()
        
        if self.bubble_detect_var.get():
            self.detect_bubbles()
    
    def update_canvas(self):
        if not self.edited_image:
            return
        
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10:
            canvas_width = 800
            canvas_height = 600
        
        img_width, img_height = self.edited_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 2.0)
        self.scale_factor = scale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        display_img = self.edited_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        scale_percent = int(scale * 100)
        self.coord_label.config(text=f"Scale: {scale_percent}% | Size: {img_width}x{img_height}")
        
        if self.vertical_mode:
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def update_vertical_display(self):
        if not self.images:
            return
        
        self.vertical_image = self.toolkit.create_vertical_strip(self.images)
        if not self.vertical_image:
            return
        
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        
        if canvas_width < 10:
            canvas_width = 800
        
        img_width, img_height = self.vertical_image.size
        scale = min(canvas_width / img_width, 2.0)
        self.scale_factor = scale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        display_img = self.vertical_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        self.coord_label.config(text=f"Vertical Mode | Images: {len(self.images)} | Total Height: {img_height}px")
    
    def on_canvas_configure(self, event):
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
    
    def toggle_vertical_mode(self):
        self.vertical_mode = not self.vertical_mode
        self.settings.settings["ui"]["vertical_mode"] = self.vertical_mode
        
        if self.vertical_mode:
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.update_vertical_display()
            self.update_status("Vertical mode enabled")
        else:
            self.scrollbar.pack_forget()
            self.canvas.configure(yscrollcommand=None)
            self.update_canvas()
            self.update_status("Vertical mode disabled")
    
    def create_vertical_strip(self):
        if len(self.images) < 2:
            messagebox.showinfo("Info", "Need at least 2 images to create vertical strip")
            return
        
        self.vertical_image = self.toolkit.create_vertical_strip(self.images)
        if self.vertical_image:
            self.vertical_mode = True
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.update_vertical_display()
            self.update_status(f"Created vertical strip with {len(self.images)} images")
    
    def export_webtoon(self):
        if not self.images:
            messagebox.showwarning("No Images", "No images to export")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        
        vertical_image = self.toolkit.create_vertical_strip(self.images)
        if vertical_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"webtoon_{timestamp}.png")
            vertical_image.save(output_path, quality=95)
            self.update_status(f"Webtoon exported to {output_path}")
            messagebox.showinfo("Success", f"Webtoon exported successfully!\n{output_path}")
    
    def on_mouse_down(self, event):
        if not self.edited_image:
            return
        
        self.selection_start = (event.x, event.y)
        if hasattr(self, 'selection_rect') and self.selection_rect:
            self.canvas.delete(self.selection_rect)
        
        self.selection_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=self.colors['accent_color'], width=2, dash=(5, 5)
        )
    
    def on_mouse_drag(self, event):
        if hasattr(self, 'selection_start') and self.selection_rect:
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            self.canvas.coords(self.selection_rect, x1, y1, x2, y2)
    
    def on_mouse_up(self, event):
        if hasattr(self, 'selection_start'):
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.bbox = self.canvas_to_image_coords(x1, y1, x2, y2)
                self.is_locked = True
                self.canvas.itemconfig(self.selection_rect, dash=())
                self.scan_selection()
            else:
                if self.selection_rect:
                    self.canvas.delete(self.selection_rect)
                self.selection_rect = None
                self.is_locked = False
    
    def on_mouse_wheel(self, event):
        if self.vertical_mode:
            self.canvas.yview_scroll(-1 * int(event.delta/120), "units")
        elif self.edited_image:
            scale_change = 0.1 if event.delta > 0 else -0.1
            self.scale_factor = max(0.1, min(3.0, self.scale_factor + scale_change))
            self.update_canvas()
    
    def canvas_to_image_coords(self, x1, y1, x2, y2):
        if self.vertical_mode and self.vertical_image:
            img_width, img_height = self.vertical_image.size
            canvas_width = self.canvas.winfo_width()
            
            scale = min(canvas_width / img_width, 2.0)
            x_offset = 0
            
            y1 = max(0, y1 / scale)
            y2 = min(img_height, y2 / scale)
            x1 = max(0, (x1 - x_offset) / scale)
            x2 = min(img_width, (x2 - x_offset) / scale)
        else:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_width, img_height = self.edited_image.size
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            x1 = max(0, (x1 - x_offset) / self.scale_factor)
            y1 = max(0, (y1 - y_offset) / self.scale_factor)
            x2 = min(img_width, (x2 - x_offset) / self.scale_factor)
            y2 = min(img_height, (y2 - y_offset) / self.scale_factor)
        
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    def scan_selection(self):
        if not self.bbox:
            return
        
        x1, y1, x2, y2 = [int(coord) for coord in self.bbox]
        
        if self.vertical_mode and self.vertical_image:
            cropped = self.vertical_image.crop((x1, y1, x2, y2))
        else:
            cropped = self.original_image.crop((x1, y1, x2, y2))
        
        cv_image = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        
        method = self.preprocess_var.get()
        processed = self.ocr.preprocess_image(cv_image, method)
        
        ocr_method = self.ocr_method_var.get()
        confidence = self.conf_scale.get()
        ocr_text = self.ocr.extract_text(processed, ocr_method, method, confidence)
        
        self.ocr_preview.delete(1.0, tk.END)
        self.ocr_preview.insert(1.0, ocr_text)
        
        self.text_editor.delete(1.0, tk.END)
        self.text_editor.insert(1.0, ocr_text)
        
        self.text_length_label.config(text=f"{len(ocr_text)} chars")
        self.update_status(f"OCR completed: {len(ocr_text)} characters")
    
    def scan_full_page(self):
        if not self.original_image:
            return
        
        self.progress.start()
        threading.Thread(target=self._scan_full_page_thread, daemon=True).start()
    
    def _scan_full_page_thread(self):
        try:
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            method = self.preprocess_var.get()
            processed = self.ocr.preprocess_image(cv_image, method)
            
            ocr_method = self.ocr_method_var.get()
            confidence = self.conf_scale.get()
            ocr_text = self.ocr.extract_text(processed, ocr_method, method, confidence)
            
            self.master.after(0, lambda: self.ocr_preview.delete(1.0, tk.END))
            self.master.after(0, lambda: self.ocr_preview.insert(1.0, ocr_text))
            self.master.after(0, lambda: self.text_editor.delete(1.0, tk.END))
            self.master.after(0, lambda: self.text_editor.insert(1.0, ocr_text))
            self.master.after(0, lambda: self.text_length_label.config(text=f"{len(ocr_text)} chars"))
            self.master.after(0, lambda: self.update_status(f"Full page OCR: {len(ocr_text)} characters"))
        finally:
            self.master.after(0, self.progress.stop)
    
    def translate_selection(self):
        ocr_text = self.ocr_preview.get(1.0, tk.END).strip()
        if not ocr_text:
            return
        
        self.progress.start()
        threading.Thread(target=self._translate_thread, args=(ocr_text,), daemon=True).start()
    
    def _translate_thread(self, text):
        try:
            method = self.trans_method_var.get()
            
            if self.context_aware_var.get() and len(text) > 50:
                text = f"[Context: Comic/Manga Dialogue] {text}"
            
            translated = self.translator.translate(text, method)
            
            self.master.after(0, lambda: self.trans_preview.delete(1.0, tk.END))
            self.master.after(0, lambda: self.trans_preview.insert(1.0, translated))
            self.master.after(0, lambda: self.text_editor.delete(1.0, tk.END))
            self.master.after(0, lambda: self.text_editor.insert(1.0, translated))
            self.master.after(0, lambda: self.update_status(f"Translation completed ({method})"))
        finally:
            self.master.after(0, self.progress.stop)
    
    def apply_edited_text(self):
        text = self.text_editor.get(1.0, tk.END).strip()
        if not text or not self.bbox:
            return
        
        self.save_to_history()
        
        if self.remove_original_var.get():
            if self.vertical_mode and self.vertical_image:
                self.vertical_image = self.text_remover.remove_text(self.vertical_image, self.bbox, "inpaint")
            else:
                self.edited_image = self.text_remover.remove_text(self.edited_image, self.bbox, "inpaint")
        
        x1, y1, x2, y2 = [int(coord) for coord in self.bbox]
        
        if self.vertical_mode and self.vertical_image:
            target_image = self.vertical_image
        else:
            target_image = self.edited_image
        
        draw = ImageDraw.Draw(target_image, 'RGBA')
        width = x2 - x1
        height = y2 - y1
        
        if self.bg_fill_var.get():
            bg_color = self.hex_to_rgba(self.bg_color, self.bg_opacity)
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
        try:
            font_path = self.find_font_path(self.current_font)
            font = ImageFont.truetype(font_path, self.font_size) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        text_color = self.hex_to_rgba(self.font_color)
        outline_color = self.hex_to_rgba(self.outline_color)
        
        lines = self.wrap_text(text, font, width - 20)
        
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            line_heights.append(line_height)
            total_height += line_height * self.line_spacing
        
        y = y1 + (height - total_height) // 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if self.align_var.get() == "center":
                x = x1 + (width - text_width) // 2
            elif self.align_var.get() == "right":
                x = x2 - text_width - 10
            else:
                x = x1 + 10
            
            if self.outline_width > 0:
                for dx in range(-self.outline_width, self.outline_width + 1):
                    for dy in range(-self.outline_width, self.outline_width + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((x + dx, y + dy), line, font=font, fill=outline_color)
            
            draw.text((x, y), line, font=font, fill=text_color)
            y += line_heights[i] * self.line_spacing
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Text applied successfully")
    
    def wrap_text(self, text, font, max_width):
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width > max_width:
                if len(current_line) == 1:
                    lines.append(test_line)
                    current_line = []
                else:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def hex_to_rgba(self, hex_color, opacity=1.0):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
        else:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(opacity * 255)
        return (r, g, b, a)
    
    def auto_translate_page(self):
        if not self.original_image:
            return
        
        self.progress.start()
        threading.Thread(target=self._auto_translate_page_thread, daemon=True).start()
    
    def _auto_translate_page_thread(self):
        try:
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            
            method = self.preprocess_var.get()
            processed = self.ocr.preprocess_image(cv_image, method)
            
            ocr_method = self.ocr_method_var.get()
            confidence = self.conf_scale.get()
            
            if self.ocr.readers.get("easyocr") and ocr_method in ['easyocr', 'combined']:
                results = self.ocr.readers["easyocr"].readtext(cv_image, detail=1)
                text_areas = []
                
                for result in results:
                    bbox, text, confidence_score = result
                    if confidence_score >= confidence:
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        x1, y1 = int(min(xs)), int(min(ys))
                        x2, y2 = int(max(xs)), int(max(ys))
                        text_areas.append((x1, y1, x2, y2, text))
            else:
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_areas = []
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20 and w < cv_image.shape[1] * 0.8:
                        cropped = self.original_image.crop((x, y, x+w, y+h))
                        cv_cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
                        ocr_text = self.ocr.extract_text(cv_cropped, ocr_method, method, confidence)
                        if ocr_text.strip():
                            text_areas.append((x, y, x+w, y+h, ocr_text))
            
            self.save_to_history()
            self.edited_image = self.original_image.copy()
            
            translation_method = self.trans_method_var.get()
            
            for area in text_areas:
                x1, y1, x2, y2, text = area
                translated = self.translator.translate(text, translation_method)
                self.bbox = (x1, y1, x2, y2)
                self.text_editor.delete(1.0, tk.END)
                self.text_editor.insert(1.0, translated)
                self.apply_edited_text()
            
            if self.vertical_mode:
                self.update_vertical_display()
            else:
                self.update_canvas()
            
            self.master.after(0, lambda: self.update_status(f"Auto-translated {len(text_areas)} text areas"))
        finally:
            self.master.after(0, self.progress.stop)
    
    def auto_process_page(self):
        self.detect_bubbles()
        self.fill_all_bubbles()
        self.auto_translate_page()
        self.update_status("Auto-process completed")
    
    def detect_bubbles(self):
        if not self.original_image:
            return
        
        self.progress.start()
        threading.Thread(target=self._detect_bubbles_thread, daemon=True).start()
    
    def _detect_bubbles_thread(self):
        try:
            method = self.bubble_method_var.get()
            self.detected_bubbles = self.bubble_detector.detect_bubbles(self.original_image, method)
            
            self.bubble_listbox.delete(0, tk.END)
            for i, bubble in enumerate(self.detected_bubbles):
                x1, y1, x2, y2 = bubble
                self.bubble_listbox.insert(tk.END, f"Bubble {i+1}: {x1},{y1} - {x2},{y2}")
            
            self.current_bubble_index = 0
            if self.detected_bubbles:
                self.current_bubble_label.config(text=f"1/{len(self.detected_bubbles)}")
                self.bbox = self.detected_bubbles[0]
            
            self.master.after(0, lambda: self.update_status(f"Detected {len(self.detected_bubbles)} speech bubbles"))
        finally:
            self.master.after(0, self.progress.stop)
    
    def select_bubble(self):
        selection = self.bubble_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.detected_bubbles):
                self.current_bubble_index = index
                self.bbox = self.detected_bubbles[index]
                self.is_locked = True
                self.current_bubble_label.config(text=f"{index+1}/{len(self.detected_bubbles)}")
                self.update_status(f"Selected bubble {index+1}")
    
    def fill_selected_bubble(self):
        if not self.bbox:
            messagebox.showwarning("No Selection", "Please select a bubble first")
            return
        
        self.save_to_history()
        
        color = self.bubble_color
        opacity = self.bubble_opacity_scale.get()
        
        self.edited_image = self.bubble_detector.fill_bubble(self.edited_image, self.bbox, color, opacity)
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Bubble filled successfully")
    
    def fill_all_bubbles(self):
        if not self.detected_bubbles:
            messagebox.showwarning("No Bubbles", "No bubbles detected. Run detection first.")
            return
        
        self.save_to_history()
        
        color = self.bubble_color
        opacity = self.bubble_opacity_scale.get()
        
        for bubble in self.detected_bubbles:
            self.edited_image = self.bubble_detector.fill_bubble(self.edited_image, bubble, color, opacity)
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status(f"Filled {len(self.detected_bubbles)} bubbles")
    
    def apply_translation_to_bubble(self):
        if not self.bbox:
            messagebox.showwarning("No Bubble", "Please select a bubble first")
            return
        
        text = self.text_editor.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter or translate text first")
            return
        
        self.save_to_history()
        
        if self.bubble_fill_var.get():
            color = self.bubble_color
            opacity = self.bubble_opacity_scale.get()
            self.edited_image = self.bubble_detector.fill_bubble(self.edited_image, self.bbox, color, opacity)
        
        x1, y1, x2, y2 = [int(coord) for coord in self.bbox]
        
        draw = ImageDraw.Draw(self.edited_image, 'RGBA')
        width = x2 - x1
        height = y2 - y1
        
        if self.bg_fill_var.get():
            bg_color = self.hex_to_rgba(self.bg_color, self.bg_opacity)
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
        
        try:
            font_path = self.find_font_path(self.current_font)
            font = ImageFont.truetype(font_path, self.font_size) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        text_color = self.hex_to_rgba(self.font_color)
        outline_color = self.hex_to_rgba(self.outline_color)
        
        lines = self.wrap_text(text, font, width - 20)
        
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            line_heights.append(line_height)
            total_height += line_height * self.line_spacing
        
        y = y1 + (height - total_height) // 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if self.align_var.get() == "center":
                x = x1 + (width - text_width) // 2
            elif self.align_var.get() == "right":
                x = x2 - text_width - 10
            else:
                x = x1 + 10
            
            if self.outline_width > 0:
                for dx in range(-self.outline_width, self.outline_width + 1):
                    for dy in range(-self.outline_width, self.outline_width + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((x + dx, y + dy), line, font=font, fill=outline_color)
            
            draw.text((x, y), line, font=font, fill=text_color)
            y += line_heights[i] * self.line_spacing
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Translation applied to bubble")
    
    def apply_and_next_bubble(self):
        self.apply_translation_to_bubble()
        if self.detected_bubbles and self.current_bubble_index < len(self.detected_bubbles) - 1:
            self.current_bubble_index += 1
            self.bbox = self.detected_bubbles[self.current_bubble_index]
            self.current_bubble_label.config(text=f"{self.current_bubble_index+1}/{len(self.detected_bubbles)}")
            
            x1, y1, x2, y2 = [int(coord) for coord in self.bbox]
            cropped = self.original_image.crop((x1, y1, x2, y2))
            cv_image = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            
            method = self.preprocess_var.get()
            processed = self.ocr.preprocess_image(cv_image, method)
            
            ocr_method = self.ocr_method_var.get()
            confidence = self.conf_scale.get()
            ocr_text = self.ocr.extract_text(processed, ocr_method, method, confidence)
            
            self.ocr_preview.delete(1.0, tk.END)
            self.ocr_preview.insert(1.0, ocr_text)
            self.text_editor.delete(1.0, tk.END)
            self.text_editor.insert(1.0, ocr_text)
            
            self.update_status(f"Moved to bubble {self.current_bubble_index+1}")
    
    def apply_and_save(self):
        self.apply_translation_to_bubble()
        self.save_current_image()
    
    def apply_to_all_bubbles(self):
        if not self.detected_bubbles:
            messagebox.showwarning("No Bubbles", "No bubbles detected. Run detection first.")
            return
        
        text = self.text_editor.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter or translate text first")
            return
        
        self.save_to_history()
        
        for bubble in self.detected_bubbles:
            self.bbox = bubble
            self.apply_translation_to_bubble()
        
        self.update_status(f"Applied translation to all {len(self.detected_bubbles)} bubbles")
    
    def prev_bubble(self):
        if self.detected_bubbles and self.current_bubble_index > 0:
            self.current_bubble_index -= 1
            self.bbox = self.detected_bubbles[self.current_bubble_index]
            self.current_bubble_label.config(text=f"{self.current_bubble_index+1}/{len(self.detected_bubbles)}")
            self.update_status(f"Selected bubble {self.current_bubble_index+1}")
    
    def next_bubble(self):
        if self.detected_bubbles and self.current_bubble_index < len(self.detected_bubbles) - 1:
            self.current_bubble_index += 1
            self.bbox = self.detected_bubbles[self.current_bubble_index]
            self.current_bubble_label.config(text=f"{self.current_bubble_index+1}/{len(self.detected_bubbles)}")
            self.update_status(f"Selected bubble {self.current_bubble_index+1}")
    
    def clean_current_image(self):
        if not self.original_image:
            return
        
        self.save_to_history()
        self.edited_image = self.toolkit.clean_scan(self.original_image)
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Image cleaned")
    
    def apply_manga_effects(self):
        if not self.original_image:
            return
        
        self.save_to_history()
        self.edited_image = self.toolkit.apply_manga_effects(self.original_image)
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Manga effects applied")
    
    def save_and_next(self):
        self.save_current_image()
        if self.current_index < len(self.images) - 1:
            self.next_image()
    
    def choose_bubble_color(self):
        current_color = self.bubble_color
        if len(current_color) == 9 and current_color.startswith('#'):
            current_color = current_color[:7]
        
        color = colorchooser.askcolor(title="Choose Bubble Color", initialcolor=current_color)[1]
        if color:
            self.bubble_color = color
            self.bubble_color_label.config(foreground=color)
            self.update_status(f"Bubble color set to {color}")
    
    def clear_bubble_list(self):
        self.detected_bubbles = []
        self.bubble_listbox.delete(0, tk.END)
        self.current_bubble_index = 0
        self.current_bubble_label.config(text="0/0")
        self.update_status("Bubble list cleared")
    
    def batch_process_all(self):
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please load images first")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return
        
        self.progress.start()
        threading.Thread(target=self._batch_process_thread, args=(output_dir,), daemon=True).start()
    
    def _batch_process_thread(self, output_dir):
        try:
            total = len(self.image_paths)
            
            for idx, path in enumerate(self.image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    
                    if self.bubble_detect_var.get():
                        method = self.bubble_method_var.get()
                        bubbles = self.bubble_detector.detect_bubbles(img, method)
                        
                        for bubble in bubbles:
                            if self.bubble_fill_var.get():
                                color = self.bubble_color
                                opacity = self.bubble_opacity_scale.get()
                                img = self.bubble_detector.fill_bubble(img, bubble, color, opacity)
                    
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    method = self.preprocess_var.get()
                    processed = self.ocr.preprocess_image(cv_img, method)
                    
                    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    draw = ImageDraw.Draw(img, 'RGBA')
                    
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        if 50 < w < cv_img.shape[1] * 0.8 and 20 < h < cv_img.shape[0] * 0.8:
                            cropped = img.crop((x, y, x+w, y+h))
                            cv_cropped = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
                            ocr_text = self.ocr.extract_text(cv_cropped, self.ocr_method_var.get(), method, self.conf_scale.get())
                            
                            if ocr_text.strip():
                                try:
                                    translated = self.translator.translate(ocr_text, self.trans_method_var.get())
                                    
                                    if self.remove_original_var.get():
                                        img = self.text_remover.remove_text(img, (x, y, x+w, y+h), "inpaint")
                                        draw = ImageDraw.Draw(img, 'RGBA')
                                    
                                    if self.bg_fill_var.get():
                                        bg_color = self.hex_to_rgba(self.bg_color, self.bg_opacity)
                                        draw.rectangle([x, y, x+w, y+h], fill=bg_color)
                                    
                                    try:
                                        font_path = self.find_font_path(self.current_font)
                                        font = ImageFont.truetype(font_path, self.font_size) if font_path else ImageFont.load_default()
                                    except:
                                        font = ImageFont.load_default()
                                    
                                    text_color = self.hex_to_rgba(self.font_color)
                                    lines = self.wrap_text(translated, font, w - 20)
                                    
                                    y_text = y + 10
                                    for line in lines:
                                        bbox = draw.textbbox((0, 0), line, font=font)
                                        text_width = bbox[2] - bbox[0]
                                        x_text = x + (w - text_width) // 2
                                        draw.text((x_text, y_text), line, font=font, fill=text_color)
                                        y_text += bbox[3] - bbox[1] + 5
                                except Exception as e:
                                    print(f"Error processing text area: {e}")
                    
                    filename = os.path.basename(path)
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_dir, f"{name}_translated{ext}")
                    img.save(output_path, quality=95)
                    
                    self.master.after(0, lambda i=idx+1: self.update_status(f"Processed {i}/{total} images"))
                    
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            self.master.after(0, lambda: self.update_status(f"Batch processing completed! Saved to {output_dir}"))
            self.master.after(0, lambda: messagebox.showinfo("Complete", f"Processed {total} images"))
        finally:
            self.master.after(0, self.progress.stop)
    
    def batch_save_all(self):
        if not self.image_paths:
            messagebox.showwarning("No Images", "No images to save")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if output_dir:
            for idx, img in enumerate(self.images):
                filename = os.path.basename(self.image_paths[idx])
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_processed{ext}")
                img.save(output_path, quality=95)
            
            messagebox.showinfo("Saved", f"Saved {len(self.images)} images to {output_dir}")
            self.update_status(f"Saved {len(self.images)} images")
    
    def load_system_fonts(self):
        try:
            font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            font_names = []
            for font_path in font_list:
                try:
                    font = ImageFont.truetype(font_path, 10)
                    font_name = os.path.splitext(os.path.basename(font_path))[0]
                    font_names.append((font_name, font_path))
                except:
                    continue
            
            sorted_fonts = sorted(set([name for name, _ in font_names]))
            self.font_combo['values'] = sorted_fonts
            
            preferred_fonts = ['Arial', 'Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR']
            found_font = False
            for font in preferred_fonts:
                if font in sorted_fonts:
                    self.font_combo.set(font)
                    self.current_font = font
                    found_font = True
                    break
            
            if not found_font and sorted_fonts:
                self.font_combo.set(sorted_fonts[0])
                self.current_font = sorted_fonts[0]
            
            self.font_paths = dict(font_names)
        except Exception as e:
            print(f"Error loading fonts: {e}")
            self.font_combo.set("System Default")
    
    def find_font_path(self, font_name):
        if hasattr(self, 'font_paths'):
            for name, path in self.font_paths.items():
                if name.lower() == font_name.lower():
                    return path
        return None
    
    def on_font_select(self, event):
        self.current_font = self.font_combo.get()
        self.update_status(f"Selected font: {self.current_font}")
    
    def adjust_font_size(self, delta):
        self.font_size = max(8, min(72, self.font_size + delta))
        self.size_label.config(text=str(self.font_size))
    
    def choose_font_color(self):
        current_color = self.font_color
        if len(current_color) == 9 and current_color.startswith('#'):
            current_color = current_color[:7]
        
        color = colorchooser.askcolor(title="Choose Text Color", initialcolor=current_color)[1]
        if color:
            self.font_color = color
            self.update_status(f"Text color: {color}")
    
    def choose_outline_color(self):
        current_color = self.outline_color
        if len(current_color) == 9 and current_color.startswith('#'):
            current_color = current_color[:7]
        
        color = colorchooser.askcolor(title="Choose Outline Color", initialcolor=current_color)[1]
        if color:
            self.outline_color = color
            self.update_status(f"Outline color: {color}")
    
    def choose_bg_color(self):
        current_color = self.bg_color
        if len(current_color) == 9 and current_color.startswith('#'):
            current_color = current_color[:7]
        
        color = colorchooser.askcolor(title="Choose Background Color", initialcolor=current_color)[1]
        if color:
            self.bg_color = color
            self.update_status(f"Background color: {color}")
    
    def update_outline_width(self):
        self.outline_width = int(self.outline_scale.get())
    
    def update_bg_opacity(self):
        self.bg_opacity = self.bg_opacity_scale.get()
    
    def save_current_image(self):
        if not self.edited_image:
            messagebox.showwarning("No Image", "No image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("WebP files", "*.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                    self.edited_image.save(file_path, 'JPEG', quality=95)
                else:
                    self.edited_image.save(file_path)
                self.update_status(f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {str(e)}")
    
    def save_to_history(self):
        if self.edited_image:
            self.work_history.append(self.edited_image.copy())
            if len(self.work_history) > 50:
                self.work_history.pop(0)
            self.update_history_list()
    
    def update_history_list(self):
        self.history_listbox.delete(0, tk.END)
        for i in range(len(self.work_history)):
            self.history_listbox.insert(tk.END, f"State {i+1}")
    
    def restore_from_history(self):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0]
            if index < len(self.work_history):
                self.edited_image = self.work_history[index].copy()
                if self.vertical_mode:
                    self.update_vertical_display()
                else:
                    self.update_canvas()
                self.update_status(f"Restored state {index+1}")
    
    def clear_history(self):
        if self.edited_image:
            self.work_history = [self.edited_image.copy()]
            self.update_history_list()
            self.update_status("History cleared")
    
    def undo_action(self):
        if len(self.work_history) > 1:
            self.work_history.pop()
            if self.work_history:
                self.edited_image = self.work_history[-1].copy()
                if self.vertical_mode:
                    self.update_vertical_display()
                else:
                    self.update_canvas()
                self.update_history_list()
                self.update_status("Undo successful")
    
    def redo_action(self):
        pass
    
    def clear_editor(self):
        self.text_editor.delete(1.0, tk.END)
        self.ocr_preview.delete(1.0, tk.END)
        self.trans_preview.delete(1.0, tk.END)
    
    def copy_to_clipboard(self):
        text = self.text_editor.get(1.0, tk.END).strip()
        if text:
            self.master.clipboard_clear()
            self.master.clipboard_append(text)
            self.update_status("Text copied to clipboard")
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self.display_current_image()
    
    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)
            self.display_current_image()
    
    def remove_image(self):
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            self.image_listbox.delete(index)
            del self.images[index]
            del self.image_paths[index]
            
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)
            
            if self.images:
                self.image_listbox.selection_clear(0, tk.END)
                self.image_listbox.selection_set(self.current_index)
                self.display_current_image()
            else:
                self.original_image = None
                self.edited_image = None
                self.canvas.delete("all")
                self.update_status("All images removed")
    
    def reset_view(self):
        self.bbox = None
        self.is_locked = False
        if hasattr(self, 'selection_rect') and self.selection_rect:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
    
    def on_mouse_move(self, event):
        if self.vertical_mode and self.vertical_image:
            img_width, img_height = self.vertical_image.size
            canvas_width = self.canvas.winfo_width()
            scale = min(canvas_width / img_width, 2.0)
            x = event.x / scale
            y = (event.y + self.canvas.yview()[0] * img_height) / scale
            self.coord_label.config(text=f"X: {int(x)}, Y: {int(y)}")
        else:
            self.coord_label.config(text=f"X: {event.x}, Y: {event.y}")
    
    def enhance_current_image(self):
        if not self.original_image:
            return
        
        settings = {
            "auto_adjust": self.auto_adjust_var.get(),
            "contrast": self.contrast_scale.get(),
            "brightness": self.brightness_scale.get(),
            "sharpness": self.sharpness_scale.get(),
            "despeckle": self.despeckle_var.get()
        }
        
        self.save_to_history()
        self.edited_image = self.image_enhancer.enhance(self.edited_image, settings)
        
        if self.vertical_mode:
            self.update_vertical_display()
        else:
            self.update_canvas()
        
        self.update_status("Image enhanced")
    
    def apply_image_effects(self):
        self.enhance_current_image()
    
    def reset_image_effects(self):
        self.contrast_scale.set(1.0)
        self.brightness_scale.set(1.0)
        self.sharpness_scale.set(1.0)
        self.auto_adjust_var.set(True)
        self.despeckle_var.set(True)
        self.update_status("Effects reset to default")
    
    def open_settings(self):
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        settings_window.configure(bg=self.colors['bg_color'])
        settings_window.transient(self.master)
        settings_window.grab_set()
        
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_api_keys_tab(notebook)
        self.create_appearance_tab(notebook)
        
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save", command=lambda: self.save_settings_and_close(settings_window)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=2)
    
    def create_api_keys_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="🔑 API Keys")
        
        api_keys = self.settings.settings["api_keys"]
        
        ttk.Label(tab, text="OpenAI API Key:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.openai_key_var = tk.StringVar(value=api_keys["openai"])
        openai_entry = ttk.Entry(tab, textvariable=self.openai_key_var, show="*", width=50)
        openai_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(tab, text="Google Gemini API Key:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.gemini_key_var = tk.StringVar(value=api_keys["gemini"])
        gemini_entry = ttk.Entry(tab, textvariable=self.gemini_key_var, show="*", width=50)
        gemini_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(tab, text="Claude API Key:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.claude_key_var = tk.StringVar(value=api_keys["claude"])
        claude_entry = ttk.Entry(tab, textvariable=self.claude_key_var, show="*", width=50)
        claude_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(tab, text="DeepL API Key:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.deepl_key_var = tk.StringVar(value=api_keys["deepl"])
        deepl_entry = ttk.Entry(tab, textvariable=self.deepl_key_var, show="*", width=50)
        deepl_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(tab, text="Microsoft Translator Key:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.microsoft_key_var = tk.StringVar(value=api_keys["microsoft"])
        microsoft_entry = ttk.Entry(tab, textvariable=self.microsoft_key_var, show="*", width=50)
        microsoft_entry.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(tab, text="Note: API keys are stored locally in settings.json", 
                 font=('Segoe UI', 8)).pack(anchor=tk.W, pady=(20, 0), padx=10)
    
    def create_appearance_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="🎨 Appearance")
        
        ui_settings = self.settings.settings["ui"]
        
        ttk.Label(tab, text="Theme:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.theme_var = tk.StringVar(value=ui_settings["theme"])
        theme_frame = ttk.Frame(tab)
        theme_frame.pack(fill=tk.X, padx=10)
        ttk.Radiobutton(theme_frame, text="Dark", variable=self.theme_var, value="dark").pack(side=tk.LEFT)
        ttk.Radiobutton(theme_frame, text="Light", variable=self.theme_var, value="light").pack(side=tk.LEFT, padx=20)
        
        ttk.Label(tab, text="UI Font Size:").pack(anchor=tk.W, pady=(10, 0), padx=10)
        self.font_size_var = tk.IntVar(value=ui_settings["font_size"])
        font_size_scale = ttk.Scale(tab, from_=8, to=16, variable=self.font_size_var)
        font_size_scale.pack(fill=tk.X, padx=10, pady=2)
        
        self.show_grid_var = tk.BooleanVar(value=ui_settings["show_grid"])
        ttk.Checkbutton(tab, text="Show Grid Overlay", variable=self.show_grid_var).pack(anchor=tk.W, pady=10, padx=10)
        
        self.vertical_mode_var = tk.BooleanVar(value=ui_settings["vertical_mode"])
        ttk.Checkbutton(tab, text="Enable Vertical Mode", variable=self.vertical_mode_var).pack(anchor=tk.W, pady=5, padx=10)
        
        self.auto_scroll_var = tk.BooleanVar(value=ui_settings["auto_scroll"])
        ttk.Checkbutton(tab, text="Auto-Scroll in Vertical Mode", variable=self.auto_scroll_var).pack(anchor=tk.W, pady=5, padx=10)
    
    def save_settings_and_close(self, window):
        self.settings.settings["api_keys"]["openai"] = self.openai_key_var.get()
        self.settings.settings["api_keys"]["gemini"] = self.gemini_key_var.get()
        self.settings.settings["api_keys"]["claude"] = self.claude_key_var.get()
        self.settings.settings["api_keys"]["deepl"] = self.deepl_key_var.get()
        self.settings.settings["api_keys"]["microsoft"] = self.microsoft_key_var.get()
        
        self.settings.settings["ui"]["theme"] = self.theme_var.get()
        self.settings.settings["ui"]["font_size"] = self.font_size_var.get()
        self.settings.settings["ui"]["show_grid"] = self.show_grid_var.get()
        self.settings.settings["ui"]["vertical_mode"] = self.vertical_mode_var.get()
        self.settings.settings["ui"]["auto_scroll"] = self.auto_scroll_var.get()
        
        self.settings.settings["text"]["bubble_color"] = self.bubble_color
        self.settings.settings["text"]["bubble_detect"] = self.bubble_detect_var.get()
        self.settings.settings["text"]["bubble_fill"] = self.bubble_fill_var.get()
        
        self.settings.save_settings()
        self.translator.loading_models()
        
        window.destroy()
        self.update_status("Settings saved successfully")
        
        if self.vertical_mode_var.get() != self.vertical_mode:
            self.vertical_mode = self.vertical_mode_var.get()
            if self.vertical_mode:
                self.toggle_vertical_mode()
    
    def update_status(self, message):
        self.status_label.config(text=message)
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.settings.save_settings()
            self.master.destroy()

def main():
    root = tk.Tk()
    app = Translations_ToolKit(root)
    root.mainloop()

if __name__ == "__main__":
    main()
