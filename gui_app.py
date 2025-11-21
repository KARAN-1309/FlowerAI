import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
import os
import joblib
import pandas as pd

# --- CONFIGURATION ---
IMAGE_MODEL_PATH = "flower_classifier_model.keras"
IRIS_MODEL_PATH = "iris_tabular_model.pkl"
IRIS_ENCODER_PATH = "iris_label_encoder.pkl"

IMG_HEIGHT = 180
IMG_WIDTH = 180
IMAGE_CLASS_NAMES = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 
                     'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 
                     'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']

# --- THEME COLORS ---
COLOR_BG = "#2b2b2b"        # Dark Background
COLOR_FG = "#ffffff"        # White Text
COLOR_ACCENT_1 = "#43a047"  # Green (Image Tab)
COLOR_ACCENT_2 = "#1e88e5"  # Blue (Iris Tab)
COLOR_SECONDARY = "#3c3f41" # Slightly lighter background for inputs
FONT_MAIN = ("Segoe UI", 12)
FONT_HEADER = ("Segoe UI", 20, "bold")
FONT_RESULT = ("Segoe UI", 14, "bold")

class FlowerSuperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flower AI System")
        self.root.geometry("650x750")
        self.root.configure(bg=COLOR_BG)
        
        # Configure Dark Theme Styles
        self.setup_styles()

        # Load Models
        self.image_model = self.load_image_model()
        self.iris_model, self.iris_encoder = self.load_iris_model()

        # --- TABS SETUP ---
        # Custom style is applied via 'TNotebook'
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Create Frames for Tabs
        self.tab_image = tk.Frame(self.notebook, bg=COLOR_BG)
        self.tab_iris = tk.Frame(self.notebook, bg=COLOR_BG)

        self.notebook.add(self.tab_image, text=" 🌻 Visual Recognition ")
        self.notebook.add(self.tab_iris, text=" 📐 Iris Measurements ")

        # Build UI for each tab
        self.build_image_tab()
        self.build_iris_tab()

    def setup_styles(self):
        style = ttk.Style()
        
        # 'clam' theme usually allows more color customization than 'vista' or 'aqua'
        try:
            style.theme_use('clam')
        except:
            pass

        # Notebook (Tabs) Styling
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", 
                        background=COLOR_SECONDARY, 
                        foreground="lightgray", 
                        padding=[20, 10], 
                        font=("Segoe UI", 11))
        style.map("TNotebook.Tab", 
                  background=[("selected", COLOR_ACCENT_2)], 
                  foreground=[("selected", "white")])

    def load_image_model(self):
        if not os.path.exists(IMAGE_MODEL_PATH):
            print(f"Warning: {IMAGE_MODEL_PATH} not found. Please run train_classifier.py.")
            return None
        try:
            return tf.keras.models.load_model(IMAGE_MODEL_PATH)
        except Exception as e:
            print(f"Error loading image model: {e}")
            return None

    def load_iris_model(self):
        if not os.path.exists(IRIS_MODEL_PATH) or not os.path.exists(IRIS_ENCODER_PATH):
            print("Warning: Iris model files not found. Please run train_iris.py.")
            return None, None
        try:
            model = joblib.load(IRIS_MODEL_PATH)
            encoder = joblib.load(IRIS_ENCODER_PATH)
            return model, encoder
        except Exception as e:
            print(f"Error loading Iris model: {e}")
            return None, None

    # -----------------------------------------------------------
    # TAB 1: IMAGE CLASSIFIER LOGIC
    # -----------------------------------------------------------
    def build_image_tab(self):
        # Container
        frame = tk.Frame(self.tab_image, bg=COLOR_BG)
        frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Title
        lbl_title = tk.Label(frame, text="Visual Flower Recognition", font=FONT_HEADER, bg=COLOR_BG, fg=COLOR_FG)
        lbl_title.pack(pady=(10, 30))

        # Image Display Area (Placeholder)
        self.image_panel = tk.Label(frame, bg=COLOR_SECONDARY, text="No Image Selected", fg="gray", height=15, width=40)
        self.image_panel.pack(pady=10, fill='x')

        # Button
        btn_load = tk.Button(frame, text="📂 Upload Photo", command=self.load_image, 
                             font=FONT_MAIN, bg=COLOR_ACCENT_1, fg="white", 
                             activebackground="#388E3C", activeforeground="white",
                             relief="flat", padx=20, pady=10, cursor="hand2")
        btn_load.pack(pady=20)

        # Result
        self.lbl_img_result = tk.Label(frame, text="...", font=FONT_RESULT, bg=COLOR_BG, fg=COLOR_FG)
        self.lbl_img_result.pack(pady=20)

    def load_image(self):
        if self.image_model is None:
            messagebox.showerror("Error", "Image Model not loaded. Check if 'flower_classifier_model.keras' exists.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        try:
            # Display
            img_display = Image.open(file_path)
            # Resize to fit nicely
            img_display.thumbnail((400, 400), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_display)
            
            # Update panel to show image
            self.image_panel.config(image=self.photo, text="", height=0, width=0, bg=COLOR_BG)
            self.image_panel.image = self.photo
            
            # Predict
            self.predict_image_file(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")

    def predict_image_file(self, file_path):
        try:
            img = tf.keras.utils.load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = self.image_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_class = IMAGE_CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)

            result_text = f"I am {confidence:.1f}% sure this is a\n{predicted_class.upper()}"
            self.lbl_img_result.config(text=result_text, fg=COLOR_ACCENT_1)
        except Exception as e:
            self.lbl_img_result.config(text="Prediction Failed", fg="red")
            print(e)

    # -----------------------------------------------------------
    # TAB 2: IRIS MEASUREMENT LOGIC
    # -----------------------------------------------------------
    def build_iris_tab(self):
        # Container
        frame = tk.Frame(self.tab_iris, bg=COLOR_BG)
        frame.pack(expand=True, fill='both', padx=40, pady=40)

        lbl_title = tk.Label(frame, text="Iris Measurement Classifier", font=FONT_HEADER, bg=COLOR_BG, fg=COLOR_FG)
        lbl_title.pack(pady=(0, 30))

        if self.iris_model is None:
            tk.Label(frame, text="Model missing! Please run train_iris.py", fg="red", bg=COLOR_BG, font=FONT_MAIN).pack()
            return

        # Input Frame
        frame_inputs = tk.Frame(frame, bg=COLOR_BG)
        frame_inputs.pack(pady=10)

        # Inputs with styling
        self.entry_sepal_l = self.create_modern_input(frame_inputs, "Sepal Length (cm):", 0)
        self.entry_sepal_w = self.create_modern_input(frame_inputs, "Sepal Width (cm):", 1)
        self.entry_petal_l = self.create_modern_input(frame_inputs, "Petal Length (cm):", 2)
        self.entry_petal_w = self.create_modern_input(frame_inputs, "Petal Width (cm):", 3)

        # Predict Button
        btn_predict_iris = tk.Button(frame, text="🔍 Identify Species", command=self.predict_iris, 
                                     font=FONT_MAIN, bg=COLOR_ACCENT_2, fg="white",
                                     activebackground="#1565C0", activeforeground="white",
                                     relief="flat", padx=20, pady=10, cursor="hand2")
        btn_predict_iris.pack(pady=30)

        # Result Label
        self.lbl_iris_result = tk.Label(frame, text="Enter measurements above", font=FONT_RESULT, bg=COLOR_BG, fg="gray")
        self.lbl_iris_result.pack(pady=10)

    def create_modern_input(self, parent, label_text, row):
        # Label
        tk.Label(parent, text=label_text, font=("Segoe UI", 11), bg=COLOR_BG, fg=COLOR_FG).grid(row=row, column=0, padx=10, pady=10, sticky="e")
        
        # Entry with dark style
        entry = tk.Entry(parent, font=("Segoe UI", 11), width=15, bg=COLOR_SECONDARY, fg="white", 
                         insertbackground="white", relief="flat", borderwidth=5)
        entry.grid(row=row, column=1, padx=10, pady=10)
        return entry

    def predict_iris(self):
        try:
            # Get values
            val1 = float(self.entry_sepal_l.get())
            val2 = float(self.entry_sepal_w.get())
            val3 = float(self.entry_petal_l.get())
            val4 = float(self.entry_petal_w.get())

            # Predict
            features = pd.DataFrame([[val1, val2, val3, val4]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']) 
            pred_idx = self.iris_model.predict(features)[0]
            pred_name = self.iris_encoder.inverse_transform([pred_idx])[0]

            self.lbl_iris_result.config(text=f"Species: {pred_name}", fg=COLOR_ACCENT_2)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FlowerSuperApp(root)
        root.mainloop()
    except ImportError:
        print("Libraries missing. Please run: pip install -r requirements.txt")