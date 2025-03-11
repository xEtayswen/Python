import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def cv2_to_tkimage(cv2_img, is_gray=False):
    """
    Konwertuje obraz (numpy array) z OpenCV na obraz odpowiedni dla Tkinter (ImageTk.PhotoImage).
    Jeśli obraz jest w skali szarości, nie wykonuje konwersji kolorów.
    """
    if is_gray:
        pil_image = Image.fromarray(cv2_img)
    else:
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_img_rgb)
    return ImageTk.PhotoImage(pil_image)

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Przetwarzacz Obrazów z OpenCV")
        self.geometry("1300x900")
        self.configure(bg="#f0f0f0")
        
        self.create_menu()
        
        header_frame = tk.Frame(self, bg="#4a7abc", padx=20, pady=10)
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(header_frame, text="Przetwarzacz Obrazów z OpenCV", 
                                font=("Helvetica", 24, "bold"), bg="#4a7abc", fg="white")
        header_label.pack()
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_processing = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_processing, text="Przetwarzanie")
        
        self.tab_histogram = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_histogram, text="Histogram")
        
        self.image_frame = tk.Frame(self.tab_processing, bg="#f0f0f0", bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_label = tk.Label(self.image_frame, bg="#f0f0f0")
        self.image_label.pack(expand=True)
        
        self.control_frame = tk.Frame(self.tab_processing, bg="#f0f0f0")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.load_button = ttk.Button(self.control_frame, text="Wczytaj obraz", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Wybierz operację:", font=("Helvetica", 10)).grid(row=0, column=1, padx=5, pady=5)
        self.operation_var = tk.StringVar()
        self.operation_combo = ttk.Combobox(self.control_frame, textvariable=self.operation_var, state="readonly",
                                            values=["Oryginalny", "Skala szarości", "Wygładzony", "Krawędzie (Canny)",
                                                    "Kontury", "Erozja", "Dylatacja"])
        self.operation_combo.current(0)
        self.operation_combo.grid(row=0, column=2, padx=5, pady=5)
        self.operation_combo.bind("<<ComboboxSelected>>", lambda e: self.update_image())
        
        ttk.Label(self.control_frame, text="Threshold1:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.thresh1 = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
        self.thresh1.set(50)
        self.thresh1.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Threshold2:").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.thresh2 = tk.Scale(self.control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
        self.thresh2.set(150)
        self.thresh2.grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Rozmiar jądra:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.kernel_slider = tk.Scale(self.control_frame, from_=1, to=31, orient=tk.HORIZONTAL, length=200)
        self.kernel_slider.set(5)
        self.kernel_slider.grid(row=2, column=1, padx=5, pady=5)
        
        self.update_button = ttk.Button(self.control_frame, text="Aktualizuj", command=self.update_image)
        self.update_button.grid(row=2, column=2, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.control_frame, text="Zapisz obraz", command=self.save_image)
        self.save_button.grid(row=2, column=3, padx=5, pady=5)
        
        self.original_image = None
        self.processed_image = None
        
        self.figure = Figure(figsize=(5,4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Histogram jasności")
        self.ax.set_xlabel("Intensywność")
        self.ax.set_ylabel("Liczba pikseli")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_histogram)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_menu(self):
        """Tworzy pasek menu aplikacji."""
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Otwórz", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjście", command=self.quit)
        menubar.add_cascade(label="Plik", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="O programie", command=self.show_about)
        menubar.add_cascade(label="Pomoc", menu=help_menu)
        self.config(menu=menubar)
    
    def show_about(self):
        """Wyświetla informacje o programie."""
        messagebox.showinfo("O programie", "Projekt przetwarzania obrazów z OpenCV\n"
                              "z interaktywnym GUI, histogramem oraz opcjami przetwarzania.\n\nAutor: Robert Koszowski, Dawid Szymański")
    
    def load_image(self):
        """Wczytuje obraz przy użyciu okna dialogowego."""
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Błąd", "Nie udało się wczytać obrazu.")
                return
            self.update_image()
    
    def update_image(self):
        """Aktualizuje przetworzony obraz na podstawie wybranej operacji i parametrów."""
        if self.original_image is None:
            return
        
        op = self.operation_var.get()
        img = self.original_image.copy()
        
        if op == "Oryginalny":
            self.processed_image = img
            
        elif op == "Skala szarości":
            self.processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        elif op == "Wygładzony":
            k = self.kernel_slider.get()
            if k % 2 == 0:
                k += 1
            self.processed_image = cv2.GaussianBlur(img, (k, k), 0)
            
        elif op == "Krawędzie (Canny)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            t1 = self.thresh1.get()
            t2 = self.thresh2.get()
            self.processed_image = cv2.Canny(gray, threshold1=t1, threshold2=t2)
            
        elif op == "Kontury":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            t1 = self.thresh1.get()
            t2 = self.thresh2.get()
            edges = cv2.Canny(gray, threshold1=t1, threshold2=t2)
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            self.processed_image = contour_img
            
        elif op == "Erozja":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            k = self.kernel_slider.get()
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)
            self.processed_image = cv2.erode(gray, kernel, iterations=1)
            
        elif op == "Dylatacja":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            k = self.kernel_slider.get()
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)
            self.processed_image = cv2.dilate(gray, kernel, iterations=1)
            
        else:
            self.processed_image = img
        
        if len(self.processed_image.shape) == 2:
            tk_img = cv2_to_tkimage(self.processed_image, is_gray=True)
        else:
            tk_img = cv2_to_tkimage(self.processed_image, is_gray=False)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img
        
        self.update_histogram()
    
    def update_histogram(self):
        """Aktualizuje wykres histogramu na podstawie przetworzonego obrazu."""
        if self.processed_image is None:
            return
        
        if len(self.processed_image.shape) == 3:
            gray_img = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.processed_image
        
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        
        self.ax.clear()
        self.ax.set_title("Histogram jasności")
        self.ax.set_xlabel("Intensywność")
        self.ax.set_ylabel("Liczba pikseli")
        self.ax.plot(hist, color="blue")
        self.ax.set_xlim([0, 256])
        self.canvas.draw()
    
    def save_image(self):
        """Zapisuje przetworzony obraz do pliku."""
        if self.processed_image is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")],
            title="Zapisz obraz jako..."
        )
        if file_path:
            if len(self.processed_image.shape) == 2:
                cv2.imwrite(file_path, self.processed_image)
            else:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            messagebox.showinfo("Sukces", f"Obraz zapisano jako:\n{file_path}")

if __name__ == '__main__':
    app = ImageProcessorApp()
    app.mainloop()
