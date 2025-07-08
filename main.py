import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from core.style_transfer import stylize
import os
import threading

class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Style Transfer")
        self.root.geometry("1000x700")
        self.content_path = None
        self.style_path = None
        self.output_path = None

        tk.Label(root, text="Neural Style Transfer", font=("Helvetica", 24)).pack(pady=10)

        # Upload Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Upload Content Image", command=self.upload_content).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Upload Style Image", command=self.upload_style).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="Start Style Transfer", command=self.run_transfer).grid(row=0, column=2, padx=10)

        # Image display frames
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=20)

        self.content_label = tk.Label(self.image_frame, text="Content Image")
        self.content_label.grid(row=0, column=0)

        self.style_label = tk.Label(self.image_frame, text="Style Image")
        self.style_label.grid(row=0, column=1)

        self.output_label = tk.Label(self.image_frame, text="Output Image")
        self.output_label.grid(row=0, column=2)

        self.content_img_panel = tk.Label(self.image_frame)
        self.content_img_panel.grid(row=1, column=0, padx=20)

        self.style_img_panel = tk.Label(self.image_frame)
        self.style_img_panel.grid(row=1, column=1, padx=20)

        self.output_img_panel = tk.Label(self.image_frame)
        self.output_img_panel.grid(row=1, column=2, padx=20)

    def upload_content(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.content_path = path
            self.display_image(path, self.content_img_panel, (300, 300))
            messagebox.showinfo("Selected", f"Content Image: {os.path.basename(path)}")

    def upload_style(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.style_path = path
            self.display_image(path, self.style_img_panel, (300, 300))
            messagebox.showinfo("Selected", f"Style Image: {os.path.basename(path)}")

    def run_transfer(self):
        if not self.content_path or not self.style_path:
            messagebox.showwarning("Missing", "Upload both content and style images first.")
            return

        def task():
            try:
                self.root.after(0, lambda: self.set_buttons_state("disabled"))
                self.root.after(0, lambda: self.update_status("Processing... Please wait."))

                output = stylize(self.content_path, self.style_path)
                self.output_path = output

                self.display_image(output, self.output_img_panel, (300, 300))

                self.root.after(0, lambda: self.update_status("Style Transfer Complete"))
                messagebox.showinfo("Completed", f"Output saved to:\n{output}")
            except Exception as e:
                print("[ERROR]", e)
                messagebox.showerror("Error", f"An error occurred:\n{e}")
            finally:
                self.root.after(0, lambda: self.set_buttons_state("normal"))

        threading.Thread(target=task).start()

    def display_image(self, path, label_widget, size=(300, 300)):
        try:
            img = Image.open(path)
            img = img.resize(size)
            img = ImageTk.PhotoImage(img)
            label_widget.configure(image=img)
            label_widget.image = img  # Keep reference!
        except Exception as e:
            print(f"[ERROR] Failed to display image {path}: {e}")

    def set_buttons_state(self, state):
        for child in self.root.winfo_children():
            for grandchild in child.winfo_children():
                if isinstance(grandchild, tk.Button):
                    grandchild.config(state=state)

    def update_status(self, text):
        self.root.title(f"Neural Style Transfer - {text}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
