import os
import cv2
import csv
import re
import numpy as np
import threading
import time
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import torch

# Suppress HuggingFace/Windows warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- VIDEO UTILITIES ---
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip

class SLAMProcessor:
    def __init__(self, input_dir, batch_name, blur_th, shift_th, use_ai, log_callback, progress_callback, frame_callback, score_callback):
        self.input_dir = input_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        raw_name = batch_name.strip()
        self.clean_batch = re.sub(r'[\\/*?:"<>|]', "_", raw_name) if raw_name else "Batch"
        
        self.output_dir = os.path.join(input_dir, f"{self.clean_batch}_{self.timestamp}")
        self.csv_path = os.path.join(self.output_dir, f"Summary_{self.clean_batch}_{self.timestamp}.csv")
        
        self.blur_th = blur_th
        self.shift_th = shift_th
        self.use_ai = use_ai
        self.log_fn = log_callback
        self.update_progress = progress_callback
        self.update_frame = frame_callback
        self.update_scores = score_callback
        self.stop_event = threading.Event()

        # --- OPTIONAL AI INIT ---
        self.model = None
        self.tokenizer = None
        if self.use_ai:
            self.log_fn("Waking up Moondream AI...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
            model_id = "vikhyatk/moondream2"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, revision="2024-03-06", torch_dtype=self.dtype
                ).to(self.device)
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-03-06")
                self.log_fn(f"AI Engine Online ({self.device.upper()})")
            except Exception as e:
                self.log_fn(f"AI LOAD ERROR: {str(e)}")
                self.use_ai = False
        else:
            self.log_fn("AI Engine Disabled. Fast processing mode active.")

        self.summary_data = {"cleared_vids": 0, "cleared_frames": 0, "unclear_vids": 0, "unclear_frames": 0, "total_frames_batch": 0, "total_time": 0}
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def get_ai_desc(self, frame):
        if not self.use_ai or self.model is None: return "AI Disabled"
        try:
            with torch.inference_mode():
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enc_image = self.model.encode_image(pil_img)
                return self.model.answer_question(enc_image, "Describe the underwater scene briefly.", self.tokenizer).strip()
        except Exception as e:
            return f"AI Error: {str(e)[:20]}"

    def analyze_video(self, path, v_idx, v_total):
        v_start = time.time()
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        blur_scores, shift_scores, frame_cache, prev_gray = [], [], {}, None
        
        for f_idx in range(total_f):
            if self.stop_event.is_set(): break
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            b_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(b_score)
            
            s_score = 0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                s_score = np.mean(mag)
            shift_scores.append(s_score)
            
            if f_idx % 5 == 0:
                self.update_scores(b_score, s_score, f_idx, total_f)
                self.update_frame(frame)
            if f_idx % 30 == 0:
                self.update_progress((v_idx / v_total) + ((f_idx / total_f) / v_total))
            
            if self.use_ai and f_idx % 20 == 0: frame_cache[f_idx] = frame.copy()
            prev_gray = gray

        cap.release()
        intervals = self.get_intervals(blur_scores, shift_scores, fps)
        ai_descriptions = []
        if self.use_ai:
            for s, e in intervals:
                mid_f = int(((s + e) / 2) * fps)
                closest = min(frame_cache.keys(), key=lambda k: abs(k-mid_f)) if frame_cache else None
                ai_descriptions.append(self.get_ai_desc(frame_cache[closest]) if closest is not None else "No frame")
        else:
            ai_descriptions = ["AI Disabled"] * len(intervals)

        return intervals, ai_descriptions, round(time.time() - v_start, 2), total_f

    def get_intervals(self, blurs, shifts, fps):
        intervals, start = [], None
        for i, (b, s) in enumerate(zip(blurs, shifts)):
            is_good = b > self.blur_th and round(s, 1) <= round(self.shift_th, 1)
            if is_good and start is None: start = i
            elif not is_good and start is not None:
                if (i - start) / fps >= 1.5: intervals.append((round(start/fps, 2), round(i/fps, 2)))
                start = None
        return intervals

    def process_all(self, on_finish):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.mp4', '.mov'))]
        records = []
        batch_start = time.time()
        
        for i, filename in enumerate(files):
            if self.stop_event.is_set(): break
            path = os.path.join(self.input_dir, filename)
            intervals, ai_descs, proc_time, total_f = self.analyze_video(path, i, len(files))
            self.summary_data["total_frames_batch"] += total_f
            
            if not intervals:
                interval_data, ai_col = "Unclear", "N/A"
                self.summary_data["unclear_vids"] += 1; self.summary_data["unclear_frames"] += total_f
            else:
                self.summary_data["cleared_vids"] += 1; self.summary_data["cleared_frames"] += total_f
                formatted = [f"{s}-{e}_CLEAN_{idx}_{filename}" for idx, (s, e) in enumerate(intervals)]
                interval_data, ai_col = "; ".join(formatted), "; ".join(ai_descs)
                if not self.stop_event.is_set(): self.export_clips(path, filename, intervals)
            
            records.append({"Filename": filename, "Intervals": interval_data, "AI_Description": ai_col, "Total_Frames": total_f, "ProcTime_Sec": proc_time})
            self.save_csv(records)

        self.summary_data["total_time"] = round(time.time() - batch_start, 2)
        avg_f = round(self.summary_data["total_time"] / self.summary_data["total_frames_batch"], 4) if self.summary_data["total_frames_batch"] > 0 else 0
        use_pct = round((self.summary_data["cleared_frames"] / self.summary_data["total_frames_batch"]) * 100, 2) if self.summary_data["total_frames_batch"] > 0 else 0
        with open(self.csv_path, 'a', newline='') as f:
            f.write(f"\nTOTAL BATCH TIME,{self.summary_data['total_time']},AVG PER FRAME,{avg_f},USABILITY PCT,{use_pct}%\n")
        on_finish(self.summary_data, avg_f, use_pct)

    def save_csv(self, records):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Intervals", "AI_Description", "Total_Frames", "ProcTime_Sec"])
            writer.writeheader(); writer.writerows(records)

    def export_clips(self, path, name, intervals):
        with VideoFileClip(path) as video:
            for i, (s, e) in enumerate(intervals):
                if self.stop_event.is_set(): break
                video.subclipped(s, min(e, video.duration - 0.01)).write_videofile(os.path.join(self.output_dir, f"CLEAN_{i}_{name}"), logger=None)

class SLAMGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("SLAM Purifier v6.1")
        self.geometry("1200x1000")
        self.proc = None
        
        # --- SIDEBAR (v5.3 UI) ---
        self.sidebar = ctk.CTkFrame(self, width=300); self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="CONTROLS", font=("Arial", 20, "bold")).pack(pady=20)
        ctk.CTkLabel(self.sidebar, text="📖 USER STEPS", font=("Arial", 14, "bold"), text_color="#3498db").pack(pady=(10, 5))
        self.instr = ctk.CTkTextbox(self.sidebar, width=260, height=200, fg_color="#000000", text_color="#FFFFFF", font=("Arial", 12))
        self.instr.pack(padx=15, pady=5)
        step_text = (
            "1. SELECT FOLDER: Choose the directory containing your raw .mp4 or .mov files.\n\n"
            "2. BATCH NAME (Optional): Enter a tag (e.g., Reef_Site_A) to label your output files.\n\n"
            "3. AI SWITCH: Toggle ON to generate video descriptions. (Disabled by default)\n\n"
            "4. THRESHOLDS: Set minimum Sharpness and maximum Stability. Use Presets for quick setup.\n\n"
            "5. START: Click to run. UI will lock until batch finishes or STOP is clicked.\n\n"
            "6. OUTPUT: Check the CSV in the new timestamped folder for processed results."
        )
        self.instr.insert("0.0", step_text)
        self.mode_switch = ctk.CTkSwitch(self.sidebar, text="Dark Mode", command=self.toggle_mode); self.mode_switch.select(); self.mode_switch.pack(pady=10)
        self.ai_switch = ctk.CTkSwitch(self.sidebar, text="AI Engine (Moondream)", progress_color="#9b59b6"); self.ai_switch.deselect(); self.ai_switch.pack(pady=10)
        
        self.batch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Batch Tag", width=250); self.batch_entry.pack(pady=5)
        self.preset_menu = ctk.CTkOptionMenu(self.sidebar, values=["Manual", "Reef Survey", "Macro", "Murky Water"], command=self.apply_preset); self.preset_menu.pack(pady=5)

        # --- MAIN PANEL ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent"); self.main_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.main_frame, text="Video Preview", fg_color="black", width=640, height=360, corner_radius=8); self.preview_label.pack(pady=10)

        self.scope_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a", corner_radius=10); self.scope_frame.pack(pady=5, padx=10, fill="x")
        si = ctk.CTkFrame(self.scope_frame, fg_color="transparent"); si.pack(pady=10, padx=20)
        self.b_scope = ctk.CTkLabel(si, text="SHARPNESS: 0 / 70", font=("Arial", 12, "bold"), text_color="white"); self.b_scope.grid(row=0, column=0, padx=25)
        self.b_meter = ctk.CTkProgressBar(si, width=240, progress_color="#2ecc71"); self.b_meter.grid(row=1, column=0, padx=25, pady=5); self.b_meter.set(0)
        self.s_scope = ctk.CTkLabel(si, text="STABILITY: 0.0 / 4.0", font=("Arial", 12, "bold"), text_color="white"); self.s_scope.grid(row=0, column=1, padx=25)
        self.s_meter = ctk.CTkProgressBar(si, width=240, progress_color="#e67e22"); self.s_meter.grid(row=1, column=1, padx=25, pady=5); self.s_meter.set(0)
        self.frame_label = ctk.CTkLabel(self.scope_frame, text="Frame: 0 / 0", font=("Courier", 12, "bold")); self.frame_label.pack(pady=(0, 5))

        self.path_var = ctk.StringVar(value="No folder selected")
        self.path_label = ctk.CTkLabel(self.main_frame, text="Selected Folder: None", font=("Arial", 11), text_color="#3498db"); self.path_label.pack(pady=5)
        ctk.CTkButton(self.main_frame, text="📁 Choose Dataset Folder", command=self.pick_folder).pack()

        # DUAL-INPUTS (v5.3 UI)
        ctk.CTkLabel(self.main_frame, text="SHARPNESS THRESHOLD", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        sf = ctk.CTkFrame(self.main_frame, fg_color="transparent"); sf.pack()
        self.b_slider = ctk.CTkSlider(sf, from_=5, to=250, width=450, command=self.sync_b_s); self.b_slider.set(70); self.b_slider.pack(side="left", padx=10)
        self.b_entry = ctk.CTkEntry(sf, width=65); self.b_entry.insert(0, "70"); self.b_entry.bind("<Return>", self.sync_b_e); self.b_entry.pack(side="left")

        ctk.CTkLabel(self.main_frame, text="STABILITY THRESHOLD", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        stf = ctk.CTkFrame(self.main_frame, fg_color="transparent"); stf.pack()
        self.s_slider = ctk.CTkSlider(stf, from_=0.1, to=25, width=450, command=self.sync_s_s); self.s_slider.set(4.0); self.s_slider.pack(side="left", padx=10)
        self.s_entry = ctk.CTkEntry(stf, width=65); self.s_entry.insert(0, "4.0"); self.s_entry.bind("<Return>", self.sync_s_e); self.s_entry.pack(side="left")

        self.progress = ctk.CTkProgressBar(self.main_frame, width=550); self.progress.set(0); self.progress.pack(pady=20)
        bf = ctk.CTkFrame(self.main_frame, fg_color="transparent"); bf.pack()
        self.btn_run = ctk.CTkButton(bf, text="START PROCESS", fg_color="#2ecc71", width=200, height=40, font=("Arial", 13, "bold"), command=self.run); self.btn_run.pack(side="left", padx=10)
        self.btn_stop = ctk.CTkButton(bf, text="STOP", fg_color="#e74c3c", width=100, height=40, state="disabled", command=self.stop); self.btn_stop.pack(side="left", padx=10)
        self.log_box = ctk.CTkTextbox(self.main_frame, width=650, height=120, fg_color="#1a1a1a"); self.log_box.pack(pady=10)

    def toggle_mode(self): ctk.set_appearance_mode("dark" if self.mode_switch.get() == 1 else "light")
    def pick_folder(self): f = filedialog.askdirectory(); (self.path_var.set(f), self.path_label.configure(text=f"Selected Folder: {f}")) if f else None
    def sync_b_s(self, v): self.b_entry.delete(0, "end"); self.b_entry.insert(0, str(int(v)))
    def sync_b_e(self, e): (self.b_slider.set(float(self.b_entry.get()))) if self.b_entry.get() else None
    def sync_s_s(self, v): self.s_entry.delete(0, "end"); self.s_entry.insert(0, str(round(v, 1)))
    def sync_s_e(self, e): (self.s_slider.set(float(self.s_entry.get()))) if self.s_entry.get() else None
    def apply_preset(self, c):
        v = {"Reef Survey": (65, 3.5), "Macro": (110, 2.0), "Murky Water": (40, 5.0)}
        if c in v: self.b_slider.set(v[c][0]); self.sync_b_s(v[c][0]); self.s_slider.set(v[c][1]); self.sync_s_s(v[c][1])
    
    def stop(self): self.proc.stop_event.set() if self.proc else None
    def run(self):
        if self.path_var.get() == "No folder selected": return
        self.btn_run.configure(state="disabled"); self.btn_stop.configure(state="normal")
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            self.proc = SLAMProcessor(self.path_var.get(), self.batch_entry.get(), float(self.b_entry.get()), float(self.s_entry.get()), 
                                     self.ai_switch.get() == 1,
                                     lambda m: self.log_box.insert("end", f"> {m}\n"), self.progress.set, self.update_frame, self.update_scope)
            self.proc.process_all(self.on_finish)
        except Exception as e:
            self.log_box.insert("end", f"ERROR: {str(e)}\n"); self.on_finish(None, 0, 0)

    def on_finish(self, stats, avg, use):
        self.btn_run.configure(state="normal"); self.btn_stop.configure(state="disabled")
        if stats: messagebox.showinfo("Complete", f"CLEARED: {stats['cleared_frames']} frames\nUsability: {use}%")

    def update_frame(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
        ctk_img = ctk.CTkImage(img, size=(640, 360)); self.preview_label.configure(image=ctk_img, text=""); self.preview_label.image = ctk_img

    def update_scope(self, b, s, cur, total):
        bt, st = float(self.b_entry.get()), round(float(self.s_entry.get()), 1)
        sr = round(s, 1)
        bp, sp = b >= bt, sr <= st
        self.b_scope.configure(text=f"{'✅' if bp else '❌'} SHARPNESS: {int(b)} / {int(bt)}", text_color="#2ecc71" if bp else "#e74c3c")
        self.s_scope.configure(text=f"{'✅' if sp else '❌'} STABILITY: {f"{sr:.1f}"} / {st}", text_color="#2ecc71" if sp else "#e74c3c")
        self.b_meter.set(min(b/250, 1)); self.s_meter.set(min(sr/20, 1))
        self.frame_label.configure(text=f"Frame: {cur} / {total}")

if __name__ == "__main__": SLAMGui().mainloop()