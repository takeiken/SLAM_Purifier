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

# --- AI & VIDEO UTILITIES ---
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip

# Note: Using a lightweight placeholder for description logic 
# to ensure the script remains portable, but hooks are ready for CLIP/BLIP.
class VisionAI:
    @staticmethod
    def describe_scene(frame):
        """
        In a full implementation, this would pass the frame to a model 
        like BLIP or CLIP. For this version, it performs a color/texture 
        heuristic to generate 'Smart Tags'.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_v = np.mean(hsv[:, :, 2])
        avg_s = np.mean(hsv[:, :, 1])
        
        if avg_v < 50: return "Low-light/Deep-water Scene"
        if avg_s > 150: return "High-saturation Tropical Reef"
        return "Standard Marine Environment"

class SLAMProcessor:
    def __init__(self, input_dir, batch_name, blur_th, shift_th, log_callback, progress_callback, frame_callback, score_callback):
        self.input_dir = input_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        raw_name = batch_name.strip()
        self.clean_batch = re.sub(r'[\\/*?:"<>|]', "_", raw_name) if raw_name else "Batch"
        
        self.output_dir = os.path.join(input_dir, f"{self.clean_batch}_{self.timestamp}")
        self.csv_path = os.path.join(self.output_dir, f"Summary_{self.clean_batch}.csv")
        
        self.blur_th = blur_th
        self.shift_th = shift_th
        self.log_fn = log_callback
        self.update_progress = progress_callback
        self.update_frame = frame_callback
        self.update_scores = score_callback
        self.stop_event = threading.Event()
        self.ai = VisionAI()
        
        self.summary_data = {
            "cleared_vids": 0, "cleared_frames": 0,
            "unclear_vids": 0, "unclear_frames": 0,
            "total_frames_batch": 0, "total_time": 0
        }
        
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def analyze_video(self, path, v_idx, v_total):
        v_start = time.time()
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        blur_scores, shift_scores, frames_cache, prev_gray = [], [], {}, None
        
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
            
            # Update UI Preview
            if f_idx % 5 == 0:
                self.update_scores(b_score, s_score, f_idx, total_f)
                self.update_frame(frame)
            if f_idx % 30 == 0:
                self.update_progress((v_idx / v_total) + ((f_idx / total_f) / v_total))
            
            # Cache frames for AI description mid-points
            frames_cache[f_idx] = frame if f_idx % 10 == 0 else None 
            prev_gray = gray

        cap.release()
        intervals = self.get_intervals(blur_scores, shift_scores, fps)
        
        # Generate AI Descriptions for each interval
        ai_desc_list = []
        for s, e in intervals:
            mid_f = int(((s + e) / 2) * fps)
            # Find closest cached frame
            closest_idx = min(frames_cache.keys(), key=lambda k: abs(k-mid_f))
            ai_desc_list.append(self.ai.describe_scene(frames_cache[closest_idx]))

        proc_time = round(time.time() - v_start, 2)
        return intervals, ai_desc_list, proc_time, total_f

    def get_intervals(self, blurs, shifts, fps):
        intervals, start = [], None
        for i, (b, s) in enumerate(zip(blurs, shifts)):
            is_good = b > self.blur_th and s < self.shift_th
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
                self.summary_data["unclear_vids"] += 1
                self.summary_data["unclear_frames"] += total_f
            else:
                self.summary_data["cleared_vids"] += 1
                self.summary_data["cleared_frames"] += total_f
                
                formatted_intervals = []
                for idx, (s, e) in enumerate(intervals):
                    exp_name = f"CLEAN_{idx}_{filename}"
                    formatted_intervals.append(f"{s}-{e}_{exp_name}")
                    self.log_fn(f"AI Tag: {ai_descs[idx]}")
                
                interval_data = "; ".join(formatted_intervals)
                ai_col = "; ".join(ai_descs)
                if not self.stop_event.is_set():
                    self.export_clips(path, filename, intervals)
            
            records.append({
                "Filename": filename, 
                "Intervals": interval_data, 
                "AI_Description": ai_col,
                "Total_Frames": total_f,
                "ProcTime_Sec": proc_time
            })
            self.save_csv(records)

        self.summary_data["total_time"] = round(time.time() - batch_start, 2)
        avg_f_time = round(self.summary_data["total_time"] / self.summary_data["total_frames_batch"], 4) if self.summary_data["total_frames_batch"] > 0 else 0
        use_pct = round((self.summary_data["cleared_frames"] / self.summary_data["total_frames_batch"]) * 100, 2) if self.summary_data["total_frames_batch"] > 0 else 0
        
        with open(self.csv_path, 'a', newline='') as f:
            f.write(f"\nTOTAL BATCH TIME,{self.summary_data['total_time']},AVG PER FRAME,{avg_f_time},USABILITY PCT,{use_pct}%\n")
            
        on_finish(self.summary_data, avg_f_time, use_pct)

    def save_csv(self, records):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Intervals", "AI_Description", "Total_Frames", "ProcTime_Sec"])
            writer.writeheader(); writer.writerows(records)

    def export_clips(self, path, name, intervals):
        with VideoFileClip(path) as video:
            for i, (s, e) in enumerate(intervals):
                if self.stop_event.is_set(): break
                video.subclipped(s, min(e, video.duration - 0.01)).write_videofile(
                    os.path.join(self.output_dir, f"CLEAN_{i}_{name}"), logger=None)

class SLAMGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("SLAM Purifier v5.3")
        self.geometry("1200(x)1000") # Fixed size
        self.proc = None
        
        # SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar, text="CONTROLS", font=("Arial", 20, "bold")).pack(pady=20)
        
        ctk.CTkLabel(self.sidebar, text="📖 USER STEPS", font=("Arial", 14, "bold"), text_color="#3498db").pack(pady=(10, 5))
        self.instr = ctk.CTkTextbox(self.sidebar, width=260, height=250, fg_color="#000000", text_color="#FFFFFF", font=("Arial", 12))
        self.instr.pack(padx=15, pady=5)
        step_text = (
            "1. SELECT FOLDER: Choose the directory containing your raw .mp4 or .mov files.\n\n"
            "2. BATCH NAME (Optional): Enter a tag (e.g., Reef_Site_A) to label your output files.\n\n"
            "3. THRESHOLDS: Set minimum Sharpness and maximum Stability. Use Presets for quick setup.\n\n"
            "4. START: Click to run. UI will lock until batch finishes or STOP is clicked.\n\n"
            "5. OUTPUT: Check the CSV in the new timestamped folder for processed results."
        )
        self.instr.insert("0.0", step_text); self.instr.configure(state="disabled")

        self.mode_switch = ctk.CTkSwitch(self.sidebar, text="Dark Mode", command=self.toggle_mode); self.mode_switch.select(); self.mode_switch.pack(pady=15)
        self.batch_entry = ctk.CTkEntry(self.sidebar, placeholder_text="e.g. Site_Alpha_Day1", width=250); self.batch_entry.pack(pady=5)
        self.preset_menu = ctk.CTkOptionMenu(self.sidebar, values=["Manual", "Reef Survey", "Macro", "Murky Water"], command=self.apply_preset); self.preset_menu.pack(pady=5)

        # MAIN PANEL
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent"); self.main_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.main_frame, text="Video Preview", fg_color="black", width=640, height=360, corner_radius=8); self.preview_label.pack(pady=10)

        # SCOPE
        self.scope_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a", corner_radius=10); self.scope_frame.pack(pady=5, padx=10, fill="x")
        scope_inner = ctk.CTkFrame(self.scope_frame, fg_color="transparent"); scope_inner.pack(pady=10, padx=20)
        self.blur_scope_label = ctk.CTkLabel(scope_inner, text="SHARPNESS: 0 / 70", font=("Arial", 12, "bold"), text_color="white"); self.blur_scope_label.grid(row=0, column=0, padx=25)
        self.blur_meter = ctk.CTkProgressBar(scope_inner, width=240, progress_color="#2ecc71"); self.blur_meter.grid(row=1, column=0, padx=25, pady=5); self.blur_meter.set(0)
        self.shift_scope_label = ctk.CTkLabel(scope_inner, text="STABILITY: 0.0 / 4.0", font=("Arial", 12, "bold"), text_color="white"); self.shift_scope_label.grid(row=0, column=1, padx=25)
        self.shift_meter = ctk.CTkProgressBar(scope_inner, width=240, progress_color="#e67e22"); self.shift_meter.grid(row=1, column=1, padx=25, pady=5); self.shift_meter.set(0)
        self.frame_label = ctk.CTkLabel(self.scope_frame, text="Frame: 0 / 0", font=("Courier", 12, "bold")); self.frame_label.pack(pady=(0, 5))

        # PATHS & INPUTS
        self.path_label = ctk.CTkLabel(self.main_frame, text="Selected Folder: None", font=("Arial", 11), text_color="#3498db"); self.path_label.pack(pady=5)
        self.path_var = ctk.StringVar(value="No folder selected")
        ctk.CTkButton(self.main_frame, text="📁 Choose Dataset Folder", command=self.pick_folder).pack()

        # INPUTS
        ctk.CTkLabel(self.main_frame, text="SHARPNESS THRESHOLD", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        sf = ctk.CTkFrame(self.main_frame, fg_color="transparent"); sf.pack()
        self.blur_slider = ctk.CTkSlider(sf, from_=5, to=250, width=450, command=self.sync_blur_slider); self.blur_slider.set(70); self.blur_slider.pack(side="left", padx=10)
        self.blur_entry = ctk.CTkEntry(sf, width=65); self.blur_entry.insert(0, "70"); self.blur_entry.bind("<Return>", self.sync_blur_entry); self.blur_entry.pack(side="left")

        ctk.CTkLabel(self.main_frame, text="STABILITY THRESHOLD", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        stf = ctk.CTkFrame(self.main_frame, fg_color="transparent"); stf.pack()
        self.shift_slider = ctk.CTkSlider(stf, from_=0.1, to=25, width=450, command=self.sync_shift_slider); self.shift_slider.set(4.0); self.shift_slider.pack(side="left", padx=10)
        self.shift_entry = ctk.CTkEntry(stf, width=65); self.shift_entry.insert(0, "4.0"); self.shift_entry.bind("<Return>", self.sync_shift_entry); self.shift_entry.pack(side="left")

        self.progress = ctk.CTkProgressBar(self.main_frame, width=550); self.progress.set(0); self.progress.pack(pady=20)
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent"); btn_frame.pack()
        self.btn_run = ctk.CTkButton(btn_frame, text="START PROCESS", fg_color="#2ecc71", width=200, height=40, font=("Arial", 13, "bold"), command=self.run); self.btn_run.pack(side="left", padx=10)
        self.btn_stop = ctk.CTkButton(btn_frame, text="STOP", fg_color="#e74c3c", width=100, height=40, state="disabled", command=self.stop_process); self.btn_stop.pack(side="left", padx=10)
        self.log_box = ctk.CTkTextbox(self.main_frame, width=650, height=120, fg_color="#1a1a1a"); self.log_box.pack(pady=10)

    def toggle_inputs(self, state):
        mode = "normal" if state else "disabled"
        for w in [self.blur_slider, self.blur_entry, self.shift_slider, self.shift_entry, self.preset_menu, self.batch_entry]: w.configure(state=mode)

    def sync_blur_slider(self, val): self.blur_entry.delete(0, "end"); self.blur_entry.insert(0, str(int(val)))
    def sync_blur_entry(self, e):
        try: self.blur_slider.set(float(self.blur_entry.get()))
        except: pass
    def sync_shift_slider(self, val): self.shift_entry.delete(0, "end"); self.shift_entry.insert(0, str(round(val, 1)))
    def sync_shift_entry(self, e):
        try: self.shift_slider.set(float(self.shift_entry.get()))
        except: pass

    def update_scope_vals(self, b, s, cur, total):
        b_target = float(self.blur_entry.get())
        s_target = float(self.shift_entry.get())
        s_rounded = round(float(s), 1)
        b_pass, s_pass = b >= b_target, s <= s_target
        self.blur_scope_label.configure(text=f"{'✅' if b_pass else '❌'} SHARPNESS: {int(b)} / {int(b_target)}", text_color="#2ecc71" if b_pass else "#e74c3c")
        self.shift_scope_label.configure(text=f"{'✅' if s_pass else '❌'} STABILITY: {s_rounded:.1f} / {s_target:.1f}", text_color="#2ecc71" if s_pass else "#e74c3c")
        self.blur_meter.set(min(b / 250, 1.0)); self.shift_meter.set(min(s / 20, 1.0)); self.frame_label.configure(text=f"Frame: {cur} / {total}")

    def toggle_mode(self): ctk.set_appearance_mode("dark" if self.mode_switch.get() == 1 else "light")
    def pick_folder(self):
        f = filedialog.askdirectory()
        if f: self.path_var.set(f); self.path_label.configure(text=f"Selected Folder: {f}")
    def apply_preset(self, choice):
        vals = {"Reef Survey": (65, 3.5), "Macro": (110, 2.0), "Murky Water": (40, 5.0)}
        if choice in vals:
            self.blur_slider.set(vals[choice][0]); self.blur_entry.delete(0, "end"); self.blur_entry.insert(0, str(vals[choice][0]))
            self.shift_slider.set(vals[choice][1]); self.shift_entry.delete(0, "end"); self.shift_entry.insert(0, str(vals[choice][1]))

    def stop_process(self):
        if self.proc: self.proc.stop_event.set(); self.log_box.insert("end", "> STOPPING BATCH...\n")

    def run(self):
        if self.path_var.get() == "No folder selected": return
        self.btn_run.configure(state="disabled"); self.btn_stop.configure(state="normal"); self.toggle_inputs(False)
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        self.proc = SLAMProcessor(self.path_var.get(), self.batch_entry.get(), float(self.blur_entry.get()), float(self.shift_entry.get()), 
                                 lambda m: self.log_box.insert("end", f"> {m}\n"), self.progress.set, self.update_frame_preview, self.update_scope_vals)
        self.proc.process_all(self.on_finish)

    def on_finish(self, stats, avg_f_time, use_pct):
        self.btn_run.configure(state="normal"); self.btn_stop.configure(state="disabled"); self.toggle_inputs(True)
        summary_table = (
            f"{'Category':<15} | {'Videos':<8} | {'Frames':<10}\n"
            f"{'-'*40}\n"
            f"{'CLEARED':<15} | {stats['cleared_vids']:<8} | {stats['cleared_frames']:<10}\n"
            f"{'UNCLEAR':<15} | {stats['unclear_vids']:<8} | {stats['unclear_frames']:<10}\n\n"
            f"Usability Percentage: {use_pct}%\n"
            f"Avg Time Per Frame: {avg_f_time}s"
        )
        messagebox.showinfo("Batch Process Summary", summary_table)

    def update_frame_preview(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
        ctk_img = ctk.CTkImage(img, size=(640, 360))
        self.preview_label.configure(image=ctk_img, text=""); self.preview_label.image = ctk_img

if __name__ == "__main__": SLAMGui().mainloop()