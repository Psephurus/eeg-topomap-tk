import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import numpy as np
import mne

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

DEFAULT_FONT = ("Microsoft YaHei", 10)

class EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("脑电PSD拓扑图绘制工具")
        self.root.geometry("1500x500")
        
        icon_path = "icon.png" 
        if os.path.exists(icon_path):
            img = ttk.PhotoImage(file=icon_path)
            self.root.iconphoto(False, img)

        # --- 变量初始化 ---
        self.file_path = tk.StringVar()
        self.status_text = tk.StringVar(value="请选择一个BDF文件开始...")
        self.cmap_name = tk.StringVar(value='RdBu_r')
        self.current_fig = None
        
        # --- Matplotlib 颜色图列表 ---
        self.colormaps = [
            'jet', 'RdBu_r', 'coolwarm', 'seismic',             # 常用/发散 (Diverging)
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', # 感知均匀 (Perceptually Uniform)
            'Reds', 'Blues', 'Greens'                           # 顺序 (Sequential)
        ]

        # --- 界面布局 ---
        self._setup_ui()

    def _setup_ui(self):
        # 1. 顶部控制栏
        control_group = ttk.Labelframe(self.root, text=" 参数设置 & 操作 ", padding=15, bootstyle="primary")
        control_group.pack(side=TOP, fill=X, padx=20, pady=15)
        
        # 第1行：文件选择
        file_frame = tk.Frame(control_group)
        file_frame.pack(fill=X, pady=5)

        # 文件路径显示框
        ttk.Label(file_frame, text="数据文件 (.bdf):", font=DEFAULT_FONT).pack(side=LEFT, padx=5)
        entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50, font=DEFAULT_FONT)
        entry.pack(side=LEFT, padx=5, fill=X, expand=True)

        # 浏览按钮
        btn_browse = ttk.Button(file_frame, text="📂 浏览文件", command=self.browse_file, bootstyle="secondary-outline")
        btn_browse.pack(side=LEFT, padx=5)

        
        # 第二行：配色与操作
        action_frame = ttk.Frame(control_group)
        action_frame.pack(fill=X, pady=(15, 5))
        
        # 配色选择
        ttk.Label(action_frame, text="配色方案:", font=DEFAULT_FONT).pack(side=LEFT, padx=5)
        
        # 使用 Combobox 替代 OptionMenu，更现代
        cmap_combo = ttk.Combobox(action_frame, textvariable=self.cmap_name, values=self.colormaps, state="readonly", width=12, font=DEFAULT_FONT)
        cmap_combo.pack(side=LEFT, padx=5)
        # cmap_combo.current(1) # 默认选中 RdBu_r

        # 占位，把按钮推到右边
        ttk.Label(action_frame, text="").pack(side=LEFT, expand=True)

        # 运行按钮 (实心主色 Primary)
        self.btn_run = ttk.Button(action_frame, text="▶ 开始绘制", command=self.start_processing_thread, bootstyle="primary", width=15)
        self.btn_run.pack(side=LEFT, padx=5)

        # 保存按钮 (成功色 Success)
        self.btn_save = ttk.Button(action_frame, text="💾 保存结果", command=self.save_figure, state=DISABLED, bootstyle="success", width=15)
        self.btn_save.pack(side=LEFT, padx=5)

        # === 2. 绘图区域 ===
        # 使用 Frame 包裹 Canvas，增加边框效果
        self.plot_container = ttk.Frame(self.root, padding=2)
        self.plot_container.pack(side=TOP, fill=BOTH, expand=True, padx=20, pady=5)
        
        # 初始背景图或文字
        self.placeholder_label = ttk.Label(
            self.plot_container, 
            text="⬇ 请在上方选择文件并运行", 
            font=("Microsoft YaHei", 14, "bold"), 
            foreground="#aaaaaa"
        )
        self.placeholder_label.pack(expand=True)

        # === 3. 底部状态栏 ===
        # 使用 Meter 或 Progressbar 也可以，这里用简单的带颜色 Label
        status_frame = ttk.Frame(self.root, bootstyle="light")
        status_frame.pack(side=BOTTOM, fill=X)
        
        self.lbl_status = ttk.Label(
            status_frame, 
            textvariable=self.status_text, 
            font=("Microsoft YaHei", 9),
            padding=5,
            bootstyle="inverse-light" # 反转色，深色背景白字（或浅色背景深字，取决于主题）
        )
        self.lbl_status.pack(side=LEFT, fill=X)


    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("BDF files", "*.bdf"), ("All files", "*.*")])
        if filename:
            self.file_path.set(filename)
            self.status_text.set(f"已就绪: {os.path.basename(filename)}")

    def detect_empty_channels(self, raw, eeg_picks, threshold=1e-10):
        """检测空通道"""
        data, _ = raw[eeg_picks]
        stds = np.std(data, axis=1)
        empty_chs = [
            ch_name
            for _, (ch_name, s) in enumerate(zip(eeg_picks, stds)) if s < threshold
        ]
        return empty_chs

    def start_processing_thread(self):
        """启动后台线程处理数据，防止界面卡死"""
        path = self.file_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("错误", "文件路径无效！")
            return

        self.btn_run.config(state=DISABLED)
        self.btn_save.config(state=DISABLED)
        self.status_text.set("⏳ 正在读取数据并计算，请稍候...")
        
        # 使用线程运行耗时操作
        thread = threading.Thread(target=self.process_data, args=(path,))
        thread.daemon = True
        thread.start()

    def process_data(self, bdf_path):
        try:
            # --- 1. 读取文件 ---
            self.update_status(f"⏳ 读取文件: {os.path.basename(bdf_path)} ...")
            raw = mne.io.read_raw_bdf(bdf_path, preload=True, stim_channel="auto", verbose=False)

            # --- 2. 筛选通道与坏道检测 ---
            eeg_ch_names = raw.copy().pick('eeg', exclude=[]).ch_names
            auto_empty = self.detect_empty_channels(raw, eeg_ch_names)

            if auto_empty:
                print(f"检测到疑似空通道: {auto_empty}")
                raw.info["bads"].extend(auto_empty)

            raw.pick(picks='eeg', exclude='bads')

            # --- 3. 设置蒙太奇 ---
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="ignore")

            # --- 4. 计算 PSD ---
            self.update_status("⏳ 正在计算 PSD (Welch)...")
            spectrum = raw.compute_psd(
                method="welch",
                fmin=1,
                fmax=40.,
                picks="eeg",
                reject_by_annotation=True,
                verbose=False
            )

            # --- 5. 绘图 ---
            self.update_status("⏳ 正在生成拓扑图...")
            bands = {
                "δ (0-4 Hz)": (0, 4),
                "θ (4–8 Hz)": (4, 8),
                "α (8–12 Hz)": (8, 12),
                "β (12–30 Hz)": (12, 30),
                "γ (30-35 Hz)": (30, 35),
            }

            selected_cmap = self.cmap_name.get()

            with plt.style.context('fast'):
                fig = spectrum.plot_topomap(
                    bands=bands,
                    ch_type="eeg",
                    normalize=False,
                    dB=True,
                    show=False,
                    cmap=selected_cmap
            )
            
            # 处理完成，回调主线程更新 UI
            self.root.after(0, self.display_result, fig)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def display_result(self, fig):
        """在 GUI 中显示 Matplotlib 图像"""
        # 清除旧图像
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        self.current_fig = fig
        
        # fig.patch.set_facecolor('white')

        # 创建 Canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        self.status_text.set("✔ 绘制完成！")
        self.btn_run.config(state=NORMAL)
        self.btn_save.config(state=NORMAL)

    def save_figure(self):
        """手动保存图片"""
        if self.current_fig is None:
            return
        
        initial_name = os.path.splitext(os.path.basename(self.file_path.get()))[0] + "-topomap.png"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        
        if file_path:
            self.current_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("成功", f"图片已保存至: {file_path}")

    def update_status(self, text):
        """线程安全的更新状态栏"""
        self.root.after(0, self.status_text.set, text)

    def show_error(self, error_msg):
        """线程安全的报错"""
        messagebox.showerror("处理出错", f"发生错误:\n{error_msg}")
        self.status_text.set("出错")
        self.btn_run.config(state=tk.NORMAL)

if __name__ == "__main__":
    app_window = ttk.Window(themename="cosmo")
    app = EEGApp(app_window)
    app_window.mainloop()