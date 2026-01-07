## Nuitka 打包配置指令
# nuitka-project: --standalone
# nuitka-project: --onefile
# nuitka-project: --enable-plugin=tk-inter
# nuitka-project: --include-package-data=mne
# nuitka-project: --windows-console-mode=disable
# nuitka-project: --remove-output
# Nuitka 基础信息
# nuitka-project: --windows-icon-from-ico=assets/app.ico
# nuitka-project: --windows-file-version=1.2.0.0
# nuitka-project: --windows-product-version=1.1
# nuitka-project: --windows-file-description="脑电地形图分析工具"
# nuitka-project: --windows-product-name="EEG Topomap App"
# nuitka-project: --windows-company-name="Personal Research Project"
# nuitka-project: --copyright="Copyright (c) 2026 Yingxin Gao"
# ====================================
# Nuitka/打包 兼容性补丁
# ====================================
import mne.utils.misc

# 绕过MNE分析源代码 _auto_weakref
def _bypass_auto_weakref(func):
    return func

mne.utils.misc._auto_weakref = _bypass_auto_weakref
# ==========================================

import os
import tkinter as tk
from threading import Thread
from tkinter import filedialog, messagebox

import ttkbootstrap as ttk
from matplotlib import use
from ttkbootstrap.constants import TOP, BOTTOM, LEFT, X, BOTH, DISABLED, NORMAL

use("TkAgg")    # matplotlib.use
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from eeg_topomap import EEGProcessor

DEFAULT_FONT = ("Microsoft YaHei", 10)


class EEGApp:

    def __init__(self, root):
        self.root = root
        self.root.title("脑电PSD拓扑图绘制工具")
        self.root.geometry("1500x500")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # 接管窗口关闭事件
        
        icon_path = "icon.png"
        if os.path.exists(icon_path):
            img = ttk.PhotoImage(file=icon_path)
            self.root.iconphoto(False, img)

        # --- 变量初始化 ---
        self.file_path = tk.StringVar()
        self.status_text = tk.StringVar(value="请选择一个BDF文件开始...")
        self.cmap_name = tk.StringVar(value="RdBu_r")
        self.current_fig = None

        # Matplotlib 颜色图列表
        self.colormaps = [
            "RdBu_r",
            "jet",
            "coolwarm",
            "seismic",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Reds",
            "Blues",
            "Greens",
        ]

        self._setup_ui()
 
        
    def on_closing(self):
        """
        当用户点击窗口右上角 X 关闭时触发。
        执行清理并强制结束进程。
        """
        try:
            # 关闭 Matplotlib 的所有图形
            import matplotlib.pyplot as plt
            plt.close('all') 
            
            # 停止 Tkinter 主循环
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        finally:
            # 强制终止当前进程
            os._exit(0)


    def _setup_ui(self):
        """搭建 GUI 界面"""
        # 顶部控制栏
        control_group = ttk.Labelframe(
            self.root, text=" 参数设置 & 操作 ", padding=15, bootstyle="primary"
        )
        control_group.pack(side=TOP, fill=X, padx=20, pady=15)

        # 第1行：文件选择
        file_frame = tk.Frame(control_group)
        file_frame.pack(fill=X, pady=5)

        ttk.Label(file_frame, text="数据文件 (.bdf):", font=DEFAULT_FONT).pack(
            side=LEFT, padx=5
        )
        entry = ttk.Entry(
            file_frame, textvariable=self.file_path, width=50, font=DEFAULT_FONT
        )
        entry.pack(side=LEFT, padx=5, fill=X, expand=True)

        btn_browse = ttk.Button(
            file_frame,
            text="📂 浏览文件",
            command=self.browse_file,
            bootstyle="secondary-outline",
        )
        btn_browse.pack(side=LEFT, padx=5)

        # 第2行：配色与操作
        action_frame = ttk.Frame(control_group)
        action_frame.pack(fill=X, pady=(15, 5))

        ttk.Label(action_frame, text="配色方案:", font=DEFAULT_FONT).pack(
            side=LEFT, padx=5
        )

        cmap_combo = ttk.Combobox(
            action_frame,
            textvariable=self.cmap_name,
            values=self.colormaps,
            state="readonly",
            width=12,
            font=DEFAULT_FONT,
        )
        cmap_combo.pack(side=LEFT, padx=5)

        ttk.Label(action_frame, text="").pack(side=LEFT, expand=True)

        self.btn_run = ttk.Button(
            action_frame,
            text="▶ 开始绘制",
            command=self.start_processing_thread,
            bootstyle="primary",
            width=15,
        )
        self.btn_run.pack(side=LEFT, padx=5)

        self.btn_save = ttk.Button(
            action_frame,
            text="💾 保存结果",
            command=self.save_figure,
            state=DISABLED,
            bootstyle="success",
            width=15,
        )
        self.btn_save.pack(side=LEFT, padx=5)

        # 绘图区域
        self.plot_container = ttk.Frame(self.root, padding=2)
        self.plot_container.pack(side=TOP, fill=BOTH, expand=True, padx=20, pady=5)

        self.placeholder_label = ttk.Label(
            self.plot_container,
            text="⬇ 请在上方选择文件并运行",
            font=("Microsoft YaHei", 14, "bold"),
            foreground="#aaaaaa",
        )
        self.placeholder_label.pack(expand=True)

        # 底部状态栏
        status_frame = ttk.Frame(self.root, bootstyle="light")
        status_frame.pack(side=BOTTOM, fill=X)

        self.lbl_status = ttk.Label(
            status_frame,
            textvariable=self.status_text,
            font=("Microsoft YaHei", 9),
            padding=5,
            bootstyle="inverse-light",
        )
        self.lbl_status.pack(side=LEFT, fill=X)


    def browse_file(self):
        """打开文件对话框选择BDF文件"""
        filename = filedialog.askopenfilename(
            filetypes=[("BioSig文件", "*.bdf"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            self.status_text.set(f"已就绪: {os.path.basename(filename)}")

    def start_processing_thread(self):
        """启动后台线程处理数据，防止界面卡死"""
        path = self.file_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("错误", "文件路径无效！")
            return

        self.btn_run.config(state=DISABLED)
        self.btn_save.config(state=DISABLED)
        self.status_text.set("⏳ 正在读取数据并计算，请稍候...")

        thread = Thread(target=self.process_data, args=(path,))
        thread.daemon = True
        thread.start()

    def process_data(self, bdf_path: str):
        """
        在线程中调用核心计算模块，完成拓扑图计算。
        """
        try:
            processor = EEGProcessor(
                cmap=self.cmap_name.get(),
                status_callback=self.update_status,  # 将 GUI 的状态更新函数传进去
            )
            spectrum = processor.compute_psd_data(bdf_path)
            self.root.after(0, self.display_result, spectrum, processor)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))


    def display_result(self, spectrum, processor):
        """在 GUI 中显示 Matplotlib 图像"""
        fig = processor.plot_topomap_figure(spectrum)

        # 清理旧的 widget
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        self.current_fig = fig

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

        initial_name = (
            os.path.splitext(os.path.basename(self.file_path.get()))[0]
            + "-topomap.png"
        )
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[
                ("PNG图像", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG图像", "*.svg"),
            ],
        )

        if file_path:
            self.current_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("成功", f"图片已保存至: {file_path}")


    def update_status(self, text: str):
        """线程安全的更新状态栏（供核心模块回调使用）"""
        self.root.after(0, self.status_text.set, text)

    def show_error(self, error_msg: str):
        """线程安全的报错"""
        messagebox.showerror("处理出错", f"发生错误:\n{error_msg}")
        self.status_text.set("出错")
        self.btn_run.config(state=tk.NORMAL)


if __name__ == "__main__":
    app_window = ttk.Window(themename="cosmo")
    app = EEGApp(app_window)
    app_window.mainloop()
