from mne.io import read_raw_bdf
from mne.channels import make_standard_montage


class EEGProcessor:
    """
    负责脑电数据的核心计算逻辑：
    - 读取 BDF
    - 检测疑似空通道
    - 设置蒙太奇
    - 计算 PSD
    - 生成拓扑图 Figure

    参数
    ----
    cmap : str
        matplotlib colormap 名称，用于绘图。
    status_callback : Callable[[str], None] or None
        用于汇报进度的回调函数（例如 GUI 的状态栏），
        如果不需要显示进度，可以不传。
    """

    def __init__(self, cmap: str = "RdBu_r", status_callback=None):
        self.cmap = cmap
        self.status_callback = status_callback or (lambda msg: None)

    def _update_status(self, text: str):
        """内部使用的状态更新函数"""
        self.status_callback(text)

    def compute_psd_data(self, bdf_path: str):  # 重命名函数，只计算和返回PSD数据
        """
        读取指定 BDF 文件并计算 PSD。

        Parameters
        ----------
        bdf_path : str
            BDF 文件路径。

        Returns
        -------
        mne.time_frequency.Spectrum
            计算得到的 Spectrum 对象。
        """

        # 读取文件
        self._update_status(f"⏳ 读取文件...")
        raw = read_raw_bdf(
            bdf_path,
            preload=True,
            stim_channel="auto",
            verbose=False
        )

        # 设置蒙太奇
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # 计算初始 PSD 用于检测坏通道
        self._update_status("⏳ 正在计算PSD...")
        spectrum = raw.compute_psd()

        psd_data = spectrum.get_data()

        bad_from_psd = [
            raw.ch_names[i]
            for i, ch_data in enumerate(psd_data)
            if max(ch_data) <= 0
        ]

        if bad_from_psd:
            print(f"MNE频谱分析检测到空通道: {bad_from_psd}")
            raw.info['bads'].extend(bad_from_psd)

        raw.pick(picks='eeg', exclude='bads')

        # 重新计算最终 PSD
        spectrum = raw.compute_psd(
            method="welch",
            fmin=1,
            fmax=40.0,
            picks="eeg",
            reject_by_annotation=True,
            verbose=False,
        )

        return spectrum  # 返回 Spectrum 对象
        # 移除绘图逻辑

    # 新增一个绘图函数，可以在主线程中调用
    def plot_topomap_figure(self, spectrum):
        """
        根据 Spectrum 对象生成 PSD 拓扑图 Figure。

        Parameters
        ----------
        spectrum : mne.time_frequency.Spectrum
            MNE Spectrum 对象。

        Returns
        -------
        matplotlib.figure.Figure
            包含多个频段拓扑图的 Figure 对象。
        """
        self._update_status("⏳ 正在生成拓扑图...")
        bands = {
            "δ (0-4 Hz)": (0, 4),
            "θ (4–8 Hz)": (4, 8),
            "α (8–12 Hz)": (8, 12),
            "β (12–30 Hz)": (12, 30),
            "γ (30-35 Hz)": (30, 35),
        }

        fig = spectrum.plot_topomap(
            bands=bands,
            ch_type="eeg",
            normalize=False,
            dB=True,
            show=False,
            cmap=self.cmap,
        )
        return fig
