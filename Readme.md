# EEG绘制Topomap

读取BDF文件，绘制PSD拓扑图的简单Tkinter程序。

## 打包

```shell
python -m nuitka --standalone --onefile --enable-plugin=tk-inter --enable-plugin=matplotlib --windows-console-mode=disable eeg_topomap.py
```
