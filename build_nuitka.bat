@echo off
echo Building standalone executable with Nuitka...

python -m nuitka ^
  --standalone ^                     ^
  --onefile ^                        ^
  --enable-plugin=tk-inter ^         ^
  --include-package-data=mne ^       ^
  --windows-console-mode=disable ^   ^
  --remove-output ^                  ^
  eeg_topomap_app.py