@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --no-gradio-queue --listen --api --xformers --always-normal-vram --disable-nan-check --cuda-malloc --cuda-stream --pin-shared-memory --cors-allow-origins * --reinstall-xformers --reinstall-torch
call webui.bat
