@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --listen --api --always-normal-vram --attention-pytorch --disable-nan-check --cuda-malloc --cuda-stream --pin-shared-memory --cors-allow-origins * --no-gradio-queue --port 7862 --unet-in-bf16
call webui.bat
