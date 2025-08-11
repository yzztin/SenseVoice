HOST=0.0.0.0
PORT=12345
DEVICE_ID=2  # npu 卡号
SENSEVOICE_MODEL_DIR="/root/.cache/modelscope/hub/models/iic/SenseVoiceSmall"  # 模型文件路径

export SENSEVOICE_DEVICE="npu:$DEVICE_ID"
export SENSEVOICE_MODEL_DIR=$SENSEVOICE_MODEL_DIR
export MODELSCOPE_SKIP_MD5_CHECK=True

python -m uvicorn api:app --host $HOST --port $PORT