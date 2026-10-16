CONFIG_PATH=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -config | --config )
            CONFIG_PATH="$2"
            shift 2
            ;;
    esac
done

# Determine config file path
CONFIG_FILE="${CONFIG_PATH:-config/chat_local_cosyvoice3_vllm_flashhead_worker.yaml}"

# Check for AvatarMusetalk module and set environment variables（if Musetalk is used, set PYTORCH_JIT=0）
if grep -q "AvatarMusetalk:" "$CONFIG_FILE" 2>/dev/null && grep -q "module: avatar/musetalk/avatar_handler_musetalk" "$CONFIG_FILE" 2>/dev/null; then
    echo "Detected AvatarMusetalk module, adding PYTORCH_JIT=0 environment variable"
    ENV_VARS="-e PYTORCH_JIT=0"
else
    echo "No AvatarMusetalk module detected in config, skipping PYTORCH_JIT environment variable"
fi

docker run -d \
    --restart unless-stopped \
    --gpus "device=1" \
    --name open-avatar-chat \
    --network=host \
    --shm-size=256m \
    -e USE_TF=0 -e USE_FLAX=0 -e TORCHINDUCTOR_COMPILE_THREADS=4 \
    $ENV_VARS \
    -v `pwd`/build:/root/open-avatar-chat/build \
    -v `pwd`/models:/root/open-avatar-chat/models \
    -v `pwd`/ssl_certs:/root/open-avatar-chat/ssl_certs \
    -v `pwd`/config:/root/open-avatar-chat/config \
    -v `pwd`/.env:/root/open-avatar-chat/.env \
    -v `pwd`/models/musetalk/s3fd-619a316812/:/root/.cache/torch/hub/checkpoints/ \
    -v `pwd`/exp:/root/open-avatar-chat/exp \
    -v `pwd`/resource:/root/open-avatar-chat/resource \
    -v `pwd`/src/handlers/tts/cosyvoice/cosyvoice_processor.py:/root/open-avatar-chat/src/handlers/tts/cosyvoice/cosyvoice_processor.py:ro \
    -e NO_PROXY=127.0.0.1,localhost \
    -e no_proxy=127.0.0.1,localhost \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    open-avatar-chat:cv3-worker-metrics-20261016 \
    --config ${CONFIG_PATH:-config/chat_local_cosyvoice3_vllm_flashhead_worker.yaml}
