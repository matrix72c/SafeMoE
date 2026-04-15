cd /inspire/qb-ilm/project/traffic-congestion-management/zhangqiaosheng-24045/SafeMoE
source .venv/bin/activate

export EVAL_CACHE_ROOT=/inspire/qb-ilm/project/traffic-congestion-management/zhangqiaosheng-24045/SafeMoE/hf_eval_cache
export HF_HOME=$EVAL_CACHE_ROOT
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

# 可选：提升下载限速/额度（没有也能下）
# export HF_TOKEN=你的token

python - <<'PY'
from datasets import get_dataset_config_names, load_dataset

def try_split(repo, cfg, split):
    try:
        load_dataset(repo, cfg, split=split)
        print(f"cached: {repo}/{cfg}:{split}")
        return True
    except Exception as e:
        print(f"skip : {repo}/{cfg}:{split} -> {type(e).__name__}: {e}")
        return False

# wmdp
for cfg in get_dataset_config_names("cais/wmdp"):
    if not try_split("cais/wmdp", cfg, "test"):
        try_split("cais/wmdp", cfg, "train")

# mmlu
for cfg in get_dataset_config_names("cais/mmlu"):
    if cfg == "all":
        continue
    ok = try_split("cais/mmlu", cfg, "dev")
    ok = try_split("cais/mmlu", cfg, "test") or ok
    if not ok:
        try_split("cais/mmlu", cfg, "train")
PY
