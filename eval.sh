cd /inspire/qb-ilm/project/traffic-congestion-management/zhangqiaosheng-24045/SafeMoE
source .venv/bin/activate

export EVAL_CACHE_ROOT=/inspire/qb-ilm/project/traffic-congestion-management/zhangqiaosheng-24045/SafeMoE/hf_eval_cache
export HF_HOME=$EVAL_CACHE_ROOT
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ckpt=/inspire/qb-ilm/project/traffic-congestion-management/zhangqiaosheng-24045/SafeMoE/checkpoints/safemoe-qwen-warmup-wmdp-harmful-bootstrap/step-00006900/ablated
litgpt evaluate "$ckpt" --batch_size 4 --out_dir "$ckpt/eval" --tasks "wmdp,mmlu"
