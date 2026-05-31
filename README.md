# RALLRec：基于检索增强的 LLM 推荐系统

利用 RAG（检索增强生成）技术，通过检索用户-物品交互模式和物品语义描述，融合到 prompt 模板中，提升 LLM 推荐效果。

## 目录结构

```
RALLRec/
├── finetune.py                  # LoRA 微调脚本
├── inference.py                 # 推理 / 评估脚本
├── train_lightgcn.py            # LightGCN 协同嵌入训练
├── .gitignore                   # 排除大文件（数据、模型权重等）
│
├── data_preprocess/             # 数据下载与预处理
│   ├── download_ml-1m.sh        # 下载 MovieLens-1M
│   ├── download_book.sh         # 下载 BookCrossing
│   ├── download_amazon.sh       # 下载 Amazon Movies
│   ├── ml-1m.ipynb / ml-1m.py   # ML-1M 预处理
│   ├── BookCrossing.ipynb
│   └── amazon-movie.ipynb
│
├── rag/                         # 检索增强生成模块
│   ├── get_text_description.py  # 获取物品文本描述
│   ├── get_semantic_embed.py    # 生成语义嵌入
│   ├── get_semantic_embed_rich.py
│   └── topK_relevant_ml1m.py    # 检索 top-K 相似物品 / 用户
│
├── prompts/                     # Prompt 模板
│   ├── load_prompt_ml1m.py      # ML-1M 模板（含全部 temp_type）
│   ├── load_prompt_BookCrossing.py
│   └── load_prompt_amazon.py
│
├── utils/                       # 工具脚本
│   ├── data2json.py             # 预处理数据转为训练 JSON
│   ├── json_items_check.py      # 校验生成的 JSON 数据
│   └── training_set_construction.py
│
├── data/                        #（生成）预处理后的数据集
├── embeddings/                  #（生成）语义 & 协同嵌入向量
├── lora_llama/                  #（生成）LoRA 微调权重
└── logs/                        #（生成）评估结果日志
```

## 环境依赖

| 包名 | 用途 |
|------|------|
| `torch >= 2.0` | 深度学习框架 |
| `transformers >= 4.40` | LLM 加载（Llama-3.1-8B-Instruct） |
| `peft >= 0.10` | LoRA 微调 |
| `bitsandbytes >= 0.43` | 4-bit 量化（QLoRA） |
| `datasets >= 2.0` | HuggingFace 数据集加载 |
| `accelerate >= 0.30` | 分布式训练 |
| `scikit-learn` | AUC / LogLoss / ACC 指标 |
| `numpy`, `pandas` | 数据处理 |
| `h5py` | HDF5 格式数据存取 |
| `sentencepiece` | LLaMA 分词器 |
| `tabulate` | 结果格式化 |

一键安装：
```bash
pip install torch transformers peft bitsandbytes datasets accelerate \
    scikit-learn numpy pandas h5py sentencepiece tabulate
```

## 参数说明

### 微调 (`finetune.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `../Llama-3.1-8B-Instruct` | 基座 LLM 路径 |
| `--dataset` | `ml-1m` | 数据集：`ml-1m` / `BookCrossing` / `ml-25m` |
| `--K` | `30` | 历史交互序列长度 |
| `--temp_type` | `high` | 模板类型（见下表） |
| `--emb_type` | `text` | 嵌入类型：`text` / `colla` / `mix` |
| `--train_type` | `high` | 训练数据划分 |
| `--train_size` | `1024` | 训练样本数 |
| `--epochs` | `5` | 训练轮数 |
| `--lr` | `5e-4` | 学习率 |
| `--total_batch_size` | `256` | 全局批次大小 |
| `--use_lora` | `1` | 是否使用 LoRA（0 = 全量微调） |
| `--output_path` | `lora_llama` | LoRA 权重输出目录 |

### 推理 (`inference.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `../Llama-3.1-8B-Instruct` | 基座 LLM 路径 |
| `--resume_from_checkpoint` | 自动推导 | LoRA adapter 路径 |
| `--dataset` | `ml-1m` | 数据集 |
| `--K` | `30` | 历史交互序列长度 |
| `--temp_type` | `high` | 模板类型 |
| `--emb_type` | `text` | 嵌入类型 |
| `--train_type` | `mixed` | 训练数据划分 |
| `--sim_user` | `False` | 是否加入相似用户检索 |

## 模板类型 (`--temp_type`)

| 类型 | 说明 |
|------|------|
| `simple` | 历史物品按时间顺序排列，无检索 |
| `sequential` | 历史物品含检索信息，保持原始顺序 |
| `high` | 历史物品按与目标物品相似度降序排列 |
| `fusion_sem_time` | 三通道：语义 + 时间 + 协同 |
| `fusion_2ch` | 双通道：语义 + 协同 |
| `fusion_3ch` | 三通道：语义 + 协同 + 时间 |

## 使用流程

### 1. 下载并预处理数据

```bash
cd data_preprocess
bash download_ml-1m.sh          # 下载到 data/ml-1m/
python ml-1m.py                  # 预处理原始数据
```

### 2. 生成嵌入向量

```bash
# 语义嵌入（基于物品文本描述）
python rag/get_text_description.py --dataset ml-1m
python rag/get_semantic_embed.py --dataset ml-1m

# 协同嵌入（LightGCN）
python train_lightgcn.py --dataset ml-1m
```

### 3. 构建训练数据

```bash
# 转为训练 JSON 格式
python utils/data2json.py --dataset ml-1m --K 30 --temp_type fusion_2ch --emb_type text
```

### 4. 微调

```bash
# 采样训练数据（可选，用于快速实验）
cd data/ml-1m/proc_data/data/train
python -c "
import json, random
random.seed(42)
data = json.load(open('train_30_fusion_2ch_text.json'))
sample = random.sample(data, min(2048, len(data)))
json.dump(sample, open('train_30_high_text_sampled.json', 'w'))
"

# 复制测试数据
cd ../test
cp test_30_fusion_2ch_text.json test_30_high_text.json

# 开始微调
cd /path/to/RALLRec
python finetune.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --dataset ml-1m \
    --train_type high \
    --emb_type text \
    --K 30 \
    --temp_type fusion_2ch \
    --train_size 2048 \
    --epochs 7 \
    --lr 5e-4
```

### 5. 推理评估

```bash
python inference.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --dataset ml-1m \
    --K 30 \
    --temp_type fusion_2ch \
    --emb_type text \
    --train_type high
```

输出 AUC、ACC、LogLoss 三项指标，详细结果保存在 `logs/new_results/` 目录下。

## 注意事项

- 大文件目录（`data/`、`lora_llama/`、`embeddings/`、`logs/`）已通过 `.gitignore` 排除，不会上传到仓库。在新机器上按流程重新生成即可。
- LoRA 权重保存路径：`lora_llama/{模型名}_{train_type}_{emb_type}_new/`
- 评估结果日志：`logs/new_results/{数据集}_{K}_{temp_type}_{emb_type}.json`
- 基座模型：[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)，需 HuggingFace 登录授权才能下载。
