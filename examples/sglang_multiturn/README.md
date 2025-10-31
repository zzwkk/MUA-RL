# Multi-Turn Rollout Example

## 🚀 Quick Start - Run MUA Code

### 1. Installation

#### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+

#### Quick Install

```bash
# Clone the repository
git clone https://github.com/zzwkk/MUA-RL.git
cd MUA-RL

# Install dependencies
pip install -e .
pip install -r requirements_sglang.txt
pip install transformers==4.51.1
```

### 2. Configure Training

Edit the training script parameters:

```bash
# Edit model path and other parameters in the script
vim examples/sglang_multiturn/mua_32b.sh
```

#### Key Parameters to Modify:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MODEL_PATH` | Path to your base model | `/path/to/your/model` |
| `N_NODE` | Number of nodes for distributed training | `4` |
| `BATCH_SIZE` | Training batch size | `32` |
| `EPOCH_NUM` | Number of training epochs | `30` |
| `API_KEY` | OpenAI API key for evaluation model | `sk-...` |
| `BASE_URL` | OpenAI API base URL | `https://api.openai.com/v1` |
| `CKPT_DIR` | Directory path to save model checkpoints | `/path/to/checkpoints` |
| `TENSORBOARD_DIR` | Directory path for TensorBoard logs | `/path/to/tensorboard` |
| `ROLLOUT_LOG_PATH` | Directory path for rollout generation logs | `/path/to/rollout_logs` |
| `VALID_LOG_PATH` | Directory path for validation logs | `/path/to/validation_logs` |

### 3. Run Training

#### Multi-Node Training (4 * 8 GPUs)

```bash
# For 4*8 GPU setup, suggest H200 141GB
bash examples/sglang_multiturn/mua_32b.sh
```

### 4. Convert Checkpoints to Hugging Face Format

After training, convert distributed checkpoints to Hugging Face format:

```bash
# Edit the merge script configuration
vim scripts/merge.sh

# Set your model path and name
BASE_DIR="/path/to/your/checkpoints/"
MODEL_NAME="your_model_name"

# Run the conversion
bash scripts/merge.sh
```

The script will automatically:
- 🔍 Find all `global_step_*` directories
- 🔄 Convert FSDP/Megatron checkpoints to Hugging Face format
- 💾 Save merged models to `iter_XXXXXX/actor/unify_checkpoint/`

