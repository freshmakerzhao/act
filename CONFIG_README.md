# config使用说明

目前项目重构为使用YAML配置文件统一管理轨迹生成、训练和评估的配置参数。

## 配置文件管理

配置文件模板存放在 `configs/` 文件夹中，按设备型号组织，方便实验归档和管理：

```
configs/
├── fairino5_single/              # Fairino FR5单臂机器人配置
│   ├── 01_record.yaml               # 轨迹生成配置
│   ├── 02_train.yaml                # 训练配置
│   └── 03_eval.yaml                 # 评估配置
```

## 配置文件结构

配置文件包含以下主要部分：

```yaml
# 任务配置
task:
  name: sim_lifting_cube_scripted          # 任务名称
  dataset_dir: ./data_sim_episodes/...     # 数据集目录
  num_episodes: 50                         # episode数量
  episode_len: 400                         # episode长度
  camera_names: ['top']                    # 相机列表

# 设备配置
equipment:
  model: fairino5_single                   # 设备型号

# 训练配置
training:
  policy_class: ACT                        # 策略类
  batch_size: 32                           # 批次大小
  num_epochs: 2000                         # 训练轮数
  lr: 1e-5                                 # 学习率
  seed: 1000                               # 随机种子

# ACT策略参数
act:
  kl_weight: 10                            # KL散度权重
  chunk_size: 100                          # chunk大小
  hidden_dim: 512                          # 隐藏层维度
  dim_feedforward: 3200                    # 前馈网络维度

# 输出配置
output:
  ckpt_dir: ./ckpts/...                    # 检查点目录

# 评估配置
eval:
  enabled: false                           # 评估模式
  clear_videos_before_eval: true           # 评估前清除视频
  batch_size: 8                            # 评估批次大小

# 渲染配置
render:
  onscreen_render: false                   # 屏幕渲染
```

## 使用方式

所有参数只能通过YAML配置文件读取

### 1. 轨迹生成

```bash
# 使用Fairino5轨迹生成配置
python3 record_sim_episodes.py --config configs/fairino5_single/01_record.yaml
```

### 2. 训练

```bash
# 使用Fairino5训练配置
python3 imitate_episodes.py --config configs/fairino5_single/02_train.yaml
```

### 3. 评估

```bash
# 使用Fairino5评估配置
python3 imitate_episodes.py --config configs/fairino5_single/03_eval.yaml
```

## 支持的设备型号

- `vx300s_bimanual`: VX300s双臂机器人
- `vx300s_single`: VX300s单臂机器人
- `fairino5_single`: 法奥FR5单臂机器人
- `excavator_simple`: 挖掘机模型

## 支持的任务类型

- `sim_transfer_cube_scripted`: 双臂搬方块任务（脚本生成）
- `sim_transfer_cube_human`: 双臂搬方块任务（人类演示）
- `sim_insertion_scripted`: 双臂插入任务（脚本生成）
- `sim_insertion_human`: 双臂插入任务（人类演示）
- `sim_lifting_cube_scripted`: 单臂搬方块任务（脚本生成）

## 配置文件示例

### 轨迹生成配置示例 (record.yaml)
```yaml
task:
  name: sim_lifting_cube_scripted
  dataset_dir: ./data_sim_episodes/sim_lifting_cube_scripted
  num_episodes: 50
  episode_len: 400
  camera_names: ['top']

equipment:
  model: fairino5_single

render:
  onscreen_render: false
```

### 训练配置示例 (train.yaml)
```yaml
task:
  name: sim_lifting_cube_scripted
  dataset_dir: ./data_sim_episodes/sim_lifting_cube_scripted
  num_episodes: 50
  episode_len: 400
  camera_names: ['top']

equipment:
  model: fairino5_single

training:
  policy_class: ACT
  batch_size: 32
  num_epochs: 2000
  lr: 1e-5
  seed: 1000
  temporal_agg: false

act:
  kl_weight: 10
  chunk_size: 100
  hidden_dim: 512
  dim_feedforward: 3200

output:
  ckpt_dir: ./ckpts/sim_lifting_cube_scripted

render:
  onscreen_render: false
```

### 评估配置示例 (eval.yaml)
```yaml
task:
  name: sim_lifting_cube_scripted
  dataset_dir: ./data_sim_episodes/sim_lifting_cube_scripted
  num_episodes: 50
  episode_len: 400
  camera_names: ['top']

equipment:
  model: fairino5_single

training:
  policy_class: ACT
  batch_size: 8
  num_epochs: 2000
  lr: 1e-5
  seed: 1000
  temporal_agg: false

act:
  kl_weight: 10
  chunk_size: 100
  hidden_dim: 512
  dim_feedforward: 3200

output:
  ckpt_dir: ./ckpts/sim_lifting_cube_scripted

eval:
  enabled: true
  clear_videos_before_eval: true
  batch_size: 8

render:
  onscreen_render: false
```