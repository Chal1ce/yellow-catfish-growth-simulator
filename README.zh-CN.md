# 运行代码

英文说明见 [README.md](README.md)。

## 1. 安装环境

在 macOS/Linux 上使用 Python 3.12 或更高版本，并确保可以通过 `cc` 命令调用 C 编译器。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. 准备输入 CSV

创建 `inputs.csv`，每行对应一个处理，包含以下必填列：

```csv
study_id,treatment_id,w0_g,days,temperature_c,ge_kj_g,protein_pct,lipid_pct,feeding
```

| 列名 | 填写内容 |
|---|---|
| `study_id` | 来源或鱼群标识 |
| `treatment_id` | 同一来源内的处理标识 |
| `w0_g` | 初始湿体重，g/尾 |
| `days` | 仿真时长，正整数天 |
| `temperature_c` | 本次运行的恒定温度，°C |
| `ge_kj_g` | 饲料总能，kJ/g |
| `protein_pct`、`lipid_pct` | 饲料蛋白质和脂肪含量，% |
| `feeding` | 以下投喂模式之一 |

| 投喂模式 | 额外输入 |
|---|---|
| `satiation` | 无 |
| `mass_updated` | `offer_fraction`：每日供料量/体重，以小数比例填写；`update_days` 默认为14天 |
| `design_schedule` | `offer_fraction`、`design_fce`：预设增重/摄食转换系数；`update_days` 默认为14天 |
| `relative_control` | `relative_ration`：相对于模拟对照摄食容量的供料比例 |
| `block_relative_control` | `relative_ration`；`update_days` 默认为14天 |

使用相对投喂模式时，同一 `study_id` 下必须包含且仅包含一行 `relative_ration=1` 的对照。

可选饲料和环境列包括 `met_pct`（默认为0.8）、`do_mg_l`（默认为6.5）、`plant_replacement` 和 `insect_replacement`（小数比例，默认为0）。

## 3. 运行仿真

```bash
python scripts/simulate.py --inputs inputs.csv --output outputs/predictions.csv
```

默认参数文件为 `models/yellow_catfish.json`。指定参数文件时运行：

```bash
python scripts/simulate.py --model models/yellow_catfish.json --inputs inputs.csv --output outputs/predictions.csv
```

如果输入中的来源标识已存在于参数文件，可使用其中保存的来源偏移：

```bash
python scripts/simulate.py --inputs inputs.csv --mode learned_context --output outputs/predictions.csv
```

输出 CSV 包含预测终重 `predicted_weight_g` 和累积摄食量 `predicted_feed_g`，均为每尾鱼的数值。

## 4. 使用已完成对照进行校准

在已完成对照行中填写 `wf_g`（最终湿体重，g/尾），或 `observed_growth_ratio`（终重/初重）。还可填写以下摄食指标之一：

- `observed_feed_g`：累积摄食量，g/尾；
- `feeding_rate_pct_bw_day`：日摄食量，占平均体重的百分比，%/天；
- `fcr_g_g`：累积摄食量/生物量增量，g/g。

同时提供多个指标时，依次优先使用累积摄食量、摄食率、FCR。其余处理行的这些结果列留空。

将 `CONTROL_ID` 替换为已完成对照的 `treatment_id`，运行：

```bash
python scripts/calibrate_source.py --inputs inputs.csv --control-id CONTROL_ID --output outputs/source_shifts.json
python scripts/simulate.py --inputs inputs.csv --calibration outputs/source_shifts.json --output outputs/calibrated_predictions.csv
```

校准文件对应对照的 `study_id`；使用该文件运行仿真时，输入应来自同一来源。

## 5. 运行 v0.46 实验

安装实验所需的额外依赖：

```bash
python -m pip install -r requirements-experiments.txt
```

将整理后的数据压缩包保存为 `yellow_catfish_data_package.zip`，导入三张输入表、已保存的拟合和预测，以及绘图数据：

```bash
python scripts/prepare_experiment_data.py --archive yellow_catfish_data_package.zip
```

使用导入的结果生成研究分数汇总、20,000 次研究 Bootstrap、全开发集预测及图 2–4 和 S1–S2：

```bash
python scripts/review_targets_v046.py stats
python scripts/finalize_review_records_v046.py
python scripts/plot_review_v046.py
```

实验结果写入 `artifacts/reviewer-revision-v0.46/`，图片和轨迹 CSV 写入 `outputs/figures_v0.46/`。

如需从输入表重新拟合所有实验，在新克隆的仓库中使用 `--data-only` 导入，然后按顺序运行：

```bash
python scripts/prepare_experiment_data.py --archive yellow_catfish_data_package.zip --data-only
python scripts/review_experiments_v046.py nested
python scripts/review_targets_v046.py baselines
python scripts/review_targets_v046.py diagnostics
python scripts/review_tabular_baselines_v046.py
python scripts/review_tabular_baselines_v046.py --tune
python scripts/review_targets_v046.py stats
python scripts/finalize_review_records_v046.py
python scripts/plot_review_v046.py
```

嵌套命令使用四个工作进程，运行四种配置的 193 个全局拟合，并复用已保存的拟合。基线命令依次运行 SGR、TGC、岭回归、XGBoost 和 TabICLv2，再在内层研究块中选择岭回归和 XGBoost 的设置。TabICLv2 首次使用时下载预训练回归权重，使用 CPU 运行。


## 6. 运行 v0.47 全局敏感性实验

按上文导入文献数据后运行：

```bash
python scripts/global_sensitivity_v047.py
python scripts/summarize_global_sensitivity_v047.py
```

输出位于 `artifacts/global-sensitivity-v0.47/`。实验使用冻结模型、两个独立 Sobol 采样重复及两个额外系数区间。原始响应保存为 NPZ，第二条命令生成汇总 CSV。
