# growth_prediction/ — 生长预测代码

论文里"体长↔体重双向预测"与"由形态反推椎数"两部分代码。

## 目录
- `allovert/` — **AlloVert** 集成模型（GRU + XGBoost + Stacking/Bagging）
  - `CNN-yugu.py` / `DNN-yugu.py` / `FT.py` / `RNN-yugu.py`（基于 GRU）/ `TLNN.py` / `XGBoost.py` / `SVR.py`
  - `Fig+AlloVert.py` — 结果出图（需先跑各模型脚本生成 7 个预测 CSV）
  - `train.xlsx` / `test.xlsx` — 示例训练/测试数据（165 条样本）
  - `data/` — 小样本数据（`体重分组训练集.xlsx` / `体重分组测试集.xlsx`，供 `Fig+AlloVert.py` 使用）
- `s3cmr/` — **S3-CMR**（Sum-Constrained Multi-target Regression）反向推断
  - `S3-CMR.py`（主脚本）/ `GP.py` / `GPR.py` / `MLP.py` / `RF.py` / `RF + Logit.py` / `SVR.py` / `XGBoost.py` / `Fig.py`
  - `体重分组训练集.xlsx` / `体重分组测试集.xlsx` — 已附在此目录

## 运行
```bash
pip install -r ../requirements.txt     # 根目录依赖

# AlloVert
cd allovert
python RNN-yugu.py                       # 训练/评估 GRU 模型，产出 best_*.pth

# S3-CMR
cd ../s3cmr
python S3-CMR.py                         # 输出 report.txt 与各模型预测结果
```

## 说明
- 各脚本读取同目录下的 `.xlsx`（相对路径），clone 后直接可跑示例数据。
- 训练好的 `best_*.pth` 不会随仓库提供，请运行对应脚本自行生成。
- `Fig+AlloVert.py` 依赖 7 个模型预测 CSV（`RNN_yugu_predictions.csv` 等），
  需先运行 AlloVert 各模型脚本，或把 CSV 放进 `allovert/data/`。
