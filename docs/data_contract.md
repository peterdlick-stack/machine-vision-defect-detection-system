# 输入、输出与标定数据

## 图片与序列

支持 PNG/JPEG/BMP/TIFF，内部 BGR uint8。单边至少8像素；16位工业原始数据需在适配器按标定范围映射为uint8，不能直接无损假设。文件夹仅读取本层图像、按名称排序，排除 `_nir` 后缀。应使用固定宽度编号，例如 `frame_000.png`。不同工件不勾选同一目标序列。

可选近红外配对：`frame_000_nir.png`，必须与可见光几何配准、同尺寸。不提供时走单谱降噪并记录 `missing-single-spectrum`。图像文件不存在传感数据时使用配置的模拟值；不会根据普通图像亮度自动伪造物理反光率。

可选旁车文件 `frame_000.json`：

```json
{
  "interference": {
    "light_lux": 6000,
    "temperature_c": 32,
    "reflectance": 0.5,
    "source": "measured-industrial-adapter"
  }
}
```

L/T/R必须是相同采集时刻的独立实测数据。示例传入数值不代表已实际测量。超出原文标定范围时明确报错；不静默裁剪。

工业SDK应实现 `vision.camera.IndustrialCamera`，返回完整 `FramePacket`：同步可见光/NIR、干扰包、时间、session_id、frame_id。16位frame_id周期复用，历史唯一性由session_id补充。USB/普通视频适配没有NIR和温度光强传感器，均标为模拟数据。

## 结果

每帧UUID生成：`_original.png`原图、`_fused.png`融合图、`_result.png`标注图、`.json`。SQLite `history.sqlite3` 保存一次检测一条记录，payload保留所有候选（包括过滤项）及128维基因、链码、帧绑定、动态阈值、参考框、理由、融合类型和清晰度。

- `unconfirmed`：没有对应基准、匹配不足或动态阈值不通过；不能解释为“已确认缺陷”。
- `matched`：同一目标序列中跨帧一致且通过动态验证；仍不等于经过真实标签验证的工业缺陷。
- `filtered`：有空间候选，但相似度低于伪缺陷阈值，保留审计记录，不显示框。
- `model`：可选真实YOLO模型输出，类别来自该模型。

Demo的confidence为加权余弦分数，首次候选为0而非伪造置信概率。CSV包括无候选帧（空缺陷字段），不会漏掉正常帧。CSV带UTF-8 BOM，适合Excel；对公式前缀做转义。

## 基准灰度标定

准备至少100帧同一目标、同一视角、无干扰的缺陷图像，可附配准NIR：

```bash
python -m scripts.build_baseline /path/to/100_frames data/input/baseline.npz
```

将 `config/config.json` 的 `baseline_path` 设置为 `data/input/baseline.npz`。逐像素平均保存为float灰度图，匹配时按基准缺陷ROI计算Gbase。软件只能校验数量和尺寸，无法判断采集是否真的“无干扰”，需实验人员保证。默认没有这100帧，使用参考帧ROI均值并明确记录 `single-reference-demo`。

## 干扰系数标定与受监督学习

样本JSON数组，每个样本含唯一ID、干扰包、独立实测灰度漂移换算的 `measured_c`，可选人工 `match_label` 和四个对应相似度：

```json
[
  {
    "sample_id": "experiment-001",
    "interference": {"light_lux": 6000, "temperature_c": 32, "reflectance": 0.5, "source": "measured"},
    "measured_c": 0.45,
    "similarities": [0.95, 0.92, 0.96, 0.88],
    "match_label": 1
  }
]
```

上述单条是格式示例，不能拿它充当100个样本。初始标定至少100组并覆盖三个独立干扰维度：

```bash
python -m scripts.calibrate measurements.json
python -m scripts.calibrate new_feedback.json --incremental
```

初始采用有界约束最小二乘，报告最大误差是否≤.02。增量按配置每1000个新ID进行一次随机顺序遍历（lr=.001）；已消费ID不重复学习。未满批次的样本留在输入JSON下次提交。measured_c更新耦合参数；存在人工标签才更新权重与阈值。状态写入 `learning_state` 路径；下次开始检测加载，正在运行的序列不热切换。收敛差与标定误差如未达标会如实保存，不强制宣告收敛。

均值误差、边缘清晰度、过滤比例不是精度/召回率；真实性能必须使用有标注的独立测试集评估。损失函数、约束投影和权重校正方式是原文未定义部分的工程补充。
