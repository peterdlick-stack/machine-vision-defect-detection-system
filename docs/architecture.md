# 软件架构与实现计划

先完成 analysis.md，再确定本架构，之后才编写业务代码。

| 层 | 文件 | 职责 |
|---|---|---|
| 入口 | main.py | GUI、无界面演示与烟雾验证 |
| 交互 | app/main_window.py、worker.py | PySide6控件、参数、历史、异步QThread；worker单独持有服务和算法状态 |
| 数据契约 | vision/types.py | 干扰包、帧、特征、检测结果及JSON转换 |
| 算法 | vision/preprocessing.py、features.py、adaptive.py | 二级小波、候选与四基因、权重/匹配/融合 |
| 适配 | models/base.py、demo.py、yolo.py | 可替换检测器，真实模型显式加载 |
| 采集 | vision/camera.py | USB/视频/工业适配Protocol |
| 服务 | services/detection.py、storage.py、learning.py | 编排、文件/SQLite/CSV、标定与SGD |
| 配置 | utils/config.py、config/config.json | 严格参数验证与项目根目录路径 |
| 演示 | scripts/generate_demo.py、scripts/calibrate.py | 确定性样本、离线标定/受监督增量更新 |
| 验证 | tests/ | 数学公式、边界、持久化、坏图、批量、GUI线程与端到端 |

数据库一行保存一次帧结果，关键字段可查询，完整payload包含基因、干扰、阈值、匹配和融合指标；图片独立PNG保存，JSON同名。学习状态独立JSON，使用原子替换。不会用演示检测“置信度”冒充校准后的概率。

线程：主线程只选择输入、显示结果；QThread逐帧采集/解码/推理/持久化，发出结果后等待主线程确认以限制待显示帧积压。停止采用线程安全Event，收尾释放VideoCapture；关闭窗口等待后台安全完成，不terminate线程。不同输入/参数启动新检测器，序列内共享基准；数据库连接不跨线程共享。

工程补充的候选提议与局部配准可分别替换。真实工业适配器应返回同步FramePacket；缺失NIR不自动造假。YOLO只提供独立目标检测模式，不宣称具备文档的全部融合功能。
