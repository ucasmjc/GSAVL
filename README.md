# When Segment Anything Model Meet Sound Source Localization.
2023年秋季学期《多媒体信息处理》大作业，得到老师和同学的高度评价，课程最终得分95，详见Report.pdf，代码在GSAVL/路径下。

我们将SAM应用到声源定位任务，在无监督的任务设置下，实现具有强大零样本泛化能力、且具有分割粒度的声源定位模型，进一步探索 SAM 在视听学习领域的潜力。
为此，我们小组提出了 **Generalizable Shape-aware Audio-Visual Localization(GSAVL)**模型，在无监督设置下实现了像素级声源定位预测，在声源分割数据集 AVSBench 上效果远超传统的
声源定位方法，相比 30.22%，我们实现了 48.84% 的定位精度。

GSAVL的先进性：

![Alt text](latex_/label.png)

模型结构图：

![Alt text](latex_/model2.jpg)