# 真机参考帧

这些 JPEG/PNG 是从 `data-cs` 那台机器上**只读**取出来的，用来做 real-to-sim 的视觉对照。

现在以 `frames/manifest.json` 与其 33 组同步记录（99 张原始 JPEG）为精确帧关联依据：每条轨迹 11 个采样点，包含 0 / 100 / 300 / 500 帧及动作附近的补充帧；核验源轨迹哈希、时间及全部状态/动作，再读取该记录的 `rgb_paths`。相机对应关系由已审计的转换器确认：cam_0 为左腕，cam_1 为右腕，cam_2 为头部。图像字节不经过裁剪或重编码，manifest 保留原路径、尺寸与哈希。它们仍只是同一 logger 记录的图像，不证明曝光同步。

下列旧静帧保留作历史参考；部分头部图已经裁剪，且未保留精确的解码帧编号，不能用于像素坐标标定。网页「真机对照」使用新原始样本。

**时间含义修正（2026-10-01）：** 以下文件名和表格中的秒数是重新编码的 30 fps 视频播放时间，不是原始采集时间。转换脚本将时间写成 frame/30；前三条原始记录实际平均约 27.06–27.75 Hz。精确对照须用视频的解码帧编号映射到 `acquisition.json` 的记录时间，不能直接比较相同的秒数。

`acquisition.json` 保存原始轨迹文件哈希、相对 logger 时间和全部 16 维状态/动作逐帧匹配证据，并绑定 `real_trajectory.json` 哈希。当前记录器源码尚未与采集时版本建立不可变关联；记录时间不等于相机曝光、反馈读取或 CAN 下发时间。头部视频实际解码和原始 JPEG 均为 1280×960，元数据的 720 高度来自转换脚本硬编码；内外参与畸变仍未完成标定。

标定口径补充：`real_trajectory.json` 顶层对 base→cell 偏移的旧说明使用了“measured/exact”措辞；但已有零 FK 残差来自同一软件运动学链，不能独立证明真实底座、桌面或 TCP 的空间精度。当前证据等级以 `sim/calibration.json` 为准：机器人几何为 `software_match`，基座仍为 `candidate`。旧 VR 左/右腕变换也不是相机外参。

```
ssh data-cs ffmpeg -ss <t> -i ~/Documents/moqi_workspace_20251119_211400/lerobot_data/videos/observation.images.<cam>/chunk-000/file-000.mp4 \
  -frames:v 1 -f image2pipe -vcodec png -
```

帧是管道到 stdout 的，那台机器上**没有写任何文件**，也没有改动任何数据。

| 文件 | 来源 | 说明 |
|---|---|---|
| `real_right_t3.png` | episode 0，第 3 秒，`observation.images.right` | 右腕相机：黑瓶立在白台面上，黑色两指夹爪在下方 |
| `real_right_t9.6.png` | episode 0，第 9.6 秒 | 右腕相机：贴在瓶身上（夹爪闭合瞬间） |
| `real_right_t14.png` | episode 0，第 14 秒 | 右腕相机：搬运中 |
| `real_left_t14.png` | episode 0，第 14 秒，`observation.images.left` | 左腕相机：同一时刻的另一侧 |
| `real_right_t17.2.png` | episode 0，第 17.2 秒 | 右腕相机：交接瞬间，黑色手指贴着瓶身，瓶身有横向环纹 |
| `real_left_t17.2.png` | episode 0，第 17.2 秒 | 左腕相机：**瓶盖朝上**，白盖橙字，下面是深色瓶身，可见瓶盖与瓶身宽度比约 1.5 |
| `real_head_t14.png` | episode 0，第 14 秒，`observation.images.head` | 头部鱼眼：整个实验台和双臂 |
| `real_head_t2.png` | episode 0，第 2 秒 | 瓶子立着，两个夹爪都张着 |
| `real_head_t4.png` | episode 0，第 4 秒 | 右手伸到瓶子旁边，瓶子立着 |
| `real_head_t6.png` | episode 0，第 6 秒 | 瓶子开始倾，右手两指正贴在瓶身上 |
| `real_head_bottle_t2to12.png` | episode 0，t = 2/4/6/8/10/12 s | **同一块区域连着六帧**，见下 |

### 瓶子到底倒没倒

上面这张连帧是专门为了回答这一件事取的：把头部相机里瓶子所在的那块裁下来，每隔 2 秒并排放。看下来是

    t=2  立着         t=4  立着
    t=6  开始倾        t=8  夹爪已经贴上去
    t=10 被握着、倾斜   t=12 被端着走

**它从来没有躺平过。** 之前这里写的是"瓶子在 t=6 s 就已经躺下了，之后夹的是躺着的瓶子"——那是从单帧的倾斜读出来的，多取几帧就站不住了。这一条影响摆位：躺平的瓶子轴在台面之上 28 mm，而记录里 t=9.33 s 的抓握点在台面之上 143 mm，够不到躺着的瓶子，只能够到立着的。

取帧命令（在那台机器上**只读**，帧直接管道到 stdout，不落任何文件）：

```
ssh data-cs ffmpeg -ss <t> -i ~/Documents/moqi_workspace_20251119_211400/lerobot_data/videos/observation.images.head/chunk-000/file-000.mp4 \
  -frames:v 1 -f image2pipe -vcodec png -
```

数据集本身（148 episodes / 107821 帧 / 30 fps）留在那台机器上，本机只留这几张对照用的静帧，不复制数据集。
