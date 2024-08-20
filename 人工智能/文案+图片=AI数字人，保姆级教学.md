# 前文
最近接了个小需求，就是家里侄女出生了，家里人拍了一张照片，想做一个让她开口说话和大家打招呼的视频，这个重任自然是落到了我的肩膀上了，索性咱们有技术有硬件，干脆就做一下，给家人添点乐趣，这也是技术应该有的人文温度，这里鉴于隐私问题，我只用动漫人物展示效果，用真人照片的效果是一样的。


# 准备
- **24G 显卡**，我测试过了使用小的显存跑实在是费劲
- **[TTSMaker](https://ttsmaker.com/)**，这里可以将你输入的文本转换成语音，并且声音有多种选项， 效果还不错。
- **[MuseV](https://github.com/TMElyralab/MuseV)**，这个可以使你传上去的照片，根据你的提示词和超参数设置生成想要的短视频。我自己下载好了安装包，安装即用。
- **[MuseTalk](https://github.com/TMElyralab/MuseTalk)**，这个和上面的 MuseV 是配套使用的，用上面生成的短视频，结合你使用 ttsmaker 生成的语音直接生成一个可以说话的短视频，并且嘴型是很接近语音的动态效果。我自己下载好了安装包，安装即用。
- **一张待用的相片**，最好背景比较简单的单人照，面部清晰一点。

# 图片生成视频

- 找一张图片 `yongen.jpeg` 放到 `MuseV\data\images` 目录下面
- `MuseV\configs\tasks\example.yaml` 修改成如下配置，注释如下：

1. **condition_images** ：照片路径
2. **eye_blinks_factor**：提示词中眨眼的权重
3. **height**： 照片高
4. **img_length_ratio** ：原始的 height 和 width 乘以 img_length_ratio 就是输出视频的高和宽，默认 1 就可以
5. **ipadapter_image** ：IP-Apdater 需要处理的图片路径，这里默认使用原图片
6. **name** : 任务名字
7. **prompt** ：生成视频的提示词
8. **refer_image** ： 参考图片的路径，这里默认使用原图片
9. **video_path** ：用于 video2video 任务，这里设置 null 即可
10. **width** ： 照片宽

```
- condition_images: ./data/images/yongen.jpeg
  eye_blinks_factor: 1.8
  height: 1308
  img_length_ratio: 0.957
  ipadapter_image: ${.condition_images}
  name: yongen
  prompt: (masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)
  refer_image: ${.condition_images}
  video_path: null
  width: 736
```
写启动脚本` MuseV\文字到视频.bat` ，如下，这里有三个参数需要修改成自己的：

1. `target_datas`：就是上面一步配置的 name ，要完全一样否则报错。
2. `time_size`：这个是生成的视频时长，它除以 fps 就是最后的总秒数。
3. `fps` ：每秒帧数，越高越流畅
```
@echo off
CHCP 65001

set PYTHON=%CD%\env\python.exe
set FFMPEG_PATH=%CD%\env\ffmpeg\bin
set HF_ENDPOINT=https://hf-mirror.com

set PYTHONPATH=%CD%\;%CD%\MMCM;%CD%\diffusers\src;%CD%\controlnet_aux\src

@rem 启动方式
%PYTHON% ^
scripts/inference/text2video.py ^
--sd_model_name majicmixRealv6Fp16 ^
--unet_model_name musev ^
-test_data_path ./configs/tasks/example.yaml ^
--output_dir ./output ^
--n_batch 1 ^
--target_datas yongen ^
--time_size 24 ^
--fps 12

pause
```

生成日志如下，总共耗时 1 分 21 秒：

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7a775e0257894396ab3a4430d56c7cb1~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722389919&x-orig-sign=i%2FP%2BUg%2BRN9BmsN3uVsDLdLxvKqc%3D)

如果看到这样的日志说明成功，生成的视频放在了 `MuseV\output` 下面。然后将其改名 `yongen1.mp4` 放到 `MuseTalk\data\video` 下面待用。

# 文字合成语音

这一步是最简单的，去 https://ttsmaker.com/ 输入你想说的话，然后选择合适的音色转成语音即可，下载下来的音频改名为 `yongen1.wav` ，然后放到 `MuseTalk\data\audio` 下面。

# 视频合成语音

将` MuseTalk\configs\inference\test.yaml` 文件修改如下：
```
task_0:
 video_path: "data/video/yongen1.mp4"
 audio_path: "data/audio/yongen1.wav"
 bbox_shift: 20
```

然后启动 `MuseTalk\获取bbox_shift.bat` 文件，内容如下，消耗显存大约 13 G：
```
@echo off
CHCP 65001

set PYTHON=%CD%\env\python.exe
set FFMPEG_PATH=%CD%\env\ffmpeg\bin


@rem 启动方式
%PYTHON% -m scripts.inference --inference_config configs/inference/test.yaml

pause
```
当看到下面日志说明成功。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3e8ce887710c498f8e146761dcda5473~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722389919&x-orig-sign=uqBjQFTAm2eKukG8%2BcvApmvKTW4%3D)

结果放在了`MuseTalk\results` 。

# 生成效果
因为这里没法上传视频，大家可以看下我生成的动图，通过口型你们能猜到说的是什么吗？其实是“掘金的朋友们大家好，我是永恩”。

![yongen1_yongen1.gif](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/29f0e6f843b84af5bd7dec76ee3e4561~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722389919&x-orig-sign=5HSsvHtbY5CCd6Mnu59BieiWz0s%3D)

# 参考

- https://ttsmaker.com/
- https://github.com/TMElyralab/MuseV
- https://github.com/TMElyralab/MuseTalk