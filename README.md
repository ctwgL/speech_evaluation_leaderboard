# 语音评测数据集

本项目使用自动语音识别（ASR）、语音增强（SE）和语音合成（TTS）数据集，覆盖中英文朗读、
会议、真实节目、地域口音、耳机与免提设备，以及零样本、跨语种和情感语音合成等场景。下列样本数与
时长来自当前本地数据版本；时长为输入或参考音频的累计时长，不代表模型生成耗时。

## 在线评测结果

ASR 与 TTS 模型的性能指标通过网页进行可视化展示，可访问：

<https://ctwgL.github.io/speech_evaluation_leaderboard/>

该网站用于查看语音识别和语音合成模型的综合指标、各测试集结果及模型间的性能对比。

## 数据规模总览

| 任务 | 数据集家族 | 子集数 | 样本数 | 总时长 |
| --- | ---: | ---: | ---: | ---: |
| 语音识别（ASR） | 7 | 55 | 158,292 | 197.37 小时 |
| 语音增强（SE） | 2 | 4 | 1,994 | 13.66 小时（带噪语音） |
| 语音合成（TTS） | 2 | 8 | 12,951 | 21.29 小时 |

## 语音识别数据集

ASR 评测以中文为主，同时包含英文朗读和演讲。中文数据通常使用 CER，
英文数据通常使用 WER；跨模型比较时应保证使用相同子集和文本归一化规则。

### 数据集家族与评测目的

| 数据集家族 | 语言与场景 | 子集数 | 样本数 | 总时长 | 主要评测目的 |
| --- | --- | ---: | ---: | ---: | --- |
| AISHELL-1/2 | 中文普通话朗读；AISHELL-2 含 Android、iOS 和麦克风通道 | 4 | 22,176 | 22.05 小时 | 普通话识别及跨设备、跨录音通道鲁棒性 |
| AliMeeting | 中文会议语音；近场/远场，eval/test 划分 | 4 | 32,813 | 32.89 小时 | 多说话人会议、重叠语音和远场拾音 |
| King-ASR | 12 类中文地域重口音普通话，使用手机录制 | 12 | 35,265 | 42.80 小时 | 不同地域重口音的识别鲁棒性 |
| LibriSpeech | 英文有声书朗读；test-clean/test-other | 2 | 5,559 | 10.75 小时 | 英文朗读语音及不同声学难度 |
| SpeechIO 中文测试集 | 中文真实节目与网络内容，涵盖新闻、访谈、教育、直播、文娱和垂直领域 | 27 | 43,178 | 65.78 小时 | 真实场景、长尾词汇和跨领域泛化能力 |
| TED-LIUM 3 | 英文 TED 演讲；legacy dev/test | 2 | 1,662 | 4.22 小时 | 英文演讲和自发语音识别 |
| 中文口音/方言扩展集 | 湖北重口音、四川、武汉和郑州日常语句 | 4 | 17,639 | 18.89 小时 | 地域口音与日常口语识别 |

SpeechIO 的 27 个中文子集来自公开网络内容并由专业人员标注，场景跨度较大。例如，
`ZH00001` 为新闻播报，`ZH00002` 为电视访谈，`ZH00005` 为在线教育，`ZH00006`～
`ZH00007` 为直播内容，后续子集还覆盖播客、相声、脱口秀、动画、体育解说、汽车、
房地产、产品评测、农业和历史哲学等领域。保留原始编号便于与官方榜单对齐。

数据集参考：[SpeechIO Leaderboard](https://github.com/SpeechColab/Leaderboard)。

<details>
<summary>AISHELL、AliMeeting、LibriSpeech 与 TED-LIUM 3 明细</summary>

| 子集 | 样本数 | 时长（小时） |
| --- | ---: | ---: |
| `speechio/AISHELL1_TEST` | 7,176 | 10.03 |
| `speechio/AISHELL2_ANDROID_TEST` | 5,000 | 4.00 |
| `speechio/AISHELL2_IOS_TEST` | 5,000 | 4.00 |
| `speechio/AISHELL2_MIC_TEST` | 5,000 | 4.00 |
| `speechio/ALIMEETING_EVAL_FAR_FIELD` | 4,495 | 4.62 |
| `speechio/ALIMEETING_EVAL_NEAR_FIELD` | 4,504 | 4.64 |
| `speechio/ALIMEETING_TEST_FAR_FIELD` | 11,891 | 11.80 |
| `speechio/ALIMEETING_TEST_NEAR_FIELD` | 11,923 | 11.83 |
| `speechio/LIBRISPEECH_TEST_CLEAN` | 2,620 | 5.40 |
| `speechio/LIBRISPEECH_TEST_OTHER` | 2,939 | 5.34 |
| `speechio/TEDLIUM_RELEASE3_LEGACY_DEV` | 507 | 1.60 |
| `speechio/TEDLIUM_RELEASE3_LEGACY_TEST` | 1,155 | 2.62 |

</details>

<details>
<summary>King-ASR 与中文口音/方言扩展集明细</summary>

| 子集 | 方言/口音类型 | 样本数 | 时长（小时） |
| --- | --- | ---: | ---: |
| `speechio/King-ASR-384-1` | 福建重口音普通话（手机） | 1,600 | 1.84 |
| `speechio/King-ASR-384-2` | 广东重口音普通话（手机） | 1,554 | 2.15 |
| `speechio/King-ASR-384-3` | 湖南重口音普通话（手机） | 1,200 | 1.44 |
| `speechio/King-ASR-384-4` | 江西重口音普通话（手机） | 1,585 | 2.23 |
| `speechio/King-ASR-384-9` | 吴重口音普通话（手机） | 5,196 | 7.02 |
| `speechio/King-ASR-384-10` | 天津重口音普通话（手机） | 3,000 | 4.05 |
| `speechio/King-ASR-384-11` | 安徽重口音普通话（手机） | 5,277 | 5.54 |
| `speechio/King-ASR-384-12` | 山东重口音普通话（手机） | 3,000 | 3.18 |
| `speechio/King-ASR-384-13` | 河南重口音普通话（手机） | 2,985 | 3.10 |
| `speechio/King-ASR-384-17` | 湖北重口音普通话（手机） | 3,000 | 3.53 |
| `speechio/King-ASR-384-18` | 甘肃重口音普通话（手机） | 5,277 | 6.60 |
| `speechio/King-ASR-881` | 四川重口音普通话（手机） | 1,591 | 2.12 |
| `speechio/hubei_zhongkouyin` | 湖北重口音普通话 | 903 | 1.66 |
| `speechio/sichuan_daily_sentence` | 四川日常语句 | 6,522 | 7.06 |
| `speechio/wuhan_daily_sentence` | 武汉日常语句 | 5,082 | 5.14 |
| `speechio/zhengzhou_daily_sentence` | 郑州日常语句 | 5,132 | 5.04 |

</details>

<details>
<summary>SpeechIO 中文测试集明细</summary>

| 子集 | 样本数 | 时长（小时） |
| --- | ---: | ---: |
| `speechio/SPEECHIO_ASR_ZH00000` | 879 | 1.60 |
| `speechio/SPEECHIO_ASR_ZH00001` | 5,069 | 8.72 |
| `speechio/SPEECHIO_ASR_ZH00002` | 2,993 | 2.75 |
| `speechio/SPEECHIO_ASR_ZH00003` | 1,683 | 2.39 |
| `speechio/SPEECHIO_ASR_ZH00004` | 1,311 | 2.82 |
| `speechio/SPEECHIO_ASR_ZH00005` | 3,148 | 4.38 |
| `speechio/SPEECHIO_ASR_ZH00006` | 1,561 | 1.66 |
| `speechio/SPEECHIO_ASR_ZH00007` | 770 | 0.98 |
| `speechio/SPEECHIO_ASR_ZH00008` | 884 | 1.28 |
| `speechio/SPEECHIO_ASR_ZH00009` | 3,466 | 4.64 |
| `speechio/SPEECHIO_ASR_ZH00010` | 2,251 | 4.21 |
| `speechio/SPEECHIO_ASR_ZH00011` | 1,053 | 3.45 |
| `speechio/SPEECHIO_ASR_ZH00012` | 1,170 | 3.39 |
| `speechio/SPEECHIO_ASR_ZH00013` | 1,321 | 1.78 |
| `speechio/SPEECHIO_ASR_ZH00014` | 856 | 1.01 |
| `speechio/SPEECHIO_ASR_ZH00015` | 1,168 | 2.15 |
| `speechio/SPEECHIO_ASR_ZH00016` | 1,201 | 1.01 |
| `speechio/SPEECHIO_ASR_ZH00017` | 1,271 | 1.80 |
| `speechio/SPEECHIO_ASR_ZH00018` | 899 | 0.85 |
| `speechio/SPEECHIO_ASR_ZH00019` | 615 | 0.70 |
| `speechio/SPEECHIO_ASR_ZH00020` | 1,590 | 2.18 |
| `speechio/SPEECHIO_ASR_ZH00021` | 1,035 | 1.74 |
| `speechio/SPEECHIO_ASR_ZH00022` | 1,026 | 1.68 |
| `speechio/SPEECHIO_ASR_ZH00023` | 1,528 | 2.11 |
| `speechio/SPEECHIO_ASR_ZH00024` | 1,930 | 2.65 |
| `speechio/SPEECHIO_ASR_ZH00025` | 1,164 | 1.41 |
| `speechio/SPEECHIO_ASR_ZH00026` | 1,336 | 2.43 |

</details>

## 语音增强数据集

语音增强数据由 V5 盲测集和开发测试集两部分组成，每部分均划分为耳机
`Track1_Headset` 和免提设备 `Track2_Speakerphone`。每条样本通过相同文件名关联：

- `enrol`：目标说话人的注册音频，供个性化增强及后续指标计算使用。
- `noisy`：需要输入语音增强模型的带噪音频。
- `enhanced`：模型生成的增强音频，不属于原始数据集，由运行结果目录提供。

| 数据集分组 | 设备场景 | 样本数 | 带噪语音时长（小时） |
| --- | --- | ---: | ---: |
| `blind/Track1_Headset` | 盲测集 · 耳机 | 389 | 4.91 |
| `blind/Track2_Speakerphone` | 盲测集 · 免提设备 | 364 | 5.30 |
| **盲测集小计** |  | **753** | **10.21** |
| `test/Track1_Headset` | 开发测试集 · 耳机 | 641 | 1.78 |
| `test/Track2_Speakerphone` | 开发测试集 · 免提设备 | 600 | 1.67 |
| **开发测试集小计** |  | **1,241** | **3.45** |
| **合计** |  | **1,994** | **13.66** |

SE 的时长按待处理的 noisy（带噪）音频统计。四个分组必须分别保存结果，以免盲测/测试集
或不同 Track 中的同名文件发生冲突。

## 语音合成数据集

TTS 评测同时关注内容正确性、说话人相似度、音质和表达能力。客观评测通常使用中文
CER/英文 WER、说话人相似度（SIM）与 DNSMOS；情感子集还可计算情感一致性，主观子集
用于人工听测。统计时长是提示音频或参考音频时长，生成结果应单独保存和计时。

### CV3-Eval

[CV3-Eval](https://github.com/QwenAudio/CV3-Eval) 面向真实开放环境下的零样本语音合成，
参考音频来自 Common Voice、FLEURS、EmoBox 及网络音频，覆盖多语言/方言、跨语种、
多领域、多录音环境、情感和风格。其客观子集用于多语种音色克隆、跨语种音色克隆和
情感克隆；主观子集用于表现力音色克隆、表现力续说及中文口音克隆。

| 子集 | 评测内容 | 样本数 | 时长（小时） |
| --- | --- | ---: | ---: |
| `CV3-Eval/zero_shot` | 多语种零样本音色克隆 | 4,624 | 8.41 |
| `CV3-Eval/cross_lingual_zeroshot` | 跨语种零样本音色克隆 | 2,524 | 4.70 |
| `CV3-Eval/emotion_zeroshot` | 情感零样本音色克隆 | 300 | 0.41 |
| `CV3-Eval/subjective_zeroshot` | 零样本合成主观听测 | 194 | 0.40 |
| `CV3-Eval/subjective_continue` | 表现力语音续说主观听测 | 180 | 0.26 |
| **小计** |  | **7,822** | **14.17** |

### Seed-TTS Eval

[Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval) 是中英文域外零样本合成
客观评测集。每条数据包含提示文本、提示音频、待合成文本和可选的真实目标音频，主要
评估合成内容正确性（CER/WER）和说话人相似度（SIM）。

| 子集 | 语言 | 样本数 | 时长（小时） |
| --- | --- | ---: | ---: |
| `seed_tts_eval_dataset/zh` | 中文普通话 | 3,030 | 4.51 |
| `seed_tts_eval_dataset/en` | 英文 | 2,099 | 2.61 |
| **小计** |  | **5,129** | **7.12** |
