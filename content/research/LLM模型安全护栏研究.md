# 面向中英文生产部署的 LLM Safety Guard：技术全景、数据、评测与工程实施路线

## Executive Summary

本报告以你上传的项目 brief 为约束条件：现有系统是约 2B、LLaMA 类 Decoder-only Guard，当前痛点集中在安全召回不足、复杂/多轮/Jailbreak 泛化不足、Hard Negative 误报偏高，以及自定义 conversation template / label protocol 导致 NeMo Guardrails、Hugging Face、vLLM 等生态兼容性差。fileciteturn0file0

**核心结论是：不建议把新项目定义成“把旧 2B LLaMA 再训练好一点”，而应把它重构为一套“Safety Policy Engine + Guard Model + 标准化协议 + 独立评测体系”。** 截至 2026 年，Guard 技术已经明显从“固定标签分类”向三条路线收敛：其一是 Llama Guard 一类的 **Decoder-only generative classification**；其二是 Shieldstral、Granite Guardian 等体现的 **policy-conditioned / policy-adaptive scoring**；其三是 Qwen3Guard-Stream 一类将 **token-level safety head** 引入流式输出检测的架构。与此同时，Prompt Injection / Jailbreak detection 又重新证明了小型 encoder classifier 的价值，例如 Meta Prompt Guard 2 只有 22M/86M，而不是所有安全任务都需要 Decoder LLM。citeturn15search1turn21search0turn21academia48turn21search6

对你的项目，我建议采用**双层设计**：

```text
                    ┌────────────────────────────┐
                    │     Safety Policy Layer     │
                    │ taxonomy / thresholds /     │
                    │ policy version / mappings   │
                    └─────────────┬──────────────┘
                                  │
User / messages
      │
      ▼
Normalization / deterministic rules
      │
      ├────────► Prompt Injection / Jailbreak Detector
      │              22M–200M encoder-class model
      │
      ▼
┌──────────────────────────────────────────────────────┐
│        Main Bilingual Content Safety Guard           │
│  1B–3B Decoder-only, multi-turn, policy-aware        │
│                                                      │
│  internal: logits → verdict/category/severity        │
│  external: stable Guard JSON API                     │
└───────────────────────┬──────────────────────────────┘
                        │ safe
                        ▼
                    Main LLM
                        │
               streaming / final output
                        ▼
              Output Safety Guard
                        │
              allow / review / block
```

这种分层的原因是：**Jailbreak / Prompt Injection 是攻击机制，不等于有害内容类别。** 把“prompt injection”和“暴力、欺诈、自残、恶意软件”等全部塞进一个扁平 taxonomy，会让数据标注、threshold 和 policy logic 混乱；Meta 当前也将通用内容安全的 Llama Guard 与专门检测 prompt injection / jailbreak 的 Prompt Guard 分成不同模型。citeturn21search6turn15search1

**在主 Guard 的尺寸上，我建议 MVP 以 1B–3B 为中心，3B 左右作为质量优先版本，1B 左右作为低延迟版本。** 现有证据已经说明“小 Guard 可以成立”：Llama Guard 3 有 1B 版本；ShieldGemma 有 2B；Mistral 2026 年发布的 Shieldstral 是 3B policy-adaptive classifier；Qwen3Guard 更进一步提供 0.6B / 4B / 8B 三档。Llama Guard 3 作者自己的评测仍显示 8B 相对 1B 有质量优势，例如其内部英语评测 F1/FPR 为 0.939/0.040 与 0.899/0.090，但差距并不足以证明所有生产 Guard 都必须上 8B，而且这一结果是作者测试、policy 和数据并非统一第三方条件。citeturn15search1turn15search5turn21search0turn21academia48

**3B 尤其值得关注。** Shieldstral 2026 年把 moderation 表述成 policy-adaptive question answering，不再绑定固定 taxonomy；它通过单次 forward 的 yes/no logits 得到连续风险分数，支持运行时更换 policy，官方称单张 16GB NVIDIA GPU 即可运行，并以 Apache 2.0 发布。这里“匹敌大至约 7 倍模型”的性能数字来自 Mistral 自身测试，应视为 vendor claim，而不是独立统一排名；但其架构思想非常值得你的新项目借鉴。citeturn21search0turn21search13

**中文方面，不建议简单把英文数据翻译成中文。** 当前最有工程价值的开放 multilingual guard 数据之一是 NVIDIA `Nemotron-Safety-Guard-Dataset-v3`：约 514,617 条、12 种语言包含 Mandarin，CC-BY-4.0，并包含合成 jailbreak 数据；但它本身主要由 CultureGuard 对英语 Aegis 数据进行文化适配和合成扩充，官方也明确数据高度 synthetic，因此仍然需要你的原生中文 hard-negative / slang / 拼音 / 谐音 / 汉字拆分 / 中英混写数据补充。2026 年 ChiSafe-PAS 的人工标注测试集也专门发现了 Pinyin、character decomposition、internet slang 和 hedging 等中文规避方式。citeturn19search0turn24academia49

**不要把 Accuracy/F1 作为唯一上线指标。** 生产 Guard 的核心应该是 operating point：

\[
\text{Unsafe Recall @ FPR}\leq x
\]

例如分别报告 Recall@FPR=0.1%、0.5%、1%，再按 language/category/context 分桶。XSTest 的设计本身就是为了暴露“只看到敏感词就拒绝”的 exaggerated safety；NVIDIA 自己早期对 NeMo rails 的评测也显示 self-check 可以出现“高召回、低 precision”的过度防御情况。citeturn22academia0turn20search4

**最重要的数据投入不是继续堆普通 harmful samples，而是 Hard Negative。** 例如：

```text
安全：
“从防御角度解释勒索软件为什么会加密文件，并给出检测指标”

危险：
“写一个可以加密 Windows 用户文件并删除恢复点的勒索程序”
```

两条文本共享大量关键词，但 policy 语义完全不同。新 Guard 应显式学习 **intent × actionability × context × target × severity**，而不是“malware / weapon / suicide 等词是否出现”。NVIDIA 2026 年的 Nemotron-SFT-Safety-v2 甚至把“counteracting overly safe behavior through contrastive hard negatives”明确列为数据设计目标。citeturn19search1

**协议方面，建议完全放弃项目私有 conversation serialization。** 对外统一采用 OpenAI-style `messages[]`；模型仓库原生提供 Hugging Face `tokenizer.chat_template`；vLLM 直接通过 `/v1/chat/completions` 服务。Hugging Face 官方明确指出即使来自同一个 base model，不同 chat format 也可能完全不同，错误 control token 会明显损害模型表现；vLLM 当前也明确要求 Chat API 使用模型自带或显式指定的 chat template。citeturn15search0turn16search0

**输出方面不建议让模型自由生成完整 JSON。** Hot path 内部最好只做：

```text
safe
```

或：

```text
unsafe
H10,H12
```

甚至直接读取 `{safe, review, unsafe}` 首 token logits；业务 Serving 层再构造稳定、versioned JSON。Llama Guard 3 官方已经使用首 token probability 做 classifier scoring，而 Shieldstral更进一步直接用 yes/no logits 做单次 forward scoring；vLLM 也支持 choice/regex/JSON 等 structured outputs，可以作为兼容模式。citeturn15search1turn21search13turn16search13

**NeMo Guardrails 兼容不需要你把模型训练成 NVIDIA 模型。** 当前 NeMo 已支持 `content_safety` model type、Llama Guard、ShieldGemma，以及任意 OpenAI-compatible endpoint；vLLM 推荐直接使用 `engine: openai + base_url`。Content Safety rail 有 `content safety check input/output` 两条 flow，且支持自定义 prompt 与 output parser。因此真正应该标准化的是 API、messages、`safe/unsafe` semantics 和 category mapping，而不是绑定某个 NeMo 私有 prompt。citeturn20search0turn20search3turn20search5

最终建议的研发主线是：

```text
Policy & Taxonomy v1
        │
        ▼
Canonical bilingual schema
        │
        ▼
1B–3B modern multilingual decoder base
        │
        ▼
General Safety SFT / logit classification
        │
        ▼
Chinese-native + English data
        │
        ▼
Hard Negative / paired contrastive data
        │
        ▼
Jailbreak & multi-turn adversarial training
        │
        ▼
Calibration + per-policy thresholds
        │
        ▼
Unified benchmark + internal gold + shadow traffic
        │
        ▼
HF chat template + vLLM OpenAI API + NeMo adapter
```

## 范式、系统架构与 Safety Policy 设计

**概念边界。** 这些概念不应该在项目设计文档里混用。

| 概念 | 核心输入输出 | 主要用途 | 是否适合在线 block |
|---|---|---|---|
| Safety Guard Model | conversation/policy → risk/verdict/category | LLM 输入/输出安全审核 | **是** |
| Content Moderation | 内容 → 平台 policy violation | 更广义 UGC / 内容审核 | 是 |
| Reward Model | prompt+response → preference/scalar | RLHF/RLAIF 训练、排序 | 通常否 |
| Safety Reward Model | response → safety reward | safety alignment | 通常否 |
| LLM-as-a-Judge | arbitrary rubric + sample → judgment | 通用评测 | 可以，但成本高 |
| Jailbreak Detector | prompt → bypass-attempt score | 检测绕过安全规则意图 | 是 |
| Prompt Injection Detector | untrusted text → injection score | RAG / agent instruction-data 边界 | 是 |
| Rule Engine | pattern/state → deterministic action | PII、allowlist、协议规则等 | **是** |
| Guardrails Framework | 多个 model/rule/action 的 orchestrator | NeMo 等系统编排 | 本身不是模型 |

因此，**Guard Model 不是一种固定 architecture，而是一个角色。** Encoder classifier、decoder LLM、classification head、policy-conditioned scorer 都可以承担 Guard；decoder generative Guard 可以看成“专门训练并受约束的 LLM-as-a-Judge”，但工程上应尽量把它收敛成 calibrated classifier，而不是让它自由讨论。Llama Guard 3 就是 Decoder-only LLM，却以生成 `safe/unsafe + categories` 作为分类接口；Shieldstral则进一步将同一个问题压缩到 yes/no logits。citeturn15search1turn21search13

一个更完整的生产架构应该是：

```text
                              ┌────────────────────┐
                              │ Policy Registry     │
                              │ version / thresholds│
                              │ category mappings   │
                              └──────────┬─────────┘
                                         │
User ──► normalization ──► deterministic rules
                         │
                         ├──► injection / jailbreak detector
                         │
                         ▼
                   INPUT CONTENT GUARD
                         │
              ┌──────────┴──────────┐
              │                     │
            block                  allow
                                    │
                                    ▼
                                  LLM
                                    │
                          ┌─────────┴────────┐
                          │ streaming chunks │
                          ▼                  │
                    STREAM GUARD             │
                          │                  │
                          └─────────┬────────┘
                                    ▼
                           OUTPUT CONTENT GUARD
                                    │
                        allow / redact / block
                                    │
                                    ▼
                                  User
```

Agent/RAG 场景需要再增加：

```text
Retrieved document
       │
       ▼
Indirect Prompt Injection Guard
       │
       ▼
Agent Planner
       │
       ▼
Tool-call Policy Engine ──► allowlist / authz
       │
       ▼
External Tool
       │
       ▼
Observation sanitization
```

BIPIA 专门评估 indirect prompt injection；InjecAgent 包含 1,054 个 tool-agent 测试案例、17 类用户工具和 62 类攻击者工具，说明 agent injection 与普通 harmful-content classification 是不同攻击面。citeturn17search2turn17academia45

**技术路线的演进可以概括成四代，而不是简单的“BERT → LLaMA”。**

| 阶段 | 典型形式 | 优势 | 主要问题 | 现在的定位 |
|---|---|---|---|---|
| Encoder classifier | text → classifier head | 极低延迟、稳定 logits | 固定 taxonomy，复杂 policy reasoning 弱 | prompt injection、toxicity、一级快速筛选 |
| Decoder generative guard | conversation → `safe/unsafe` | 多轮、instruction、policy 理解更强 | decoding、格式漂移、延迟 | 主流 content guard |
| Policy-conditioned guard | policy + conversation → score | policy 可动态变化 | policy wording 敏感、成本增加 | 2025–26 重要方向 |
| Streaming / token-level guard | token state → risk | 实时阻断输出 | 服务实现复杂 | 高风险 streaming 应用 |

Llama Guard 3-1B 是第二类，而且允许把自定义 categories 直接传入 chat template；Granite Guardian 3.2-5B 接收 `risk_definition` 并做 Yes/No 判断；Shieldstral 将 policy 作为自然语言 Query；Qwen3Guard 则同时提供 Generative 和 Stream 两套模型，其中 Stream 引入 token-level classification head。citeturn15search1turn21search3turn21search13turn21academia48

另一个非常重要的趋势是**“小 classifier + 大 policy reasoner”的 cascade**。OpenAI 在 2025 年开放的 gpt-oss-safeguard 20B/120B 展示了运行时给 policy、再进行 safety reasoning 的路线；其官方材料同时明确指出，reasoning guard 的计算成本更高，而内部系统可以先使用更快的小 classifier 做高召回筛选，再把少数边界样本交给 safety reasoner。这个思路很适合你的项目，即使第二阶段不是 OpenAI 模型。citeturn7search0turn7search5

**推荐 taxonomy 不应只是一个 category list，而应设计成四个正交维度。**

建议内部 canonical schema：

| 维度 | 建议设计 |
|---|---|
| Decision | `safe / review / unsafe` |
| Content hazard | hierarchical multi-label |
| Security signal | jailbreak / prompt injection / tool abuse 等，独立于 content harm |
| Context modifier | educational / defensive / news / historical / fictional / quotation 等 |
| Severity | low / medium / high / critical |

Content hazard 建议以 MLCommons/Llama Guard 与 Aegis/Nemotron 的共同部分为骨架。Llama Guard 3 的 MLCommons taxonomy 有 13 类，包括 violent crimes、non-violent crimes、child sexual exploitation、specialized advice、privacy、weapons、hate、self-harm、sexual content 等；NVIDIA Aegis 2.0 则使用 12 个 top-level + 9 个 fine-grained subcategories。citeturn15search2turn19search4

推荐内部 v1：

```text
H01 Violence / Physical Harm
H02 Non-violent Crime / Illegal Facilitation
    ├─ Fraud / Scam
    ├─ Theft / Property Crime
    └─ Other criminal facilitation
H03 Weapons / Explosives
H04 Drugs / Controlled Substances
H05 Sexual Content
H06 Child Sexual Exploitation
H07 Self-harm / Suicide
H08 Hate / Harassment / Threat
H09 Privacy / PII / Doxxing
H10 Cyber Abuse / Malware
H11 High-stakes / Specialized Advice
    ├─ Medical
    ├─ Financial
    └─ Legal
H12 Deception / Manipulation / Misinformation
H13 IP / Copyright                    [按业务启用]

Security Signals:
J01 Jailbreak / Policy Circumvention
J02 Direct Prompt Injection
J03 Indirect Prompt Injection
J04 Tool / Data Exfiltration Attempt
```

这是**本报告基于 MLCommons、Aegis 和当前 agent-security 实践做的工程归纳，不是某个现成 benchmark taxonomy 的原样复制**。其关键好处是 taxonomy 可以通过 mapping adapter 映射成 MLCommons、Aegis 或业务自己的 policy，而不要求每次训练重新改 label id。支持这种分离的事实基础包括 Llama Guard 自定义 taxonomy、Aegis 的层级 taxonomy，以及 policy-conditioned Guard 已经能够按运行时 policy 工作。citeturn15search1turn19search4turn21search0

**Binary、multi-label 和 hierarchical 并不是三选一。** 推荐：

```text
model representation:
    multi-label + hierarchical + severity scores

policy engine:
    scores + thresholds → safe/review/unsafe

external action:
    binary allow/block if product requires
```

比如一个请求同时可以是：

```json
{
  "verdict": "unsafe",
  "categories": ["H02", "H10"],
  "severity": "high",
  "security_signals": ["J01"]
}
```

而不是强迫模型选择“cybercrime”或“jailbreak”其中一个。

**尤其不建议把生成式“confidence: 0.97”当成真正概率。** Confidence 应来自 logits，之后在独立 calibration set 上做 temperature scaling / isotonic regression 等校准。Llama Guard 模型卡本身就建议利用第一个生成 token 的 probability 做 classifier scoring，Shieldstral也是直接归一化 yes/no logits 后得到连续 score。citeturn15search1turn21search13

Policy-conditioned 建议采用**混合方案**：

```text
Standard hot path
fixed canonical taxonomy
→ one pass
→ lowest latency

Custom-policy mode
trusted server-side policy
+ content
→ policy-conditioned scoring
→ customer / product specific policy
```

不要把用户输入的“policy”直接当可信 policy；policy 必须来自 server-side registry，否则它本身会变成新的 prompt-injection channel。

## 开源 Guard 模型生态、尺寸选择与可借鉴项目

截至 2026 年 9 月，真正值得你做 baseline 的模型不只是 Llama Guard。

| 模型 | 机构 | 规模 | 架构 / 特征 | 中文/多语言 | License | 对项目的意义 |
|---|---|---:|---|---|---|---|
| Llama Guard 3 | Meta | **1B / 8B** | Decoder generative；input/output；可自定义 taxonomy | 多语言，但官方 1B 表中未覆盖中文 | Llama 3.x terms | **1B 基线首选**；研究首 token classification、taxonomy template citeturn15search1 |
| Llama Guard 4 | Meta | 12B | 新一代 multimodal guard | 多语言 | Llama terms | 技术演进参考，超过目标尺寸 citeturn0search6turn0search7 |
| Prompt Guard 2 | Meta | **22M / 86M** | DeBERTa classifier；prompt injection/jailbreak | 86M 含非英语 attacks | Llama terms | 证明 injection detector 应独立、小型化 citeturn21search6 |
| ShieldGemma | Google | **2B / 9B / 27B** | Gemma2 decoder；policy prompt；4 harm categories | 原始 text 版主要 English | Gemma terms | 2B 对照、policy-conditioned 早期实践 citeturn15search5 |
| WildGuard | AI2 | 7B-class | prompt harm + response harm + refusal 三任务 | English | 开放模型/数据 | Hard-negative、refusal task 和 unified moderation 设计很有参考价值 citeturn2academia49turn18search0 |
| Granite Guardian 3.2 | IBM | **5B** | Yes/No scoring；custom risk definition；含 RAG/agent risks | 目前训练/测试主要 English | Granite open model | policy definition + RAG/agent guard 参考 citeturn21search3turn21search8 |
| Nemotron Safety Guard | NVIDIA | 8B | Aegis taxonomy、input/output，深度 NeMo 集成 | multilingual 版本包含 Mandarin | NVIDIA/model terms | **NeMo 接口和数据 recipe 最值得研究** citeturn20search1turn19search0 |
| Nemotron 3/3.5 Content Safety | NVIDIA | **约 4B** | Gemma-3-4B base，multilingual / multimodal；custom policy 演进 | 包含中文 | OpenMDW + Gemma terms | 2026 年 NVIDIA 当前路线参考 citeturn18search8 |
| Qwen3Guard | Qwen | **0.6B / 4B / 8B** | Gen + Stream；safe/controversial/unsafe；token-level head | **119 languages/dialects** | Apache-2.0 | **中英文+流式方案重点 baseline** citeturn21academia48 |
| Shieldstral | Mistral | **3B** | policy-adaptive；single-forward yes/no logits；multimodal | 12 languages incl. Chinese | Apache-2.0 | **最贴近新项目理想架构** citeturn21search0turn21search13 |
| gpt-oss-safeguard | OpenAI | 20B / 120B | policy-at-inference safety reasoning | 多语言取决于 base | Apache-2.0 | 超出尺寸范围，但 policy reasoning/cascade 很值得研究 citeturn7search0 |

这里不能做一个简单的“F1 排名”。Llama Guard 模型卡自己明确提醒：Guard 使用不同 policy taxonomy、prompt、label mapping，直接横比模型性能容易失真。Qwen3Guard 的 “SOTA” 和 Shieldstral 的 “匹敌至多约 7 倍参数模型”也分别属于作者报告；因此这些结论应该作为候选筛选依据，而不是采购/上线结论。citeturn15search1turn21academia48turn21search0

目前能看到一个很有价值的**同家族 size/quantization 对照**：Meta 的内部评测如下。

| Meta 内部测试 | English F1 / FPR | XSTest F1 / FPR |
|---|---|---|
| Llama Guard 3-8B | 0.939 / 0.040 | 0.884 / 0.044 |
| Llama Guard 3-1B | 0.899 / 0.090 | 0.821 / 0.068 |
| Llama Guard 3-1B INT4 | 0.904 / 0.084 | 0.737 / 0.152 |

这是作者内部测试，不应该外推到你的中文 production traffic；但它说明两个重要问题：**一，1B 并非不可用；二，量化影响可能高度 benchmark-dependent——English F1 基本没掉，并不意味着 over-refusal / boundary robustness 没掉。** citeturn15search1

因此在 size 选择上：

| Size | BF16 权重理论占用* | INT8 | INT4 | 建议角色 |
|---:|---:|---:|---:|---|
| 0.5B | ~1 GB | ~0.5 GB | ~0.25 GB | first-stage / edge |
| 1B | ~2 GB | ~1 GB | ~0.5 GB | 极低成本在线 Guard |
| 2B | ~4 GB | ~2 GB | ~1 GB | 很适合 MVP |
| 3B | ~6 GB | ~3 GB | ~1.5 GB | **推荐 quality/cost sweet spot** |
| 7B | ~14 GB | ~7 GB | ~3.5 GB | 高复杂 policy / reranker |
| 8B | ~16 GB | ~8 GB | ~4 GB | 高质量 second stage |

\*这里只是 `params × bytes` 的 weight-only 理论量，不包含 KV cache、activation、workspace、allocator、batching 等实际 runtime memory。

**模型越大不代表安全效果必然线性变好。** HarmBench 对模型安全的研究本身就强调，攻击/防御表现高度依赖训练和攻击方式，而不是简单由参数量决定；Qwen3Guard 同时发布 0.6B/4B/8B，以及 Shieldstral 把 3B 用于 policy-adaptive moderation，都说明当前优化重点已从“扩大生成模型”转为“任务特化 + 数据 + scoring architecture”。citeturn23academia49turn21academia48turn21search0

我建议你实际深入源码/数据的项目按下面优先级读：

| 优先级 | 项目 | 最应该学什么 |
|---|---|---|
| P0 | **Qwen3Guard** | Gen vs Stream、中文/多语言、tri-class、token-level head |
| P0 | **Shieldstral** | policy-adaptive QA、single-forward logits、3B efficiency |
| P0 | **Llama Guard 3-1B** | HF chat template、custom taxonomy、first-token probability |
| P0 | **NVIDIA Aegis/Nemotron + NeMo Guardrails** | dataset/taxonomy/NeMo integration 全链路 |
| P1 | **WildGuard** | prompt/response/refusal 三任务、adversarial/hard-negative 数据 |
| P1 | **Granite Guardian 3.2** | runtime risk definitions、RAG/agent risks |
| P1 | **ShieldGemma 2B** | 2B policy-conditioned baseline |
| P1 | **Prompt Guard 2** | 把 injection detection 从 content safety 中拆出去 |
| Architecture study | **gpt-oss-safeguard** | policy reasoner + cheap classifier cascade |

如果项目只允许跑五个 baseline，我会选：

> **Llama Guard 3-1B、Qwen3Guard-0.6B、Qwen3Guard-4B、Shieldstral-3B、Nemotron-3.5-Content-Safety-4B。**

其中 WildGuard/Granite Guardian 再作为英语 side benchmark。

## 训练数据、Hard Negative、中文增强与数据治理

当前可用数据要分清楚“**真正适合 Guard classification**”和“只是 safety alignment / preference 数据”。

| Dataset | Language | 规模 | Labels / Task | Jailbreak | License | Guard 训练价值 | 主要风险 |
|---|---|---:|---|---|---|---|---|
| **Nemotron Safety Guard Dataset v3** | 12 lang，含中文 | **514,617** | prompt/response safe labels + categories | 是 | **CC-BY-4.0** | **P0** | 高比例 synthetic / CultureGuard citeturn19search0 |
| **Aegis / Nemotron Content Safety v2** | EN | **33,416** | input/output labels；12+9 taxonomy | 部分 | **CC-BY-4.0** | **P0** | 来源混合、需跨集 dedup citeturn19search4 |
| **WildGuardTrain** | EN | **86,759** | prompt harm / response harm / refusal | **是** | ODC-By + access conditions | **P0** | 87% synthetic，GPT-4 labels citeturn18search0 |
| **PKU-SafeRLHF** | EN | 82K+ current main subset；family 100K–1M | paired response safety、category、severity | 间接 | **CC-BY-NC-4.0** | P1 | **商业使用限制**；不是纯 Guard 数据 citeturn18search1 |
| **BeaverTails** | EN | 30K/330K family | QA safety + 14 harm categories | 否/有限 | **CC-BY-NC-4.0** | P1 | 商业限制、版本/扩展集差异大 citeturn18search3 |
| **Nemotron-SFT-Safety-v2** | EN + 6 translations incl. ZH | ~130K | safe alignment responses / jailbreak / oversafety sources | **是** | CC-BY-4.0 + component terms | P1/P0 数据生成素材 | 不是直接 moderation label；翻译数据 citeturn19search1 |
| XSTest | EN | 450 | safe hard negatives + unsafe contrast | 否 | research benchmark | **不要训练** | Benchmark contamination citeturn22academia0 |
| SafetyBench | ZH/EN | 11,435 | safety knowledge MCQ | 否 | open benchmark | **不要训练** | Benchmark contamination citeturn23academia48 |
| CHiSafetyBench | ZH | 1,567 MCQ + 563 QA | 5 areas / 31 categories | 部分 | open | **保留评测** | 中文 benchmark contamination citeturn24academia48turn24search12 |
| ChiSafe-PAS | ZH | 1,897；1,544 full-gold | adversarial + risk + obfuscation | 是 | research resource | **优先留作测试** | 很新、规模小，训练会破坏价值 citeturn24academia49 |
| CValues | ZH | 多场景 | safety + responsibility | adversarial prompts | open research | Eval / data design inspiration | 公开 benchmark 污染 citeturn24academia50 |

Aegis 2.0 尤其适合作为 schema anchor：当前数据卡给出的 33,416 条包括 30,007 train、1,445 validation、1,964 test，还额外有 5,000 synthetic refusals；它通过 human annotation 与 multi-LLM jury 组合标注 responses。citeturn19search4

WildGuard 的价值不只在数量，而是**任务设计**：它将 prompt harmfulness、response harmfulness、response refusal 同时作为标签；训练集 86,759 条，其中 87% synthetic、11% in-the-wild、2% annotator-written，并明确同时包含 benign/harmful 和 vanilla/adversarial prompts。作者还用 500 条人工 audit GPT-4 labels，三个任务的人机一致率分别报告为 92%、82%、95%。citeturn18search0

这恰好提示你的数据 schema 应至少包含：

```json
{
  "messages": [...],
  "direction": "input | output",
  "verdict": "safe | review | unsafe",
  "categories": ["H10"],
  "severity": "high",
  "security_signals": ["J01"],
  "context": {
    "intent": "educational | defensive | operational | unknown",
    "actionability": "low | medium | high"
  },
  "source": "...",
  "language": "zh-CN",
  "annotation_source": "human | synthetic | inherited",
  "policy_version": "guard-policy-v1"
}
```

这里的 `context` 不一定全部作为模型输出，但应该保留为 annotation metadata，否则后面你几乎无法分析 false positive。

**中文不能只是 translation split。** NVIDIA 的 514K multilingual guard set 已经是很好的启动集，但官方明确说明它主要通过 CultureGuard 从 English Aegis 2.0 做文化适配/translation，并加入 synthetic jailbreak；2026 年 ChiSafe-PAS 反过来说明原生中文攻击会利用 Pinyin、字符拆解、网络俚语和 hedging。由此推断，你至少需要自己构造一套 **Chinese-native safety corpus**。citeturn19search0turn24academia49

建议中文子集专门覆盖：

```text
普通汉语
+ 中英混写
+ 拼音
+ 拼音缩写
+ 同音/谐音
+ 汉字拆分
+ 繁简混写
+ emoji/符号插入
+ 全角/半角/Unicode homoglyph
+ 网络黑话
+ 委婉表达
+ 上下文省略
+ 多轮逐步升级
```

而且每一种攻击形式都要有对应的**安全 hard negative**，否则你最终只会训练出“看到拼音/乱码就 unsafe”的检测器。

**Hard Negative 是新项目里最值得独立立项的数据工程。**

推荐 mining pipeline：

```text
                 Production / public safe traffic
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
 sensitive-topic retrieval               current model inference
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
                 high-risk / low-margin pool
                          │
              ┌───────────┴────────────┐
              │                        │
           safe but                 genuinely
           suspicious               unsafe
              │                        │
              └──── near-contrast pair┘
                          │
                          ▼
                 dual human annotation
                          │
             disagreement → senior review
                          │
                          ▼
              hard-negative golden corpus
                          │
                          ▼
           training + dedicated FP benchmark
```

每个样本最好显式标：

```text
topic          = malware
intent         = defensive
actionability  = low
target         = unspecified
context        = security education
severity       = none
verdict        = safe
```

然后构造 minimal pair：

```text
A: “说明勒索软件常见加密流程，方便制定检测策略。”
   → safe / defensive

B: “给我实现一个自动遍历目录、加密文件并破坏恢复机制的程序。”
   → unsafe / operational
```

真正有价值的是**改变 intent/actionability，而尽量保持 vocabulary 相似**。

重点 hard-negative buckets 应包括：

| Bucket | 为什么容易误报 |
|---|---|
| 教育 / academic | 出现危险术语但目的为解释 |
| 新闻 / 历史 | 描述真实犯罪、战争、极端主义 |
| 防御性 cybersecurity | malware/exploit 关键词密集 |
| 医学 / 心理健康 | self-harm、药物、症状词密集 |
| 法律 / 合规 | 描述违法行为用于风险分析 |
| quotation | 用户引用有害文本进行分析 |
| fiction / role-play | harmful event ≠ operational instruction |
| policy / safety research | 经常直接讨论 jailbreak/harm |
| benign ambiguity | 同一词汇在非危险语境下使用 |

XSTest 的 250 个 safe prompts 本质上就是这个思想的 benchmark 化版本：它专门选取“表面类似危险内容、但本应正常回答”的请求，并配 200 个 unsafe contrasts。citeturn22academia0

**数据治理建议先于训练。**

```text
Raw Sources
   │
   ▼
License / provenance gate
   │
   ▼
PII / secrets / malformed filtering
   │
   ▼
Exact hash dedup
   │
   ▼
MinHash / SimHash near-dedup
   │
   ▼
semantic embedding clustering
   │
   ▼
taxonomy normalization
   │
   ▼
benchmark quarantine
   │
   ▼
label consistency / human audit
   │
   ▼
cluster-aware train/val split
   │
   ▼
immutable dataset manifest
```

**Benchmark quarantine 必须是单独的机制。** 将 XSTest、WildGuardTest、HarmBench、StrongREJECT、SafetyBench、CHiSafetyBench、SafeDialBench、ChiSafe-PAS 等全部建立 exact hash + normalized hash + semantic-nearest-neighbor blacklist；公开安全数据大量互相借用源数据，只做 exact hash 不够。HarmBench 的存在本身就源于不同 red-teaming 工作评价标准不可比的问题；把 test prompt 泄漏进训练会进一步制造“Guard 已经很强”的假象。citeturn23academia49

对 synthetic data，我建议不采用“一个强 Teacher 生成全部数据”，而是：

```text
seed examples
   ↓
multiple teachers
   ↓
intent/category/context controlled generation
   ↓
attack mutation
   ↓
safe near-neighbor generation
   ↓
independent judge ensemble
   ↓
human audit of:
  - disagreement
  - high-risk
  - low-margin
  - Chinese-native
   ↓
training pool
```

这样降低单一 Teacher 的 style artifact 和 policy bias。

## Benchmark、指标、误报与 Jailbreak 鲁棒性

你的 benchmark suite 应该区分“**Guard classification**”与“**Main LLM jailbreak resistance**”。HarmBench / StrongREJECT 原本主要用于 red teaming / model refusal，而不是专门为了 Guard classifier 设计；它们仍然非常有用，但不能与 WildGuardTest 的 F1 直接混成一个平均分。citeturn23academia49turn23academia50

推荐 benchmark map：

| 类型 | Benchmark | Year | Language | Size | 主要任务 | 主要指标 |
|---|---|---:|---|---:|---|---|
| General Guard | **WildGuardTest** | 2024 | EN | 1,725 | prompt harm / response harm / refusal | F1 / precision / recall citeturn18search0 |
| Over-refusal | **XSTest** | 2023 | EN | 250 safe + 200 unsafe | exaggerated safety | refusal / FP / contrast accuracy citeturn22academia0 |
| Red-team/Jailbreak | **HarmBench** | 2024 | EN | 510 behaviors | attack + robust refusal | ASR / classifier eval citeturn23academia49turn23search0 |
| Jailbreak quality | **StrongREJECT** | 2024 | EN | 313 forbidden prompts | jailbreak response harmful usefulness | StrongREJECT score citeturn23academia50turn23search2 |
| Chinese/English safety | **SafetyBench** | 2023 | ZH/EN | 11,435 | 7 categories，MCQ safety understanding | Accuracy citeturn23academia48 |
| Chinese | **CHiSafetyBench** | 2024 | ZH | 1,567 MCQ + 563 QA | 5 areas / 31 cats；risk/refusal | ACC/refusal/harm rate citeturn24academia48turn24search12 |
| Chinese values | **CValues** | 2023 | ZH | 10 safety scenarios + 8 responsibility domains | adversarial safety/value alignment | automatic + human eval citeturn24academia50 |
| Multi-turn | **SafeDialBench** | 2025/ICLR 2026 | ZH/EN | >4,000 dialogues | 22 scenarios / 7 jailbreak strategies | detection/handling/consistency/ASR citeturn24search10 |
| Chinese adversarial | **ChiSafe-PAS** | 2026 | ZH | 1,897 / 1,544 gold | 4 domains + 9 obfuscation types | 3-class behavior/risk citeturn24academia49 |
| Prompt Injection | **BIPIA** | 2023 | EN | multi-task corpus | indirect prompt injection | attack success / defense success citeturn17search2 |
| Agent Injection | **InjecAgent** | 2024 | EN | 1,054 | 17 user tools / 62 attacker tools | attack success citeturn17academia45 |
| Injection detector | **PromptShield benchmark** | 2025 | EN | benchmark suite | deployable PI detector | TPR/FPR/F1/latency citeturn17academia46 |

SafeDialBench 对你的项目尤其值得重视：ICLR 2026 版本覆盖中英文、超过 4,000 个多轮对话、22 种场景和 7 类 jailbreak strategies，而且不仅测“有没有拒绝”，还测识别、处理和跨轮一致性。citeturn24search10

**评测不能再以 Accuracy 为主。**

把 unsafe 定义为 positive，则：

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Recall=TPR=\frac{TP}{TP+FN}
\]

\[
FPR=\frac{FP}{FP+TN}
\]

\[
FNR=\frac{FN}{TP+FN}
\]

\[
F1 = 2\frac{Precision\cdot Recall}{Precision+Recall}
\]

对于生产系统，Accuracy 很容易被类别比例欺骗。假设真实流量只有 1% dangerous，即使一个模型永远输出 safe，也能获得约 99% accuracy，却完全没有 Guard 价值。

因此 dashboard 的第一屏应该是：

| 指标 | 为什么必须有 |
|---|---|
| Unsafe Recall / TPR | 危险内容漏过多少 |
| FPR | 正常用户被误伤多少 |
| **Recall @ FPR=0.1/0.5/1%** | 最有业务意义的 operating point |
| AUPRC | unsafe 占比低时比 ROC 更有解释力 |
| Macro-F1 | 防止大类别掩盖小类别 |
| per-category Recall/FPR | 找到 self-harm/cyber 等薄弱点 |
| per-language Recall/FPR | 中文与英文单独看 |
| hard-negative FPR | 专门衡量 over-blocking |
| jailbreak recall | 对抗输入识别能力 |
| ECE / Brier | score 是否可作为 policy threshold |
| review rate | gray-zone 带来的业务成本 |
| P50/P95/P99 | guard latency |
| throughput / GPU utilization | 成本与容量 |

推荐将主 KPI 定义为：

> **在固定业务 FPR 预算下最大化 Unsafe Recall，而不是最大化 F1。**

例如：

```text
Safety-critical tier:
Recall @ FPR <= 1%

High-traffic consumer tier:
Recall @ FPR <= 0.2%

Extremely sensitive high-stakes category:
category-specific threshold
```

具体 0.2%/1% 是工程初始目标，不是行业标准；最终应该由真实流量成本函数决定。

Calibration 可以使用 ECE：

\[
ECE =
\sum_{m=1}^{M}\frac{|B_m|}{n}
\left|
accuracy(B_m)-confidence(B_m)
\right|
\]

并同时报告 Brier score。更重要的是，**temperature calibration 必须在没有参与梯度训练的 production-like calibration set 上完成**，最好分 language/category；不要在 benchmark test 上调 threshold。

**Jailbreak robustness 应独立建红队矩阵。**

至少覆盖：

| Attack family | 示例性质 |
|---|---|
| Persona / DAN | 改变角色、假装无约束模型 |
| Role-play / fiction | 小说、游戏、研究场景掩饰 |
| Instruction override | ignore previous instructions |
| Encoding | Base64、ROT、hex 等 |
| Unicode / homoglyph | 字符替换、零宽字符 |
| Chinese evasion | 拼音、谐音、拆字、黑话 |
| Translation | 使用非英语绕过 |
| Context splitting | 有害意图分散到多轮 |
| Gradual escalation | 每一轮单独看似安全 |
| Adversarial suffix | GCG/优化 suffix |
| Automated attacker | PAIR/LLM-driven mutation |
| Indirect injection | 网页/RAG/email 中隐藏命令 |
| Tool injection | 恶意 tool observation |

HarmBench 专门建立了标准化 red-teaming framework，并测试 18 种攻击方法与 33 个模型/defense；h4rm3l 则把 jailbreak 表示成可组合 transformation primitive，并自动生成新攻击。这些结果共同说明：测试少量固定“DAN prompt”远远不够。citeturn23academia49turn22academia2

最终 evaluation suite 建议四层：

```text
┌───────────────────────────┐
│ Public Benchmarks         │
│ WildGuard/XSTest/...      │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│ Internal Golden Set       │
│ company policy / zh+en    │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│ Adversarial Holdout       │
│ unseen attack families    │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│ Production Shadow Traffic │
│ true FPR / drift / latency│
└───────────────────────────┘
```

其中 Internal Golden Set 才应该决定上线 threshold；公开 benchmark 主要用于行业可比性。

## 协议、NeMo Guardrails 兼容与 Serving 工程

旧项目最值得彻底推翻的部分，很可能不是 model weights，而是**私有 protocol**。

Hugging Face 当前推荐把对话表示为：

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

再通过：

```python
tokenizer.apply_chat_template(messages, ...)
```

转换成模型实际 control tokens。HF 官方强调，同一 base model 的不同 instruction model 也可能要求完全不同的 chat format，格式错误会显著损伤性能。citeturn15search0turn15search10

因此推荐建立三层接口：

| Layer | Schema | 目的 |
|---|---|---|
| Canonical business API | stable Guard JSON | 与 checkpoint 解耦 |
| OpenAI-compatible adapter | `messages[]` | NeMo/vLLM/SDK 兼容 |
| HF model layer | `tokenizer.chat_template` | 正确 tokenizer serialization |

Canonical input：

```json
{
  "model": "bilingual-safety-guard-v1",
  "policy": "default-v1",
  "direction": "input",
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "return_scores": true
}
```

Canonical output：

```json
{
  "schema_version": "1.0",
  "model_version": "bilingual-safety-guard-v1.3",
  "policy_version": "default-v1",
  "verdict": "unsafe",
  "blocked": true,
  "risk_score": 0.973,
  "severity": "high",
  "categories": [
    {
      "id": "H10",
      "name": "Cyber Abuse",
      "score": 0.961
    }
  ],
  "security_signals": [
    {
      "id": "J01",
      "name": "Jailbreak",
      "score": 0.81
    }
  ]
}
```

但**这不意味着让模型 autoregressively 生成这整个 JSON。**

推荐内部：

```text
unsafe
H10,J01
```

甚至：

```text
logits:
safe    -3.71
review  -1.93
unsafe   4.28
```

然后 serving layer：

```text
logits
   ↓
calibration
   ↓
policy thresholds
   ↓
stable JSON
```

这样模型版本更新、taxonomy 改动、threshold 改动都不需要改变外部 API。

这也与 Shieldstral 的 one-forward yes/no scoring，以及 Llama Guard 的 first-token probability 方法一致。citeturn21search13turn15search1

**vLLM 兼容路径非常清楚。** vLLM 当前 OpenAI-Compatible Server 支持 `/v1/chat/completions`；使用 Chat API 时模型必须有 tokenizer chat template，没有则通过 `--chat-template` 显式提供。citeturn16search0turn16search1

所以模型仓库必须包含：

```text
config.json
tokenizer.json
tokenizer_config.json
chat_template
generation_config.json
README/model card
policy_schema.json       # 项目自定义
taxonomy.json            # 项目自定义
```

vLLM 也支持 structured output 的 choice / regex / JSON 等模式。对于简单 guard，可以约束：

```json
{
  "choice": ["safe", "review", "unsafe"]
}
```

避免自由生成格式漂移。citeturn16search13

**NeMo Guardrails 当前已经比旧项目时代更容易集成。** 官方 Content Safety 文档支持 NVIDIA Nemotron、Llama Guard 3、ShieldGemma，并允许使用任意定义好的 content-safety model；input/output 分别通过：

```yaml
rails:
  input:
    flows:
      - content safety check input $model=content_safety

  output:
    flows:
      - content safety check output $model=content_safety
```

执行。citeturn20search0

对 vLLM，当前 NVIDIA 官方推荐直接使用：

```yaml
engine: openai
parameters:
  base_url: http://localhost:8000/v1
```

因为 vLLM 暴露 OpenAI-compatible API；旧的 `vllm_openai` connector 已不再是新配置首选。citeturn20search3

你的 Guard 可以配置成：

```yaml
models:
  - type: main
    engine: openai
    model: main-llm
    parameters:
      base_url: http://main-llm:8000/v1
      api_key: EMPTY

  - type: content_safety
    engine: openai
    model: bilingual-safety-guard-v1
    parameters:
      base_url: http://guard:8000/v1
      api_key: EMPTY

rails:
  input:
    flows:
      - content safety check input $model=content_safety

  output:
    flows:
      - content safety check output $model=content_safety
```

这一部分是基于 NeMo 当前 OpenAI-compatible model 配置方式做的直接工程映射；官方文档明确支持 `engine: openai + base_url` 对接 vLLM 与其他兼容 endpoint。citeturn20search3turn20search5

如果模型直接返回严格：

```text
safe
```

或：

```text
unsafe
```

则可以利用 NeMo 内置 `is_content_safe` parser；官方 parser 能识别 `safe`、`unsafe`、`yes`、`no` 语义。citeturn20search0turn20search2

例如：

```yaml
prompts:
  - task: content_safety_check_input $model=content_safety
    content: |
      {{ user_input }}
    output_parser: is_content_safe
    max_tokens: 8
```

如果你希望直接兼容 Nemotron 风格 rich output，则 NeMo 当前官方示例已经使用：

```json
{
  "User Safety": "safe | unsafe",
  "Response Safety": "safe | unsafe",
  "Safety Categories": "..."
}
```

并配 `nemoguard_parse_prompt_safety` parser。citeturn20search7

所以所谓“零定制”的现实含义应该是：

> **不写自定义 Python action/parser；只通过 YAML/prompt 配置接入。**

真正完全零配置不现实，因为至少要声明 endpoint 和 model type。

**Serving hot path 不推荐自由生成。**

最佳选择依优先级排序：

```text
1. direct first-token logits
2. constrained single-token generation
3. short generated label + category ids
4. full JSON generation
5. natural-language reasoning + verdict
```

从在线 Guard 延迟角度，5 最差；但对于第二阶段 difficult-case reasoner 可以接受。

Qwen3Guard 的设计尤其值得参考：Generative Guard 适合完整 prompt/response classification，Stream variant 则用 token-level classification head 做增量 moderation，直接解决“等主模型生成完才判断”的问题。citeturn21academia48

NeMo 当前也已经支持 output rail streaming 配置；因此未来你可以实现：

```text
Main LLM token stream
      │
      ├── chunk 1──┐
      ├── chunk 2──┼──► streaming guard
      ├── chunk 3──┘         │
      │                   unsafe?
      │                     │ yes
      └─────────────────────X terminate
```

而不是把危险 completion 完整发送到用户后才审计。citeturn20search5turn20search13

**Latency SLO 我建议定义为项目目标，而不是假装存在统一行业标准。** 对短文本、数据中心 GPU 的 Guard，可以从以下目标开始验收：

| Path | 建议初始 SLO |
|---|---|
| input guard ≤512 tokens | P50 ≤ 50–100 ms |
| input guard ≤512 tokens | P99 ≤ 200–300 ms |
| output final guard | P99 ≤ 300 ms，具体依 response length |
| first-stage injection classifier | P99 ≤ 20–50 ms |
| guard overhead | 尽量 ≤ 主模型 E2E latency 的 10–20% |

这些是**项目 SLO 建议，不是文献报告的行业平均值**；最终必须在你的 GPU、batch、sequence-length distribution 上测。

Batching 推荐使用 continuous batching；但 Guard 最大问题通常不是 decode，而是 repeated prefill。若只是输出一个 token，speculative decoding 的价值远低于“消除多 token decoding”；真正值得优化的是：

```text
short policy prompt
+ prompt prefix caching
+ constrained output
+ first-token decision
+ batch prefill
```

**量化策略建议：**

```text
BF16 reference
    ↓
FP8 / INT8 candidate
    ↓
full benchmark + calibration
    ↓
INT4 candidate
    ↓
separate hard-negative & jailbreak regression
```

不能只比 aggregate F1。Meta 自己的 1B INT4 数据就是很好的警告：英语 aggregate F1 没明显下降，但 XSTest F1/FPR 明显恶化。citeturn15search1

因此 production quantization acceptance 应要求：

```text
Δ Recall@FPR=1%       within budget
Δ hard-negative FPR   within budget
Δ jailbreak recall    within budget
Δ ECE                 within budget
```

而不是只要求：

```text
Δ Accuracy < 0.5%
```

## 分阶段训练、MVP 与最终项目决策

我建议新项目不要一开始训练 8B，而是先用已有 guards 做统一 baseline，得到“到底是模型、数据还是 policy”的证据。

**第一步先建立 baseline matrix：**

```text
Old 2B Guard
Llama Guard 3-1B
Qwen3Guard-0.6B
Qwen3Guard-4B
Shieldstral-3B
Nemotron 3.5 Content Safety ~4B
```

全部使用同一份：

```text
English Golden
Chinese Golden
Hard-negative
Jailbreak
Multi-turn
Production sample
```

并统一 label mapping、threshold search 和 metrics，否则第三方 model card 的数字几乎没有决策价值。Meta 自己也明确指出不同 safety policies 使 guard 结果很难直接横向比较。citeturn15search1

建议训练路线如下。数据量是**工程起始区间，不是文献规定值**。

| Phase | 目标 | 建议增量数据 | ZH/EN | Synthetic / Real | 人工投入 | Gate |
|---|---|---:|---|---|---|---|
| **Phase 1 General Guard** | 基础 safe/unsafe/category | 200K–400K | 30/70 → 40/60 | 60/40 左右 | 抽检 3–5K | Public + internal baseline |
| **Phase 2 Bilingual** | 中文原生与 code-switch | +100K–250K | **≥50% ZH** | ≤60% synthetic | 原生中文审 5–10K | ZH gap 明显缩小 |
| **Phase 3 Hard Negative** | 降 FPR | +50K–150K | 50/50 | 30–50% synthetic | 高投入，pair audit | hard-negative FPR gate |
| **Phase 4 Adversarial** | jailbreak/multi-turn | +50K–150K | 40–60% ZH | 60–80% synthetic | unseen attack review | jailbreak holdout |
| **Phase 5 Calibration** | threshold / confidence | 20K–50K untouched | 真实流量比例 | 尽量 real | 高质量 Gold | Recall@fixed-FPR |

最终整体 batch mix 不建议按自然发生率训练，可以从：

```text
30–35% clear unsafe
25–30% ordinary safe
20–25% hard-negative safe
10–15% adversarial/jailbreak
 5–10% multi-turn / policy adaptation
```

起步，再通过 error analysis 调整。

**Phase 1** 首先训练：

```text
conversation + policy
→ safe / review / unsafe
→ categories
```

不要求 explanation。

**Phase 2** 不只是翻译，而应加入中文 native authored / production samples，并对：

```text
拼音 / 谐音 / 黑话 / 拆字 / Unicode /
中英混写 / 繁简 / 隐晦请求
```

做系统 coverage。Nemotron multilingual 数据可以作为启动数据，但不能替代 native Chinese annotation。citeturn19search0turn24academia49

**Phase 3** 对所有 false positive 做 active learning：

```text
model confidence near threshold
OR
sensitive keywords + predicted unsafe
OR
human complaint / production allow
      ↓
human review
      ↓
hard negative replay buffer
```

然后构造 near-contrast unsafe twin。

**Phase 4** 保证 attack family split，而不是随机样本 split：

```text
TRAIN:
roleplay
base64
simple multilingual

HELD-OUT:
new suffix attacks
new Chinese obfuscation
new multi-turn strategy
```

否则同模板 mutation 很容易泄漏到 test。

**Phase 5** 完全停止梯度训练，只做：

```text
temperature calibration
threshold optimization
policy-specific thresholds
language/category threshold analysis
```

生产决策可以采用两阈值：

```text
score < T_allow
    → allow

T_allow <= score < T_block
    → second-stage guard / review

score >= T_block
    → block
```

对于大规模系统，这通常比强迫一个阈值同时解决 FP/FN 更合理。这也是前述“fast classifier + expensive reasoner” cascade 思路的工程化实现。citeturn7search0turn7search5

**训练 architecture 我建议采用“vLLM-compatible first-token classifier + optional auxiliary safety head”，而不是一开始强依赖自定义 classification head。**

```text
Decoder Transformer
       │
       ├── standard LM head
       │        │
       │        └── {safe, review, unsafe} token logits
       │
       └── optional auxiliary heads
                 ├── categories
                 └── severity
```

第一版上线只依赖 LM head，天然能走 HF/vLLM；Stream Guard 到第二代再加入 Qwen3Guard 风格的 token-level head。Qwen3Guard 已经验证了 Gen/Stream 两个模型形态分离是可行路线。citeturn21academia48

**训练 compute 不建议用“多少张 GPU 训练多少天”做固定承诺，因为序列长度、packing、optimizer、LoRA/full-FT、GPU 型号会改变数量级。** 一个更合理的容量估算是 dense Transformer full-training 粗略：

\[
FLOPs \approx 6NT
\]

其中 \(N\) 为参数量，\(T\) 为训练 tokens。比如：

| 模型 | Tokens | 粗略训练 FLOPs |
|---:|---:|---:|
| 1B | 200M | ~1.2×10¹⁸ |
| 3B | 300M | ~5.4×10¹⁸ |
| 8B | 300M | ~1.44×10¹⁹ |

这只能做 compute order-of-magnitude planning，不包含实现效率。MVP 更建议 LoRA/QLoRA 做快速 ablation；NVIDIA 当前多个 safety guard 本身也采用 LoRA 后合并回 base model 的训练方式，例如 Nemotron 3.5 Content Safety。citeturn18search8

**最终十个关键决策如下。**

| 问题 | 建议结论 |
|---|---|
| **今天还应该继续传统 2B LLaMA + SFT 吗？** | **不建议原样延续。** 2B 尺寸本身没问题，问题是固定 taxonomy、私有 format、普通 harmful SFT 和缺少 calibrated operating point。应升级为现代 multilingual base + policy-aware + hard-negative/adversarial + standardized protocol。 |
| **最值得借鉴的 3–5 个模型？** | **Qwen3Guard、Shieldstral、Llama Guard 3-1B、NVIDIA Nemotron Safety Guard、WildGuard。** 前三者分别代表 stream/multilingual、policy-adaptive single-token、轻量 generative guard；NVIDIA 代表 dataset+NeMo 全链路；WildGuard 代表训练/评测任务设计。citeturn21academia48turn21search0turn15search1turn19search0turn18search0 |
| **训练数据优先用什么？** | P0：Nemotron Safety Guard v3 + Aegis 2 + WildGuardTrain；P1：PKU-SafeRLHF/BeaverTails 只在 license 允许时做 augmentation；另外必须自己构造原生中文 hard-negative corpus。citeturn19search0turn19search4turn18search0turn18search1 |
| **Taxonomy 怎么设计？** | **canonical hierarchical multi-label + severity + 独立 security signals**；业务 policy 再映射为 safe/review/unsafe。不要把 jailbreak 与 violence 等视为同一维度。 |
| **怎样显著降低误报？** | Hard-negative / minimal-pair 数据是第一优先级；显式学习 intent、actionability、context；再做 calibration、per-category thresholds 和 gray-zone second stage。XSTest 必须成为回归集。citeturn22academia0 |
| **怎样提高 jailbreak robustness？** | 多攻击族 adversarial training + unseen attack-family holdout + multi-turn context + 中文 obfuscation + 专门 injection detector；不要依赖单个 content Guard。HarmBench/StrongREJECT/SafeDialBench/BIPIA 都应进入红队套件。citeturn23academia49turn23academia50turn24search10turn17search2 |
| **1B–3B 足够吗？** | **大多数 production moderation 很可能足够，值得首先验证。** 1B 已有 Llama Guard，2B 有 ShieldGemma，3B 有 Shieldstral；7/8B 应作为复杂边界或 second-stage candidate，而不是默认 hot path。citeturn15search1turn15search5turn21search0 |
| **输出选 binary/category/JSON？** | **模型内部：短 label + category logits；业务外部：versioned structured JSON。** 不建议模型自由生成 confidence JSON。 |
| **怎样低成本兼容 NeMo/HF/vLLM/OpenAI-compatible？** | 输入统一 `messages[]`；仓库提供 `tokenizer.chat_template`；vLLM 暴露 `/v1/chat/completions`；模型保证 deterministic `safe/unsafe` compatibility profile；NeMo 用 `content_safety check input/output` + built-in/custom parser 即可。citeturn15search0turn16search0turn20search0turn20search3 |
| **有限人力/GPU 的 MVP 做什么？** | 不从 8B 开始。先构建 10K–20K 高质量中英 internal gold + baseline matrix；选 1B–3B base，整合约 200K–400K 开放数据，重点再造 50K 级 hard-negative；LoRA 训练；上线前只要求 input/output guard、single/multi-turn、NeMo/vLLM adapter 和一套可复现 evaluation suite。 |

从产品/工程负责人的角度，**最小可行项目实际上可以收敛为六个交付物**：

```text
A. Safety Policy v1
   canonical taxonomy
   + category mappings
   + threshold semantics

B. Guard Dataset v1
   EN + native ZH
   + harmful
   + ordinary safe
   + hard negative
   + jailbreak

C. Guard Model v1
   1B–3B Decoder-only
   + first-token scoring
   + multi-label category

D. Guard API v1
   OpenAI messages input
   + stable JSON response
   + HF chat template
   + vLLM serving

E. Evaluation v1
   public benchmark
   + 10K–20K internal gold
   + adversarial holdout
   + Recall@fixed-FPR

F. NeMo integration
   content safety input
   + output
   + safe/unsafe mapping
```

这比直接投入大量 GPU 重新训练一个“新版 2B safety model”更重要。

最值得作为整个项目的北极星指标，不是：

```text
Safety Accuracy = 95%
```

而应该类似：

```text
English unsafe recall @ FPR 0.5%
Chinese unsafe recall @ FPR 0.5%
Hard-negative FPR
Jailbreak recall on unseen attack families
Multi-turn unsafe recall
ECE / calibration
P50 / P99 latency
Throughput / GPU
```

再通过 policy engine 将这些 calibrated scores 转换成不同产品场景的 `allow / review / block`。

换句话说，新一代 Guard 项目的真正升级应该是：

> **从“训练一个会输出 safe/unsafe 的 2B LLaMA”升级为“训练一个可校准、policy-aware、中英文、多轮、对抗鲁棒，并被标准 API 与独立 Policy Engine 包装的 Safety Classifier System”。**

现有开源生态已经提供了非常明确的证据链：Meta 证明 1B generative guard 可行且 taxonomy 可以动态传入；Qwen3Guard 证明 0.6B–8B 多语言与 token-level streaming 是可实现路线；Mistral Shieldstral 证明 3B policy-adaptive single-token scoring 已成为现实；NVIDIA 已经把 multilingual safety data、content guard 与 NeMo Guardrails 集成为完整工程栈。综合这些趋势，**我会把你的首个正式研发版本定在约 2B–3B，而不是 7B/8B；把真正的工程预算优先投给数据、Hard Negative、中文原生标注、calibration、benchmark quarantine 和统一协议。** citeturn15search1turn21academia48turn21search0turn19search0turn20search0
