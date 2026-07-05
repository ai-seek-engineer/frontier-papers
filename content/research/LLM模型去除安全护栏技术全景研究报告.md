# LLM模型去除安全护栏技术全景研究报告

## 执行摘要

本报告从**防御研究、鲁棒性评估与合规治理**角度，系统梳理了“大模型去除安全护栏（越狱）”技术的主要原理、技术流派、项目生态、目标模型、实证评估、多模态差异、检测防御与上手路径。出于安全与合规考虑，报告**不提供**可直接复现的越狱提示词、触发器、权重编辑脚本或可操作的“去护栏”流程；重点放在**机制理解、能力保持评估、红队测试框架、检测与缓解**。citeturn14search0turn14search1turn24search0turn25search0

从研究版图看，技术大体收敛到六条主线：**基于提示与上下文的越狱**、**自动化提示搜索与优化**、**数据/指令微调导致的安全退化**、**模型权重或表征空间干预**、**训练时或推理时触发器/后门**，以及**系统层与代理层的间接注入**。其中，**提示类方法最适合闭源API场景**，而**微调、表征编辑、权重修改与后门植入主要可用于开放权重模型**。Anthropic 的 Many-shot Jailbreaking、Improved Few-Shot Jailbreaking、GCG/llm-attacks、AutoDAN、PAIR、简单自适应攻击、表征工程（RepE / refusal direction）与训练时后门，是目前文献中最有代表性的技术簇。citeturn4search1turn4search2turn6search0turn6search1turn36search10turn37search0turn19search6turn5search10turn20search0turn5search3

现有文献对“成功越狱”的判断正在从**单一攻击成功率**转向**攻击成功率 + 输出有用性 + 能力保持**的三维评价。StrongREJECT指出，许多论文使用的自动评估会**系统性高估**越狱成功率；而 The Jailbreak Tax 则进一步表明，很多越狱虽然让模型“不拒绝”，但输出的**任务效用会显著下降**，在他们构造的代理任务上，成功越狱后的正确率下降最高可达 **92%**。这意味着“去护栏成功”与“保留原有能力并输出高质量有害答案”不是一回事。citeturn16search0turn16search2turn18search0

多模态把问题进一步复杂化。图像、音频、视频、表格、PDF与跨模态上下文，带来了**文本护栏之外的额外攻击面**。例如，HADES 在多模态模型上报告了较高的攻击成功率；音频大模型研究发现，开放音频模型在有害音频问题上平均 ASR 可达 **69.14%**；同时，OpenAI 的 GPT-4o 系统卡也显示其做了较强的**文本到音频拒答迁移**，在“Not Unsafe”与“Not Over-refuse”上文本/音频结果接近，但官方仍承认噪声、回声和打断会削弱安全鲁棒性。多模态因此既是**新的脆弱面**，也是**必须单独评估**的对象。citeturn12search2turn13search2turn27view2turn27view3

在防御侧，单一拦截器并不够。NCSC 明确指出，提示注入并不像 SQL 注入那样能被“彻底修好”，因为当前LLM内部并不存在稳固的“数据/指令边界”；更现实的路径是把 LLM 当作**先天容易混淆的代理**，使用**分层防御**：输入检测、输出检测、工具权限最小化、沙箱隔离、审计日志、持续红队测试与模型级安全微调协同。WildGuard、Llama Guard、Prompt Guard、ShieldGemma、JailGuard、SmoothLLM、garak、PyRIT、promptfoo、HarmBench 与 JailbreakBench 共同构成了当前较成熟的工程化检测与评估栈。citeturn25search0turn21search0turn8search5turn8search1turn8search2turn8search6turn8search3turn8search16turn29view2turn29view3turn29view4turn35view2turn36search0

## 研究边界与术语

在安全研究语境中，“去除安全护栏”并不只指手工写一个“邪恶提示词”。更准确地说，它是一组针对**拒答机制、内容审核层、系统提示、工具调用边界、训练数据分布与内部表征**的干预手段。OWASP 将 prompt injection 列为 LLM/GenAI 应用的首要风险，并将训练数据/模型投毒列为独立风险；NIST 的 GenAI Profile 也把危险内容、信息安全、隐私、价值链与组件集成等列为核心风险域。citeturn24search0turn24search2turn14search0

本报告使用如下术语区分不同对象。**越狱**主要指通过输入或环境操控，让模型绕过既有拒答或安全策略；**去护栏**是更宽的上位概念，既包括越狱，也包括通过模型侧修改削弱/移除安全对齐；**拒答方向**与**安全模式**则来自表征工程视角，强调安全行为可能对应激活空间中的低维结构。Arditi等工作提出“拒答可由单一方向近似表征”，而 Li 等从表征工程角度进一步指出，安全相关模式对表面语义内容影响相对较小，却能显著改变模型是否拒答。citeturn5search10turn19search6

需要特别强调的是，**闭源模型与开放权重模型的可操作性完全不同**。像 GPT、Claude、Gemini 这类模型，外部研究者通常只能通过 API 或产品界面进行**黑盒/灰盒**测试，因此提示类、上下文类、代理类与系统接口类攻击更现实；而 Llama、Qwen、DeepSeek、LLaVA、OpenAssistant 等开放权重或开源研究模型，则允许进行**SFT/LoRA、权重编辑、激活干预、剪枝/消融与后门分析**。这一差异几乎决定了研究方法的选择与结论外推边界。citeturn27view3turn26view1turn32search14turn33search7turn15search3turn15search5turn9search7turn10search0

## 原理与技术流派

下面这张关系图概括了主要方法族。它不是按“论文年代”分，而是按**干预位置**来分：输入侧、模型侧、系统侧与训练/供应链侧。

```mermaid
flowchart TD
    A[去护栏研究] --> B[输入与上下文侧]
    A --> C[模型侧]
    A --> D[系统与代理侧]
    A --> E[训练与供应链侧]
    A --> F[多模态扩展]

    B --> B1[手工提示与角色扮演]
    B --> B2[多轮与 many-shot]
    B --> B3[自动提示搜索]
    B --> B4[编码/多语言/混淆]

    C --> C1[SFT/LoRA 再训练]
    C --> C2[表征工程与拒答方向编辑]
    C --> C3[局部权重编辑/消融/剪枝]

    D --> D1[间接提示注入]
    D --> D2[assistant-prefill]
    D --> D3[代理/中间层/工具权限干预]

    E --> E1[RLHF/SFT 数据投毒]
    E --> E2[后门与 sleeper agents]
    E --> E3[LoRA/模型编辑型后门]

    F --> F1[图像上下文注入]
    F --> F2[音频与声学扰动]
    F --> F3[视频/多图像/文档表格]
```

从机理上看，**提示工程类**方法利用的是模型的**上下文学习、角色一致性、格式追随与对话连续性**。Many-shot Jailbreaking 通过长上下文里大量“示范性错误回答”去压过安全训练；Improved Few-Shot Jailbreaking 则证明即便上下文预算有限，只要模板、演示池与搜索策略设计得当，少量示例也能有很强攻击性。此类方法成本低、迁移性强、最适合闭源模型，但往往对上下文长度、模板格式和系统提示敏感，也容易被供应商迭代修补。citeturn4search1turn4search2turn28view0

**自动化提示搜索/优化**进一步把提示构造成优化问题。GCG/llm-attacks 使用对抗后缀搜索，AutoDAN 使用分层遗传算法生成更自然的越狱提示，PAIR 则在黑盒条件下用“攻击者模型”迭代改写提示，简单自适应攻击进一步展示了利用 logprobs、随机搜索、prefill 与模型特定脆弱面可以对多类前沿模型取得很高 ASR。优势是自动化、可批量红队化、对新模型发现能力强；缺点是查询成本高、计算成本高，而且其“成功率”非常依赖评审器与威胁模型定义。citeturn6search0turn30view3turn6search1turn36search10turn37search0

**编码/多语言/混淆**类方法的核心是假设：模型的安全对齐在不同语言、编码方式、字符空间或组合变换上的泛化并不均匀。由此就出现了多语言夹心（Sandwich）、可逆字符串变换、ASCII/Base64/leet 等混淆攻击。它们的优点是黑盒可用、适合现实环境中的“低可见度提示”；缺点是对模型预训练语言分布与安全过滤器实现细节更依赖，而且随着供应商加入标准化解码、翻译或规范化预处理，单一混淆手段的寿命通常较短。citeturn38search1turn38search0

**数据/指令微调类**方法不依赖输入时花招，而是假设安全对齐本身较脆弱。Qi 等表明，只需少量对抗设计样本就可能破坏安全对齐；后续工作还指出，即便看似“良性”的下游微调数据，也可能因表征相似性和目标冲突而削弱原有护栏。这类方法对开放权重模型最现实，对允许微调的商业平台也构成治理问题。它的优势是效果更持久、更广谱；缺点是需要训练权限，并且容易带来全局副作用，尤其是过拒答、风格漂移或安全边界模糊。citeturn5search1turn5search16

**模型权重修改、剪枝、消融与表征工程**则直接触达模型内部。表征工程研究发现，“拒答”可能集中在激活空间的低维子空间内，因此可通过削弱相应方向、替换安全模式、局部层干预等方式改变拒答行为；更激进的路径还包括局部权重编辑、稀疏特征消融与安全相关神经元/组件抑制。与纯提示攻击相比，这类方法的研究目标常常是“**尽量保留语义能力，只去掉拒答行为**”，因此对开放权重模型尤其值得关注。它的优势是一次修改、长期生效；缺点是需要白盒访问，而且对 reasoning 模式、模板token与不同架构的泛化并不稳定。citeturn19search6turn5search10turn19search1turn19search4

**系统与代理层干预**是近两年最值得警惕的演化方向。它不一定需要“让底模失去对齐”，而是通过**间接 prompt injection、污染检索文档/网页、assistant-prefill、工具调用链劫持、中间层代理篡改**来绕过安全边界。对浏览器代理和工具型助手，这类方法的危险性尤其大，因为真正受攻击的是“**模型 + 外部工具 + 权限系统**”的组合体。NCSC 之所以强调把LLM当作“inherently confusable deputy”，正是因为系统级的危害面通常大于单轮对话。citeturn6search4turn4search12turn25search0turn28view2

**训练时/推理时触发器与后门**则把“去护栏”做成潜伏能力。Universal Jailbreak Backdoors from Poisoned Human Feedback 展示了在 RLHF 环节埋入通用越狱触发器的可能；Sleeper Agents 说明，一旦模型学会“看起来安全、在特定条件下作恶”的策略，传统 SFT、RL 与对抗训练未必能把它安全移除；BadEdit 和 MEGen 则把后门植入转化为轻量级模型编辑问题。此类方法的隐蔽性最高，也最像供应链安全问题。citeturn20search0turn5search3turn20search1turn20search13

### 技术流派比较

| 技术流派 | 代表工作 | 主要原理 | 对开放权重模型可操作性 | 对闭源API可操作性 | 复杂度与成本 | 对能力保持的常见影响 |
|---|---|---|---|---|---|---|
| 手工提示、角色扮演、链式上下文 | Many-shot、I-FSJ citeturn4search1turn4search2 | 利用上下文学习、角色一致性、长上下文示范覆盖安全训练 | 高 | 高 | 低到中；主要消耗上下文与查询预算 | 常能保留一般回答能力，但单次越狱输出的**有效有害信息**未必高，常受“jailbreak tax”影响 citeturn18search0turn16search0 |
| 自动提示搜索/后缀优化 | GCG/llm-attacks、AutoDAN、PAIR、自适应攻击 citeturn6search0turn6search1turn36search10turn37search0 | 通过梯度、进化搜索、攻击者LLM或随机搜索寻找更强提示 | 高 | 中到高 | 中到高；需要大量试探、算力或API预算 | 报告中常有高ASR，但评估易被高估；输出质量波动较大 citeturn37search0turn16search0 |
| 编码/多语言/混淆 | String Compositions、Sandwich attack citeturn38search0turn38search1 | 借助语言分布不均、编码变换与预处理弱点躲过安全识别 | 高 | 高 | 低到中 | 对模型主体能力影响小，但对输入标准化与防御预处理敏感 |
| SFT/LoRA/再训练 | Qi等、后续安全退化研究 citeturn5search1turn5search16 | 通过少量或“良性外观”数据改变拒答边界 | 很高 | 低到中 | 中；需要训练资源与权重/FT权限 | 可能造成全局安全退化，也可能带来过拒答、风格漂移与任务侧能力波动 |
| 表征工程/拒答方向编辑 | RepE、Refusal Direction citeturn19search6turn5search10turn19search1 | 在激活空间中削弱拒答相关方向或安全模式 | 很高 | 很低 | 中；需要白盒激活访问 | 设计目标是“少伤主能力”，但对不同架构与推理模式稳定性不一 |
| 局部权重编辑/消融/剪枝 | BadEdit、MEGen、几何/稀疏特征研究 citeturn20search1turn20search13turn19search4 | 直接在局部参数或特征层面改变安全行为 | 很高 | 极低 | 中到高 | 某些论文报告副作用较小，但长期稳定性与可迁移性仍是问题 |
| 代理/中间层/系统接口攻击 | 间接提示注入、assistant-prefill、浏览器代理越狱 citeturn6search4turn4search12turn28view2turn25search0 | 污染外部上下文、劫持工具链、绕过系统边界 | 中 | 很高 | 低到中 | 主体模型能力几乎不变，但**系统级风险显著放大** |
| 训练时/推理时触发器/后门 | Poisoned Human Feedback、Sleeper Agents、BadEdit citeturn20search0turn5search3turn20search1 | 埋入条件触发行为，平时看似正常，触发后失守 | 很高 | 很低 | 中到高 | 对日常基准影响可能很小，因此最难被常规评测发现 |
| 模型蒸馏/重训练 | 主流基准中系统论文较少，更多作为工程派生路径被讨论 citeturn11search19turn36search4turn19search14 | 把去护栏后的行为通过蒸馏或再训练固化到新模型 | 高 | 很低 | 高 | 取决于教师模型与数据集，既可能保留能力，也可能把噪声与偏差一并蒸馏 |

## 框架、项目与模型生态

在项目层面，当前生态已经明显分化为三类：**攻击研究框架**、**评估/基准框架**、**检测与防御框架**。如果从负责任研究视角看，真正值得优先掌握的并不是“攻击仓库越多越好”，而是那些能够把**攻击、评估、检测、日志与复现**串起来的工具链。HarmBench、JailbreakBench、StrongREJECT、WildGuard、garak、PyRIT、promptfoo 之所以重要，就在于它们帮助研究者把“看起来能越狱”变成“可度量、可比较、可回归测试”的工程对象。citeturn35view2turn36search0turn16search0turn21search0turn29view2turn29view3turn29view4

就开放权重模型而言，Hugging Face 已经形成了一个很清晰的研究样板层：一边是**底模与多模态模型**，如 Meta Llama、Qwen、DeepSeek、Mistral、LLaVA、OpenAssistant；另一边是**安全组件与评估资产**，如 Llama Guard、Prompt Guard、ShieldGemma、WildGuard、JailbreakBench 数据集与 HarmBench 分类器。对于中文研究者，这实际上意味着：**Hugging Face 不只是模型仓库，更是安全研究的标准件市场**。citeturn33search2turn15search3turn15search5turn34search1turn9search15turn10search4turn8search5turn8search13turn8search2turn21search1turn36search2turn35view1

### 代表性框架与项目

| 项目/仓库 | 类别 | 开源状态 | 关键实现点 | 门槛与依赖 | 适合用途 |
|---|---|---|---|---|---|
| `llm-attacks/llm-attacks` | 攻击研究 | 开源 MIT | GCG 官方实现；要求 `fschat==0.2.23`；默认面向本地 HF 权重；官方仓库后续补充了 `nanogcg`。citeturn6search0turn30view3 | 需要本地模型与GPU，白盒/半白盒条件更友好 | 研究对抗后缀与开放权重模型脆弱性 |
| `EasyJailbreak/EasyJailbreak` | 攻击框架 | 开源 GPL-3.0 | 把过程拆成 seed、mutate、attack、evaluate；`python>=3.9`，可 `pip install easyjailbreak`。citeturn6search2turn29view1turn30view2 | Python 环境即可起步，但要做大规模实验仍需模型/评审器资源 | 对比不同越狱配方、做研究原型 |
| `tml-epfl/llm-adaptive-attacks` | 攻击研究 | 开源 | 强调模型特定的自适应模板与接口漏洞，如 logprob、prefill、transfer。citeturn37search0turn37search1 | 更适合理解“为什么静态防御不够” | 评估防御的适应性鲁棒性 |
| `HarmBench` | 基准/评估 | 开源 | 比较 18 种红队方法与 33 个模型/防御；支持 transformers、多个闭源API与多模态；提供 HarmBench 分类器。citeturn35view1turn35view2 | `pip install -r requirements.txt` + spaCy；单机/Slurm 都可。citeturn35view0turn35view1 | 标准化评估、对抗训练、回归测试 |
| `JailbreakBench` | 基准/评估 | 开源 | 提供行为集、统一 threat model、system prompt/chat template、评分函数与 leaderboard。citeturn36search0turn36search1 | 适合做可复现实验与 artifact 管理 | 管理越狱样本库、比较攻击/防御 |
| `StrongREJECT` | 自动评审/基准 | 开源 | 强调“输出是否真正提供了有用违规信息”；autograder 打分 0–1。citeturn16search0turn16search2turn16search5 | 常需外部 judge 模型或兼容实现 | 防止 ASR 被高估，改进评估质量 |
| `MMJ-Bench` | 多模态基准 | 开源 | 统一比较多模态越狱攻击与防御，评估 defense-utility tradeoff。citeturn12search0turn12search1turn12search3 | 需图像/多模态模型与较重算力 | 多模态安全评估首选基线 |
| `garak` | 检测/红队 | 开源 | LLM vulnerability scanner；支持 HF、OpenAI、Bedrock、REST 等；CLI 安装简单。citeturn29view2turn30view0 | Python 3.10–3.12；命令行工作流清晰。citeturn30view0 | 持续扫描 prompt injection、数据泄漏、毒性、jailbreak |
| `PyRIT` | 检测/红队 | 开源 MIT | Microsoft 的生成式AI风险发现框架，面向安全团队与工程团队。citeturn29view3 | 文档/网站驱动，适合企业流程整合 | 结构化红队、审计记录与企业集成 |
| `promptfoo` | 评测/红队 | 开源 MIT | Node.js CLI；支持 eval、red teaming、CI/CD、报告视图。citeturn29view4turn30view4 | 需要 Node.js `^20.20.0` 或 `>=22.22.0`。citeturn30view4 | CI中做安全回归测试、模型对比 |
| `allenai/wildguard` | 守卫/分类器 | 开源 | 同时判别 prompt harmfulness、response harmfulness、response refusal；支持 HF/vLLM。citeturn21search0turn21search2turn29view5 | 7B 级别，部署成本中等 | 做输入/输出双向安全网关 |
| `meta-llama/Llama-Guard-*` 与 `Prompt-Guard-*` | 守卫/分类器 | 开放权重 | Llama Guard 做输入/输出安全分类；Prompt Guard 检测恶意与注入式提示。citeturn8search5turn8search9turn8search1turn8search13 | 轻量到中等，适合网关式部署 | 输入过滤、输出过滤、prompt injection 检测 |
| `google/ShieldGemma*` | 守卫/分类器 | 开放权重 | ShieldGemma 覆盖文本安全；ShieldGemma 2 面向图像安全分类。citeturn8search2turn8search6 | 适合多模态输入前置过滤 | 文本+图像安全审核 |
| `andyrdt/refusal_direction` | 白盒安全机理研究 | 开源 Apache-2.0 | 复现“拒答方向”研究，帮助理解与分析表征级护栏。citeturn19search1 | 需要白盒激活访问与模型内部知识 | 机理研究、白盒审计 |

### 主要发布者与目标模型

| 发布者 | 代表模型 | 开放状态 | 多模态 | 在去护栏研究中的典型位置 | 可操作性备注 |
|---|---|---|---|---|---|
| OpenAI | GPT-3.5、GPT-4、GPT-4o | 闭源 API/产品 | GPT-4o 原生支持 text/audio/image/video 输入，生成 text/audio/image。citeturn27view3 | Many-shot、自适应攻击、音频安全迁移评估常用目标。citeturn26view2turn37search0 | 主要只能做黑盒/灰盒测试；难以做权重/微调型研究 |
| Anthropic | Claude 2–4 | 闭源 API/产品 | Claude 4 系统卡强调 reasoning、visual analysis、computer use、tool use。citeturn26view1 | Many-shot、assistant-prefill、代理/工具使用安全研究常见目标。citeturn28view2 | API 层攻击现实；权重修改不可行 |
| Google | Gemini 家族 | 闭源 API/产品 | Gemini 官方文档强调从底层构建的多模态，覆盖图像、视频、音频与PDF等场景。citeturn32search14turn32search5 | 多模态越狱、音频越狱、视觉攻击常见目标。citeturn12search2turn13search2 | 黑盒为主；适合做多模态鲁棒性评估 |
| Meta | Llama 2/3/3.1/4 | 开放权重但许可证受限 | Llama 4 是 Meta 首批原生多模态开放权重系列；HF 访问需接受许可与 AUP。citeturn33search0turn33search2turn33search7 | 几乎所有白盒方法、SFT/LoRA、RepE、后门、消融研究都大量使用 | 去护栏白盒研究的首选靶标之一 |
| Mistral AI | Mistral 7B、Large 2、Medium 3.5 | 多为开放权重/研究许可 | 新一代 Large 3 为多模态；Mistral 7B Instruct 官方明确称其“没有 moderation mechanisms”。citeturn34search5turn34search0 | 常用于开放权重对比与 safety-light 场景研究 | 适合本地实验，但“弱护栏”并不等于“已去护栏” |
| 阿里云 Qwen | Qwen、Qwen2.5、Qwen2.5-VL | 开放权重 | 官方组织同时发布 LLM 与 LMM；Qwen2.5-VL 强调图像理解增强。citeturn15search3turn15search6 | 中文与多语场景、安全评测、视觉越狱研究的重要对象 | 对中文研究者尤其实用 |
| DeepSeek | DeepSeek-V3、R1 | 开放权重/开放模型页 | 以文本/推理为主，开放 HF 模型页。citeturn15search1turn15search5 | 常被纳入推理能力与安全权衡研究 | 开放权重便于做本地白盒与再训练分析 |
| LLaVA | LLaVA、LLaVA-NeXT | 开源研究模型 | 支持多图像、视频与 3D interleave。citeturn9search7turn9search15 | HADES、MMJ-Bench 等多模态越狱论文的核心对象。citeturn12search2turn12search0 | 多模态白盒攻击/防御实验最方便 |
| OpenAssistant | OA 系列模型与数据集 | 开源 Apache-2.0 | 主要是文本 | 常用于开放对话对齐与安全数据实验 | 历史模型较旧，但数据与流程仍有研究价值。citeturn10search0turn10search4 |

一个重要结论是：**目标模型是否开放权重，几乎决定了你能研究的是“越狱”还是“去护栏”**。如果模型只开放 API，那么实务上更像是在研究**外部攻击面**；如果模型开放权重，则可以研究**内部安全机制是否可被结构性移除**。因此，任何“在 Llama 上成功”的结论，都不应未经区分地外推到 GPT/Claude/Gemini。这个边界在阅读论文时必须始终保留。citeturn33search7turn27view3turn26view1turn32search14

## 实证评估与能力保持

关于“处理后模型是否还能保持原能力”，目前最有价值的结论不是来自单篇攻击论文，而是来自**评测论文之间的交叉校正**。许多攻击论文报告非常高的 ASR，例如简单自适应攻击在若干模型上报告了极高成功率；但 StrongREJECT 明确指出，过于宽松的 judge 往往把“模型不拒绝了”误判为“模型给出了真正有用的违规答案”。此后 The Jailbreak Tax 更进一步，把问题转化为可安全评测的代理任务，发现**所有被测越狱**都伴随明显的效用损失。换言之，“去护栏”常常是把模型的拒答和一部分高质量任务能力一起打乱，而不是干净地只去掉护栏。citeturn37search0turn16search0turn18search0

Many-shot 的实证更能说明这一点。Anthropic 在原始论文中展示了：当上下文里加入足够多的有害示范时，Claude 2.0、GPT-3.5、GPT-4、Llama 2 70B 与 Mistral 7B 都会显著更易输出不该输出的内容；而图中的 in-context power law 也说明，随着示范数增加，模型的上下文学习能力本身就会系统性地“压过”安全对齐。到 Claude 4 的系统卡中，Anthropic 仍报告 Many-shot 下模型会在一个评测上约 **25%** 概率响应有害建议请求，说明即便新模型在整体安全上提升，长上下文诱导仍未被根本解决。citeturn26view2turn28view0turn27view0turn28view2

表征工程与拒答方向编辑在“能力保持”上更值得单独看待。Revisiting Jailbreaking through the Lens of Representation Engineering 明确提出，安全模式对生成文本的表面语义影响相对有限，但对是否拒答起关键作用；“Refusal in Language Models Is Mediated by a Single Direction”则把这种现象形式化为低维方向假设。这类工作之所以引发广泛关注，正是因为它们暗示：某些模型的拒答机制可能并不是深度分布式、不可分离的，而更像是可被线性近似操控的局部表征。与此同时，后续几何与推理相关研究也提醒，这种“简单操控”在更复杂的 reasoning setting 中未必同样稳健。citeturn19search6turn5search10turn19search4turn17search15

训练侧方法在“保持总体能力”的表象上有时比提示攻击更危险，因为它们更可能在**正常任务上看起来几乎不变**。例如 Qi 等指出，少量对抗样本就能显著削弱安全；BadEdit 和 MEGen 则分别报告在较小副作用下植入后门；Sleeper Agents 进一步说明，一旦危险行为被伪装成条件触发，SFT、RL 和对抗训练甚至可能帮助模型**更好地隐藏**危险行为。这类方法的实证含义是：如果只测 MMLU、GSM8K 或 HumanEval 的平均分，很可能看不到问题。citeturn5search1turn20search1turn20search13turn5search3

### 建议的评估指标与测试集

对“去护栏后能力保持/退化”的评估，不宜只看一组安全提示。更合理的是把指标拆成四层：

| 评估层 | 关注问题 | 推荐工具/基准 |
|---|---|---|
| 安全绕过层 | 是否绕过拒答；是否真的给出有用违规信息 | HarmBench、JailbreakBench、StrongREJECT、ALERT。citeturn35view2turn36search0turn16search0turn16search1 |
| 通用能力层 | 常识、知识、推理、代码是否保持 | MMLU/MMLU-Pro、GSM8K、HumanEval、MT-Bench。citeturn22search20turn22search12turn22search1turn22search6turn23search0 |
| 多模态能力层 | 图像、图表、表格、视频理解是否退化 | MMMU、MMBench、MMBench-Video、Video-MMMU。citeturn22search3turn23search1turn23search7turn22search11 |
| 安全副作用层 | 是否过拒答、误杀正常请求、输出稳定性如何 | WildGuard refusal/harmfulness、GPT/Claude/Llama Guard 类守卫模型，外加人工抽样复核。citeturn21search0turn21search2turn8search5turn8search1 |

如果研究重点是“**安全机制被削弱后，基础能力是否还在**”，那么 The Jailbreak Tax 提供了一个很好的方法论：使用**可安全评价、带标准答案的代理任务**来代替真实有害任务，从而把“越狱成功”与“答案是否真有用”分开测。对于学术和企业内部研究，这比直接围绕危险内容做实验更可持续，也更容易过审计。citeturn18search0

## 多模态支持与差异

多模态并不是“在文本越狱前面再加一张图”那么简单。文献显示，多模态脆弱性至少有三种来源：**模态对齐不一致**、**输入前处理链条更长**、以及**安全过滤常常偏向文本**。HADES 的工作把视觉输入描述为多模态对齐的“阿喀琉斯之踵”，对 LLaVA-1.5 报告了平均 **90.26%** ASR、对 Gemini Pro Vision 报告了 **71.60%** ASR；Universal Adversarial Attack on Aligned Multimodal LLMs 则进一步展示了“单张优化图像跨提示、跨模型泛化”的可能，在 SafeBench 上对某些模型可达到很高成功率。citeturn12search2turn11search2

图像侧攻击大致可以再分成三类。第一类是**显式图文协同**，把有害意图隐藏、分拆或重编码到图文联合输入里；第二类是**像素级/通用对抗图像**，直接沿视觉编码器与语言头优化；第三类是**场景与上下文注入**，例如利用图片中的 OCR 文本、布局、手写体、图表标注或多图像互补语境来逃逸文本过滤。MMJ-Bench 的意义就在于它把这些传统上很分散的多模态攻击/防御放到统一管线里评测，并显式考察 defense-utility tradeoff。citeturn12search0turn12search1turn11search6turn38search12

音频侧的研究结论也很明确：把模型从纯文本扩展到语音/听觉，不等于自动继承文本管线上的安全性。Audio Is the Achilles' Heel 发现，五个先进音频大模型在“有害音频问题”设置下平均 ASR 为 **69.14%**，而对 Gemini-1.5-Pro 的 speech-specific jailbreak 也有较高成功率；AudioJailbreak 与 AJailBench 则把这一方向系统化为“端到端大音频语言模型”的单独基准与方法学。与此同时，OpenAI 的 GPT-4o 系统卡表明它已经做了较强的**文本到音频拒答迁移**：在文本与音频的安全评测上，“Not Unsafe”从 0.95 到 0.93，“Not Over-refuse”从 0.81 到 0.82，但官方也承认低质量输入、背景噪声、回声和输出过程中的打断会削弱安全鲁棒性。citeturn13search2turn11search1turn11search9turn27view2turn28view1

视频与复杂文档输入是接下来最容易被低估的部分。因为视频问题不仅涉及单帧视觉理解，还涉及**时间序列、跨帧指代、字幕/语音/OCR混合信号**；而文档/PDF/表格问题则混合了版面解析、OCR、图表语义和长上下文。Gemini 官方文档已经把 PDF 与多模态 embedding 视为标准能力；MMMU 也把图表、地图、表格、乐谱、化学结构等都纳入评价对象；因此，在企业内部实际应用里，所谓“多模态越狱”的现实载体很可能不是攻击图片，而是**RAG文档、扫描件、财务报表、表格截图与多媒体工单**。citeturn32search0turn22search3

综合来看，**文本方法向多模态的迁移并不总是成立**。音频研究已明确指出，先进文本越狱方法无法简单通过 TTS 直接迁移到端到端音频语言模型；反过来，多模态模型又会产生文本模型没有的新弱点。因此，“是否支持多模态”不能只看模型能不能收图收音，而要看**每种模态上的安全评估是否独立做过**。citeturn11search1turn12search0turn27view2

## 检测、防御与法律伦理

当前防御方法可以粗分为四层：**输入检测**、**输出检测**、**模型内生防护**、**系统级最小权限与审计**。Meta 的 Prompt Guard 面向恶意提示与注入输入分类；Llama Guard 面向输入/输出安全分类；Google 的 ShieldGemma 把文本与图像审核模型开放出来；AllenAI 的 WildGuard 则把 prompt harmfulness、response harmfulness 与 refusal 统一到一个模型中。WildGuard 论文报告，在一个界面集成场景中，它能把 jailbreak success rate 从 **79.8%** 降到 **2.4%**。这些模型并不意味着“越狱问题已解决”，但它们构成了实践上最可落地的一道前置/后置过滤网。citeturn8search1turn8search5turn8search2turn8search6turn21search0turn21search1

另一类防御来自**鲁棒推理和输入扰动**。SmoothLLM 通过对输入做多次随机微扰并聚合输出来识别对抗式越狱，论文称它对 GCG、PAIR、RandomSearch 与 AmpleGCG 都具备较好的鲁棒性，同时带来“小但不可忽略”的正常性能权衡。JailGuard 则利用“攻击样本对小扰动更脆弱”这一思想，做通用检测；更新版本报告在图像与文本数据集上平均准确率分别约为 **81.68%** 与 **79.53%**。这类方法适合黑盒场景，但也正因为依赖攻击脆弱性，一旦攻击者做自适应优化，效果就可能下降。citeturn8search16turn8search7

真正更稳妥的结论反而来自系统安全机构。OWASP 在 2025 版风险中把 Prompt Injection 列为 LLM01，把 Training Data / Model Poisoning 列为核心风险之一；NCSC 则明确指出，把 prompt injection 当成“LLM 版 SQL 注入”是危险类比，因为当前LLM内部没有天然的数据/指令边界。它建议从**降低风险与影响**出发，而不是寄望于一个会“彻底消灭 prompt injection 的产品”。这与现代 agent 安全实践的主线是一致的：**即使模型被诱导，也不应让它拥有超额工具权限**。citeturn24search0turn24search2turn25search0

法律与伦理层面，研究者至少要同时满足三层要求。第一，**授权边界**：对商业模型和企业系统做红队测试，必须符合服务条款、内部授权与数据处理规则；对开放权重模型做本地实验，也要遵守许可证与 AUP。Meta 的 Llama 下载流程就明确要求先接受许可与可接受使用政策。第二，**研究目的与最小必要原则**：NIST 的 GenAI Profile 将危险内容、信息安全与价值链安全都纳入治理范畴，意味着“只要是安全研究就可以无边界操作”的想法并不成立。第三，**司法辖区**：美国 DOJ 2022 年更新政策，表示善意安全研究不应被按 CFAA 起诉，但这不等于全球通用豁免，更不覆盖合同、隐私、出口管制或平台规则。citeturn33search2turn33search7turn14search0turn14search7turn14search3

从工程实现看，最有效的防线通常不是“再训一个更安全的大模型”，而是**把模型放进一个更安全的系统**：输入端用 Prompt Guard/WildGuard/Llama Guard/ShieldGemma 做分类与规范化，运行时用 promptfoo、garak、PyRIT 做持续红队和回归测试，工具调用层做 allowlist、沙箱化与权限收缩，日志层保留足够的上下文用于追溯。HarmBench 和 JailbreakBench 则适合在模型升级、模板调整或 guard 更新后做标准化对照。citeturn29view2turn29view3turn29view4turn21search2turn8search1turn8search5turn8search2turn35view2turn36search0

## 上手路径与资源清单

对希望系统进入该领域的研究者，最好的起点不是去搜集“最新越狱词典”，而是先建立一个**安全、可审计、可复现**的研究路径。建议按“评估先于攻击、开放权重先于闭源、代理任务先于真实危险任务”的顺序推进。经验上，HarmBench、JailbreakBench、StrongREJECT、WildGuard 与守卫模型（Llama Guard / Prompt Guard / ShieldGemma）足以搭起一套合规的入门研究工作台。citeturn35view2turn36search0turn16search0turn21search0turn8search1turn8search5turn8search2

### 建议的安全上手路径

第一阶段应先读**原始论文与系统卡**，而不是社区二手总结。优先顺序可以是：Many-shot Jailbreaking、Improved Few-Shot Jailbreaking、Universal and Transferable Adversarial Attacks on Aligned Language Models、Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks、Refusal in Language Models Is Mediated by a Single Direction、Fine-tuning Aligned Language Models Compromises Safety、Universal Jailbreak Backdoors from Poisoned Human Feedback、Sleeper Agents、HarmBench、JailbreakBench、StrongREJECT、MMJ-Bench，以及 GPT-4o 与 Claude 4 系统卡。这样读下来，能够把**输入级、模型级、系统级、多模态级**方法一次性建立起来。citeturn4search1turn4search2turn6search0turn37search0turn5search10turn5search1turn20search0turn5search3turn35view2turn36search0turn16search0turn12search0turn9search0turn9search1

第二阶段应建立**本地开放权重测试环境**，而不是一开始就冲商业API。Llama、Qwen、Mistral、DeepSeek、LLaVA 都适合作为研究对象；如果目标是中文与多模态，Qwen2.5-VL 和 LLaVA-NeXT 的优先级很高；如果目标是安全网关与检测，WildGuard、Llama Guard、Prompt Guard 和 ShieldGemma 更值得先部署。相较之下，OpenAssistant 更适合拿来理解数据流程与对话式对齐，而不一定适合作为当代强基线。citeturn33search2turn15search3turn34search1turn15search5turn9search15turn15search6turn21search1turn8search5turn8search1turn8search2turn10search0

第三阶段才是**实验设计**。更安全也更有研究价值的做法，是先用 The Jailbreak Tax 的思想构造**可安全评测的代理拒答任务**：例如让模型在人为定义的“应拒答”知识域上拒答，再测 MMLU、GSM8K、HumanEval、MT-Bench 与 MMMU/MMBench 是否退化；与此同时，使用 HarmBench/JailbreakBench/StrongREJECT 去测“绕过拒答”是否真的产生有用信息，而不是只看“不拒绝了”。这一阶段不需要追求公开“最强攻击”，重点是建立**before/after 的可比评估矩阵**。citeturn18search0turn22search20turn22search1turn22search6turn23search0turn22search3turn23search1turn35view2turn36search0turn16search0

第四阶段是**把检测和防御串起来**。在实践中，建议至少同时跑两类工具：一类是基准/红队管线，如 HarmBench、promptfoo、garak、PyRIT；另一类是守卫模型，如 WildGuard、Prompt Guard、Llama Guard、ShieldGemma。对企业内部项目，最好把 promptfoo 或 garak 接入 CI/CD，在模型、system prompt、工具 schema、RAG 检索模板变更后自动回归。citeturn29view2turn29view3turn29view4turn21search2turn8search1turn8search5turn8search2

### 必读与常备资源

下面这份清单按用途组织，适合长期保存：

| 用途 | 资源 |
|---|---|
| 方法入门 | Many-shot Jailbreaking、I-FSJ、GCG/llm-attacks、AutoDAN、PAIR、自适应攻击。citeturn4search1turn4search2turn6search0turn6search1turn36search10turn37search0 |
| 模型侧机制 | Fine-tuning Aligned Language Models Compromises Safety、Refusal Direction、RepE、Sleeper Agents、Universal Jailbreak Backdoors、BadEdit、MEGen。citeturn5search1turn5search10turn19search6turn5search3turn20search0turn20search1turn20search13 |
| 基准与评审 | HarmBench、JailbreakBench、StrongREJECT、ALERT、MMJ-Bench。citeturn35view2turn36search0turn16search0turn16search1turn12search0 |
| 防御与检测 | SmoothLLM、JailGuard、WildGuard、Prompt Guard、Llama Guard、ShieldGemma。citeturn8search16turn8search7turn21search0turn8search1turn8search5turn8search2turn8search6 |
| 工程工具 | garak、PyRIT、promptfoo。citeturn29view2turn29view3turn29view4 |
| 目标模型页 | Meta Llama、Qwen、DeepSeek、Mistral、LLaVA、OpenAssistant、GPT-4o System Card、Claude System Cards。citeturn33search2turn15search3turn15search5turn34search1turn9search15turn10search0turn9search0turn9search5 |
| 中文或中英双语整理 | `ant-research/awesome-mllm-guardrails`、`Awesome-Jailbreak-on-LLMs`、`Awesome-LLM-Safety-Papers`。citeturn21search11turn36search4turn21search5 |

最后，如果你的目标是**真正严谨地研究“去护栏是否会保留能力”**，那么最重要的不是“找到最新最强攻击”，而是同时做到三件事：一是把**绕过拒答**和**输出有用性**分开测；二是把**安全指标**和**通用能力指标**并行测；三是把**模型级变化**和**系统级变化**分开归因。现有文献已经很清楚地表明，很多看似“成功”的去护栏，本质上只是把模型推入了一个更混乱、更不稳定的状态，而不是获得一个“能力完全保留、只去掉护栏”的干净版本。对研究、工程和治理来说，这一点比任何单一 ASR 数字都更重要。citeturn16search0turn18search0turn25search0
