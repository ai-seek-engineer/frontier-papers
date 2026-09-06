# LLM 数值精度、低精度计算与量化：从位级表示到 Tensor Core、Kernel 与 cost/token


这份报告的核心观点可以先压缩成一句话：

> **“模型是 FP8 / INT4 / FP4”通常不是一个完整的工程描述。真正决定数值稳定性和性能的是：每个 tensor 如何存储、如何 scale，算子以什么 dtype 读入，乘法与累加用什么精度，中间 reduction 用什么精度，结果写回什么 dtype，以及这一整条链路是否有硬件原生指令与成熟 kernel。**

截至 **2026 年 9 月 6 日**，产业主线已经从“单一 dtype”明显转向 **mixed precision + fine-grained scaling + microscaling + operator-specific precision**。OCP MX 标准把“低比特元素 + block shared scale”明确标准化；Hopper/Ada 将 FP8 变成实用 Tensor Core 类型；Blackwell 将 NVFP4/MXFP4/MXFP8 推向原生硬件路径；CUDA/CUTLASS 2026 年的 Rubin 初始支持则继续强化 FP8/FP4 及 block-scaled mixed-precision MMA。citeturn2view0turn0search11turn15search25turn14search10

## 数值表示：范围、精度与误差的统一底座

### 从 IEEE 浮点数真正理解 precision

一个典型二进制浮点数由三部分构成：

```text
┌─────────┬───────────────────┬────────────────────────────┐
│ sign s  │ exponent E        │ fraction / mantissa bits   │
└─────────┴───────────────────┴────────────────────────────┘
     1           e bits                   m bits
```

对于 normal number：

\[
x=(-1)^s \times (1.f)_2 \times 2^{E-\mathrm{bias}}
\]

其中：

- `sign` 决定正负；
- exponent 决定**数量级、dynamic range**；
- fraction/mantissa 决定同一数量级内能切得多细，即**precision**。

对于 IEEE 风格 subnormal：

\[
x=(-1)^s\times(0.f)_2\times2^{1-\mathrm{bias}}
\]

也就是说，subnormal 放弃隐含的 leading `1`，用更差的相对精度换取从最小 normal 到 0 的渐进过渡，从而避免一次性“掉到 0”。CUDA 对 IEEE 754 浮点语义、舍入模式和特殊值有明确说明；NVIDIA 与 Intel 的格式文档也分别给出了 FP16/BF16 的 exponent/fraction 配置。citeturn4search4turn4search0turn4search18turn4search2

**一个具体 FP32 编码例子：13.25。**

十进制：

\[
13.25=8+4+1+0.25
\]

二进制：

\[
13.25=(1101.01)_2
\]

规格化：

\[
1101.01_2=1.10101_2\times2^3
\]

因此：

```text
sign      = 0

real exp  = 3
bias      = 127
encoded E = 3 + 127 = 130 = 10000010₂

fraction  = 10101000000000000000000

FP32:
0 | 10000010 | 10101000000000000000000
```

即：

```text
01000001010101000000000000000000
```

这说明了一个非常重要的事实：**浮点数不是固定间隔网格。** 当 exponent 增大时，相邻 representable number 的绝对间隔也随之增加。

对于具有 \(m\) 个显式 fraction bits 的标准二进制 normal floating point，1 附近的 machine epsilon 通常定义为：

\[
\epsilon_{\mathrm{mach}}=2^{-m}
\]

例如：

\[
\epsilon_{FP32}=2^{-23}\approx1.192\times10^{-7}
\]

\[
\epsilon_{FP16}=2^{-10}=9.765625\times10^{-4}
\]

\[
\epsilon_{BF16}=2^{-7}=0.0078125
\]

注意数值分析中有时还定义 **unit roundoff**

\[
u=\frac{\epsilon_{\mathrm{mach}}}{2}
\]

用于 round-to-nearest 的误差界，所以阅读论文时不要把 `epsilon` 与 `u` 混用。

**ULP（Unit in the Last Place）**则与当前 exponent 有关。粗略地，对正常范围内的 \(x\)：

\[
ULP(x)\approx
2^{\lfloor\log_2|x|\rfloor-m}
\]

所以 FP16 在 \(x\approx1\) 时分辨率约 \(10^{-3}\)，但到 \(x\approx1024\) 时，绝对 spacing 已经接近 1。相对精度大致固定，绝对精度不是固定的。

CUDA 默认 IEEE 风格舍入是 **round-to-nearest, ties-to-even**：取最近的 representable value；刚好位于两个值中间时，使最后一位为偶数。NVIDIA 给出的典型整数舍入例子包括 `0.5 → 0`、`1.5 → 2`。这种规则相对于永远“0.5 向上”能减小长时间累计的统计偏差。citeturn4search0turn4search20

### overflow、underflow、subnormal、saturation 是不同问题

**Overflow** 是数值绝对值超过格式最大有限值。IEEE 浮点算术在适用情况下可能产生 `±Inf`；量化格式或转换操作则经常采用 **saturation/clamping**：

\[
x_q=\mathrm{clip}(x,q_{\min},q_{\max})
\]

例如 OCP FP4 E2M1 没有 Inf/NaN 的普通编码空间，超范围转换要求饱和到最大有限值；TensorRT 的量化描述同样把 rounding 与 clamping 视为核心量化误差来源。citeturn2view0turn12search15

**Underflow** 是绝对值太小，低于 normal range。如果格式具有 subnormal，则会逐渐降低有效 precision；更小则最终变成 0。citeturn4search18

这正是 **FP16 与 BF16 最大差异之一**。FP16 只有 5 个 exponent bits，最大有限值：

\[
65504
\]

BF16 有与 FP32 一样的 8-bit exponent，动态范围基本与 FP32 相同，因此训练中的 activation/gradient 更不容易因为 exponent 不够而 overflow/underflow。NVIDIA TensorRT 文档也明确指出 FP16 的 5-bit exponent 使它比 BF16/TF32/FP32 更容易 overflow。citeturn4search8turn4search6

### datatype 全景对比表

下表中的 FP8 数字针对常见 E4M3FN/E5M2 定义；**E4M3FN、E4M3FNUZ 等并不是完全相同的 bit interpretation**。ONNX 明确指出 FNUZ 变体只有一个 zero、一个 NaN，而且 exponent bias 不同，因此不能仅把它们视为相同字节的别名。citeturn0search11turn21search5turn4search31

| 格式 | 总 bits | Exponent | Fraction | 最大有限值 | 最小 normal | ε near 1 | 大致有效十进制精度 | 核心特征 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 | 32 | 8 | 23 | \(3.40\times10^{38}\) | \(1.175\times10^{-38}\) | \(1.19\times10^{-7}\) | ~7.2 digits | 高 range + 高 precision |
| TF32 | ~19 个计算有效 bits | 8 | 10 | ≈FP32 | ≈FP32 | \(9.77\times10^{-4}\) | ~3.3 | **计算模式，不是普通 storage dtype** |
| FP16 | 16 | 5 | 10 | 65504 | \(6.10\times10^{-5}\) | \(9.77\times10^{-4}\) | ~3.3 | precision 好于 BF16、range 小 |
| BF16 | 16 | 8 | 7 | \(\approx3.39\times10^{38}\) | \(1.175\times10^{-38}\) | 0.0078125 | ~2.4 | FP32-like range |
| FP8 E4M3FN | 8 | 4 | 3 | 448 | \(2^{-6}=0.015625\) | 0.125 | ~1.2 | 更多 mantissa，较窄 range |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | \(2^{-14}\) | 0.25 | ~0.9 | 更多 range，较少 precision |
| INT8 | 8 | — | — | 127/255 等 | scale 决定 | scale 决定 | 均匀量化 | 依赖 scale/zero-point |
| INT4 | 4 | — | — | 7/15 等 | scale 决定 | scale 决定 | 极粗均匀量化 | 必须细粒度 scaling |
| FP4 E2M1 | 4 | 2 | 1 | 6 | 1 | 约 0.5 | 极低 | 值集合极少，必须 block scale |
| MXFP8 | 8+scale | element FP8 | — | block scale 决定 | block scale 决定 | block-dependent | — | 32 values / E8M0 scale |
| MXFP6 | 6+scale | E3M2/E2M3 | — | block-dependent | block-dependent | — | — | 32 values / E8M0 scale |
| MXFP4 | 4+scale | E2M1 | — | block-dependent | block-dependent | — | — | 32 values / E8M0 scale |
| NVFP4 | 4+scale | E2M1 payload | — | 2-level scale | 2-level scale | — | — | 16-value E4M3 micro-scale + tensor scale |

FP32、FP16、BF16、TF32 的组成由 NVIDIA/Intel 官方文档确认；OCP MX v1.0 标准规定 MXFP8/MXFP6/MXFP4 采用 32-element block 与 E8M0 shared scale，FP4 E2M1 最大有限值为 ±6。citeturn14search0turn4search2turn3view0turn3view1

这直接回答三个常见问题。

**为什么 BF16 mantissa 很少，却非常适合深度学习？**

因为深度学习尤其是训练中，很多问题首先是“数值能不能表示”，其次才是“最后几位准不准”。BF16 保留 FP32 的 8-bit exponent，因此 activation、gradient、loss 等跨度巨大的 tensor 不容易溢出；神经网络本身又对大量局部舍入噪声有一定容忍性，而 GEMM accumulation 通常会升到 FP32。BF16 因而牺牲局部 mantissa precision，换来了训练中更重要的 range 与工程稳定性，并显著减轻 FP16 对 loss scaling 的依赖。citeturn4search6turn20academia19

**为什么 FP16 更容易 overflow？**

不是因为“16 bit 太少”，而是因为它把 10 bits 给了 fraction，只剩 5 exponent bits；BF16 则把 8 bits 给 exponent，只留下 7 fraction bits。前者 precision 更好，后者 range 更好。citeturn4search8turn4search2

因此：

> **bit width 只告诉你 storage cost；exponent/fraction 分配才告诉你 numerical behavior。**

这也是为什么“FP8”“FP4”本身仍不是足够精确的规格名称。

### LLM Precision 全景知识图

下面是一张后文所有内容的统一索引；其核心层次对应 IEEE/FP8、OCP MX、Transformer Engine 以及 CUDA low-precision execution model。citeturn4search4turn0search2turn2view0turn14search12

```text
                         LLM Precision
                              │
       ┌──────────────────────┼───────────────────────┐
       │                      │                       │
   表示 Representation      Scaling                Execution
       │                      │                       │
 FP32/TF32/FP16          tensor/channel         storage dtype
 BF16/FP8/INT            token/group/block           ↓
 FP4/MX/NVFP4            dynamic/static         quantize/cast
       │                 current/delayed              ↓
       │                      │                  GEMM input
       └──────────┬───────────┘                       ↓
                  │                              Tensor Core
                  │                         multiply / MMA path
                  │                                   ↓
             Quantization                        accumulator
                  │                                   ↓
       PTQ / QAT / weight-only                  reduction
       W8A8 / W4A16 / W4A8                           ↓
       GPTQ / AWQ / SmoothQuant                  epilogue
       rotation / outlier handling                    ↓
                  │                              output dtype
                  │
          Transformer Operators
                  │
  ┌───────────────┼──────────────────────────────────────────┐
  │               │              │            │              │
 GEMM          Attention       Norm         Router        KV Cache
 low-bit       mixed          higher        higher        independently
 weights       precision      reduction     precision     quantizable
  │
  └───────────────────────── Hardware ───────────────────────┐
                 Tensor Core / HBM / Cache / SMEM / RF
                                   │
                                   ↓
                               Kernel
                 pack → scale → MMA → accumulate → epilogue
                                   │
                                   ↓
                           Serving Engine
           vLLM / SGLang / TensorRT-LLM / llama.cpp / MLC
                                   │
                                   ↓
                  TTFT / TPOT / tokens/s / J/token / $/token
```

## Transformer 数值误差与 Mixed Precision：哪里能降，哪里不能乱降

低精度神经网络之所以成立，不是因为“误差不存在”，而是因为我们可以把误差放在**不敏感且计算量最大的地方**，同时把关键 reduction、normalization、probability、routing 等路径留在较高精度。

一个非常有用的数值模型是：

\[
\mathrm{fl}(x\;\mathrm{op}\;y)
=(x\;\mathrm{op}\;y)(1+\delta),
\qquad |\delta|\lesssim u
\]

但如果连续累加 \(n\) 项，经典误差界会出现：

\[
\gamma_n=\frac{nu}{1-nu}
\]

以及类似：

\[
|\mathrm{fl}(a^Tb)-a^Tb|
\lesssim
\gamma_n |a|^T|b|
\]

这正解释了为什么 **multiply 输入可以很低精度，而 accumulator 往往不能同样低**。

### GEMM 为什么天然适合低精度

Transformer 中的大头计算几乎都是：

\[
C=A B
\]

例如一个 hidden size 4096、GQA QKV 输出总宽度 6144 的 projection：

```text
Activation A: [M, 4096]
Weight     B: [4096, 6144]
Output     C: [M, 6144]
```

每个输出元素：

\[
C_{ij}=\sum_{k=1}^{4096}A_{ik}B_{kj}
\]

这里有两个完全不同的数值问题：

1. 每次 \(A_{ik}B_{kj}\) 的输入能否被低精度近似；
2. 4096 个乘积累加时是否会不断丢低位。

前者神经网络往往能够容忍，后者会随着 reduction 长度迅速恶化。因此从 Volta 开始，NVIDIA Tensor Core 的经典 mixed path 就是 FP16 operands 配 FP32 accumulation；V100 whitepaper 明确给出了 FP16 inputs + FP32 accumulation 的 Tensor Core GEMM。Ampere 又增加 BF16、TF32 等混合精度 MMA。citeturn15search0turn15search1turn14search0

可以把真实计算链统一写成：

\[
\boxed{
\text{storage}
\rightarrow
\text{input quant/cast}
\rightarrow
\text{multiply/MMA}
\rightarrow
\text{accumulator}
\rightarrow
\text{epilogue}
\rightarrow
\text{output}
}
\]

典型组合是：

| Input × Weight | 累加 | 输出 | 为什么 |
|---|---|---|---|
| FP16 × FP16 | FP32 | FP16/BF16/FP32 | 防止长 reduction 丢失太多有效位 |
| BF16 × BF16 | FP32 | BF16/FP32 | BF16 operand range 大，accumulation 仍需 precision |
| TF32 × TF32 | FP32 | FP32 | FP32 range + reduced operand mantissa |
| FP8 × FP8 | 通常 FP32，部分推理路径可更低 | BF16/FP16/FP8 | FP8 input error 已大，accum 不宜继续无控制放大 |
| INT8 × INT8 | INT32 | INT8/FP16/BF16 | 8-bit products 的长整数和需要更宽 accumulator |
| FP4/MXFP4 × FP4/MXFP4 | 较高精度 accumulator | BF16/FP16/低精度 | payload 极粗，依赖 scale 和 high-precision accumulation |

cuBLAS 明确把 **compute type、intermediate precision、output type** 分开，并甚至提供“禁止 reduced-precision reduction”的数学模式；CUTLASS 也已有 FP8 输入、blockwise dequantization、FP32 accumulation 的具体 GEMM 路径。citeturn14search12turn14search4

### Transformer 各算子的数值敏感性

| 算子 | 低精度友好度 | 关键原因 | 常见工程策略 |
|---|---|---|---|
| Linear / GEMM | 很高 | 大量独立 MAC；高算术密度 | operands FP8/INT8/INT4/FP4，accum 高精度 |
| QKV projection | 很高 | 本质是 GEMM | 与 Linear 相同 |
| \(QK^T\) | 中高 | GEMM，但结果进入指数函数 | Q/K 可低精度，score/reduction 更谨慎 |
| Softmax | 低 | `exp` 对误差敏感，概率归一化 | max/reduction/exp 常高精度 |
| Attention × V | 中高 | GEMM/reduction | V/KV 可量化，accum 较高 |
| RMSNorm | 中低 | square + reduction + reciprocal sqrt | reduction 常 FP32 |
| LayerNorm | 更低 | mean/variance、减法 cancellation | FP32 reduction 常见 |
| RoPE | 中 | 长 context 下 phase error 有累积意义 | trig/rotation 常 BF16/FP32 |
| SiLU/GELU | 中 | 非线性但无超长 reduction | BF16/FP16 常见 |
| MoE router | 低 | top-k 是离散决策边界 | logits/router 常留高精度 |
| Embedding | 高 | lookup，没有计算 reduction | storage 可压缩 |
| LM Head | 中高 | GEMM 巨大，但 logit margin 决定 token | GEMM 可低精度，logits 通常高精度输出 |
| KV Cache | 中 | 被长期重复消费 | 独立 FP8/INT8/FP4 策略 |
| Sum/Reduce | 低 | rounding 随长度累积 | FP32 或宽 accumulator |
| AllReduce | 中低 | reduction + 分布式顺序变化 | 通信可低精度，累加需谨慎 |

这些原则与实际 FP8 训练设计相符：例如 DeepSeek-V3 将大量线性层纳入 FP8，但 attention score、normalization/routing 等路径并不会机械地“全 FP8”；相关系统研究也明确指出这些 activation/reduction tensors 往往仍留在更高精度。citeturn18search0turn18search28

**Softmax 是最典型的反例。**

稳定 softmax 应计算：

\[
p_i=
\frac{\exp(z_i-m)}
{\sum_j\exp(z_j-m)},
\qquad
m=\max_j z_j
\]

减去最大值把最大 exponent 变成 \(e^0=1\)，避免直接：

\[
e^{1000}\rightarrow\infty
\]

但即便避免 overflow，低精度量化 score 仍可能改变相对 logit difference：

\[
(z_i-z_j)
\]

而概率比：

\[
\frac{p_i}{p_j}=e^{z_i-z_j}
\]

因此 score 的小误差会经过 exponential 变成乘性概率误差。

**Norm 的风险则是 reduction。**

RMSNorm：

\[
y_i=
\frac{x_i}
{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}
\gamma_i
\]

真正敏感的不是最后那次乘法，而是：

\[
\sum_j x_j^2
\]

如果这里直接用 FP8/FP4 accumulate，几千个 channel 的 rounding、overflow/underflow 会影响**整个向量的共同归一化因子**。

这说明了一个重要原则：

> **误差在一个 element 上是局部的；误差进入 reduction、normalization、routing 或 probability normalization 后，会变成整个 tensor 或控制流的全局误差。**

### 为什么权重、activation、KV、accumulator 应该分别选 dtype

四类 tensor 的统计性质与生命周期完全不同：

```text
Weights:
静态、可离线分析、分布稳定
→ 最容易 PTQ / groupwise quantize

Activations:
依赖 prompt/token/layer
→ dynamic range 变化更大，outlier 更麻烦

KV Cache:
持续增长、跨 decode step 重复读取
→ memory capacity + bandwidth 极重要

Accumulator:
短暂存在但承担长 reduction
→ 不占长期 HBM，却极其影响数值稳定
```

因此把 accumulator 从 8 bit 提到 32 bit，长期 memory penalty 通常很小；而把权重/KV 从 16 bit 降到 8/4 bit，却会显著降低 HBM footprint。这就是 mixed precision 的根本经济学。

### Training mixed precision 与 inference precision

经典 FP16 mixed-precision training 使用：

```text
FP32 master weights
      │
      ▼ cast
FP16 forward / backward
      │
      ▼
scaled gradients
      │
      ▼ unscale
FP32 optimizer update
```

Micikevicius 等人的原始 mixed-precision 方法明确提出 **FP32 master copy + loss scaling**，用来解决 FP16 weight update 精度和小梯度 underflow。citeturn20academia19

Loss scaling 本质是：

\[
L'=S L
\]

因此：

\[
\nabla L'=S\nabla L
\]

先把很小的 gradient 放大到 FP16 normal/subnormal 可表示区间，optimizer update 前再：

\[
g=\frac{g'}S
\]

BF16 的 exponent 与 FP32 相同，因此通常无需像 FP16 那样频繁担忧梯度因 range 不够而 underflow，实践上大幅简化 mixed-precision training；但这不意味着 optimizer state 或所有 reduction 都应该 BF16。Intel 与 DeepSpeed 的文档都体现了 BF16/FP16 与 FP32 optimizer/master-state 的区别。citeturn4search6turn20search1

**Training FP8 比 inference FP8 难得多。** 推理时 weight 是固定的，activation 只有 forward；训练还存在 gradient、backward activations、optimizer update，tensor range 每个 step 都变化，而且 scale 需要跨 distributed rank 协调。Transformer Engine 因此提供 delayed scaling、amax history、distributed amax reduction 等机制。citeturn0search2turn0search11

## 从 INT8 到 FP8、FP4 与 Microscaling：低精度为什么这样演进

低精度历史最好不要理解为“bit 一直砍半”，而应理解为每代技术在解决上一代留下的瓶颈。

```text
FP32
  │  太贵：compute / memory
  ▼
FP16 mixed precision
  │  range 太小，loss scaling 麻烦
  ├──────────────► BF16
  │                 FP32-like range
  │
  └──► TF32
        FP32 storage，Tensor Core 加速
              │
              ▼
       INT8 inference
       memory + integer TOPS
              │
       activation outlier
              ▼
  SmoothQuant / mixed outlier handling
              │
              ▼
            FP8
   floating dynamic range + Tensor Core
              │
       scale 太粗仍受 outlier
              ▼
     block FP8 / MXFP8
              │
              ▼
     INT4 weight-only / W4Ax
              │
       4 bit accuracy困难
              ▼
       FP4 / NVFP4 / MXFP4
              │
     very fine block scaling
              ▼
     MXFP6 / mixed 4×6 等
```

Ampere 正式把 BF16、TF32、FP64 加入 Tensor Core；Hopper 引入 FP8 Transformer Engine；Blackwell 加入 FP4/FP6 和新的 low-precision scaling 路径；OCP MX 则标准化 MXFP8/MXFP6/MXFP4。citeturn15search1turn15search5turn15search24turn3view0

### FP8：不是“FP16 切一半”

FP8 的难题是：

\[
8 = 1 + e + m
\]

符号已经占 1 bit，只剩 7 bit 在 **range 和 precision** 之间分。

NVIDIA Transformer Engine 常用：

```text
E4M3:
S EEEE MMM
1 + 4 + 3
max finite ≈ 448

E5M2:
S EEEEE MM
1 + 5 + 2
max finite ≈ 57344
```

E4M3 多一个 mantissa bit，因此同一数量级的量化更细；E5M2 多一个 exponent bit，因此动态范围大得多。Transformer Engine 的传统 HYBRID recipe 通常利用这一差异，让 forward 类 tensor 偏向 E4M3，而动态范围更大的 backward gradients 可使用 E5M2。citeturn0search11turn0search2

这就是：

> **训练与推理可以使用不同 FP8 format，甚至同一次训练的 forward 与 backward 也可以不同。**

而 blockwise scaling 进一步改变了这个判断：如果每个小 block 都拥有自己的 scale，单个 FP8 元素不再需要自己承担那么大的 global range，Transformer Engine 文档因此指出 block-scaled FP8 可以在更多路径使用 E4M3，而不必完全依赖 E5M2 的 wider exponent range。citeturn0search5

### FP8 为什么几乎离不开 scaling

假定采用一种 convention：

\[
q=\mathrm{FP8}(x\cdot s)
\]

\[
\hat{x}=\frac{q}{s}
\]

为了让 tensor 最大值充分利用 FP8 representable range：

\[
s\approx
\frac{x_{\max}^{FP8}}{\operatorname{amax}(|x|)}
\]

另一套软件 API 可能存 inverse scale：

\[
q=\mathrm{FP8}(x/s_x),
\qquad
\hat{x}=q s_x
\]

所以看到 `scale`、`scale_inv` 时，首先要确认 convention，不要仅凭名字判断。

典型 scaled FP8 GEMM 可以抽象为：

\[
A_q=Q_{FP8}(A/s_A)
\]

\[
B_q=Q_{FP8}(B/s_B)
\]

MMA 计算：

\[
C_{\mathrm{acc}}
\approx
\sum_k A_{q,ik}B_{q,kj}
\]

然后等效恢复：

\[
C
\approx
s_A s_B C_{\mathrm{acc}}
\]

并在 epilogue 写成 BF16/FP16/FP8。

Transformer Engine 的 current scaling 实际使用当前 tensor 的 `amax` 生成 FP32 scale；delayed scaling 则维护 amax history，默认 recipe 可使用长达 1024 的历史窗口，并通过 margin 控制保守程度。citeturn0search11turn0search2

不同 scale granularity 可以看成一个 accuracy/metadata/compute 三角：

| Scaling | scale 数量 | 优点 | 缺点 | 常见适用 |
|---|---:|---|---|---|
| per-tensor | 1/tensor | 极便宜 | 一个 outlier 污染整个 tensor | FP8 baseline |
| per-channel | 1/channel | weight 很有效 | metadata/indexing 增多 | weight quant |
| per-token | 1/token/vector | activation 自适应强 | runtime scale reduction | dynamic activation |
| per-group | 1/group | INT4 常见平衡点 | 需要 group metadata | W4A16 |
| per-block | 1/small block | 最能控制 outlier | scale overhead 最大 | FP8/FP4/MX |
| static | 预先固定 | runtime 最便宜 | distribution shift 风险 | 可校准推理 |
| dynamic/current | 当前输入计算 | 鲁棒 | 每次 amax/reduction | activation |
| delayed | 历史 amax | 可 pipeline、减少同步 | scale 滞后 | FP8 training |

Google TPU7x 的 2026 年官方优化文档也把 per-tensor 作为 FP8 baseline，把 dynamic scaling 作为质量优先的默认选择，并指出 static scaling 可以减少 runtime scale calculation，但需要稳定的 calibrated range。citeturn17search3

### Transformer Engine 的真正思想

Transformer Engine 并不是“把 tensor `.to(float8)`”。

它实际管理的是一个状态机：

```text
high precision tensor
        │
        ▼
     compute amax
        │
        ├── current scaling ─────────┐
        │                           │
        └── amax history ─► delayed │
                                    ▼
                               choose scale
                                    │
                                    ▼
                             quantize to FP8
                                    │
                                    ▼
                                FP8 GEMM
                                    │
                             high precision acc
                                    │
                                    ▼
                              BF16/FP16 output
```

对于 MXFP8，Transformer Engine 采用 **32 consecutive values / E8M0 scale**；对于 NVFP4，则采用 **16-value E4M3 first-level scale + global FP32 second-level scale**。NVFP4 的训练 recipe 还包含 stochastic rounding、Hadamard transform 等手段来控制低比特梯度和 outlier。citeturn0search20turn0search2

FP8 理论 storage 从 BF16 的 16 bits/value 降到约 8 bits/value，因此 weight/eligible activation traffic 约减半；vLLM 文档报告 W8A8 FP8 在支持硬件上可带来约 2× model-memory reduction 和最高约 1.6× throughput 改善，但这是具体 serving stack 的实测/文档结果，而不是“所有模型必然 2×”。citeturn20search7

### INT quantization 的统一公式

整数均匀量化：

\[
x_q=
\mathrm{clip}
\left(
\operatorname{round}\left(\frac{x}{s}\right)+z,
q_{\min},
q_{\max}
\right)
\]

反量化：

\[
\hat{x}
=s(x_q-z)
\]

其中：

- \(s\)：scale；
- \(z\)：zero point；
- clipping：牺牲极端值换取主体 resolution；
- granularity：决定一组多少值共用 \(s,z\)。

对 asymmetric quantization，可近似取：

\[
s=
\frac{x_{\max}-x_{\min}}
{q_{\max}-q_{\min}}
\]

\[
z=
\operatorname{round}
\left(
q_{\min}-\frac{x_{\min}}s
\right)
\]

对 symmetric quantization：

\[
s=
\frac{\max |x|}
{q_{\max}}
,\qquad z=0
\]

TensorRT 的 explicit quantization 文档采用这一类 round/clamp/scale 语义，并区分 INT、FP8、FP4 与 block quantization。citeturn12search7turn12search11

**Symmetric** 的优势是 kernel 简单，因为：

\[
(x_q-z)
\]

退化为 \(x_q\)；**asymmetric** 能更充分利用偏置分布的码点，但需要 zero-point correction。

### W8A8、W4A16 到 W4A8KV4 到底是什么意思

这类命名本质上是 precision map 的压缩写法：

```text
W8A8
weights 8 bit
activations 8 bit

W8A16
weights 8 bit
activations 16 bit

W4A16
weights 4 bit
activations BF16/FP16

W4A8
weights 4 bit
activations 8 bit

W4A4
both 4 bit

W4A8KV4
weights 4
activations 8
KV cache 4
```

但仍然不充分。一个真正完整的规格至少应写成类似：

```text
Weight:
INT4 symmetric
group_size=128
per-group scale BF16

Activation:
FP8 E4M3
per-token dynamic scale

Accumulator:
FP32

KV:
FP8 E4M3
per-token-head scale

Output:
BF16
```

这才接近 kernel contract。

### Transformer activation outlier 为什么是 INT8/INT4 的大敌

假设一个 activation vector：

\[
x=[0.1, -0.2,0.4,\ldots,0.3,50]
\]

如果整个 tensor 用 INT8 symmetric scale：

\[
s=\frac{50}{127}\approx0.394
\]

那么所有：

\[
|x|<0.197
\]

附近的值都很容易量化到 0，而主体 activation 的 resolution 被那个 `50` 决定。

LLM.int8() 的原始工作观察到大模型中会出现少数 magnitude 很大的 “outlier feature dimensions”，因此把绝大多数矩阵乘留在 INT8，而 outlier dimensions 分解到 16-bit 路径。citeturn6search0

SmoothQuant 则提出更漂亮的等价变换。对于：

\[
Y=XW
\]

插入 channel-wise diagonal scale \(S\)：

\[
Y=
(XS^{-1})(SW)
\]

数学结果不变，但 activation 的 difficult-to-quantize magnitude 可以“迁移”到相对容易量化的 weight 上，从而让 W8A8 更稳定。该工作是 training-free PTQ，并报告了硬件上的实际 INT8 speedup，而非仅压缩 checkpoint。citeturn6search3

### FP4、NVFP4 与 MXFP4

OCP FP4 E2M1 可表示的非负有限 magnitude 极少，大体可以理解为：

```text
0
0.5
1
1.5
2
3
4
6
```

再加符号。OCP 标准规定 E2M1 最大 normal 为 6、最小 normal 为 1，并存在 0.5 这一 subnormal magnitude。citeturn3view1turn2view0

如果你直接对整个巨大 tensor 用这种格式：

\[
x\rightarrow FP4(x)
\]

几乎一定太粗。

真正可用的表示是：

\[
\boxed{x_i\approx S_b\,p_i}
\]

其中：

- \(p_i\)：4-bit FP4 payload；
- \(S_b\)：该小 block 的 shared scale。

因此真实数值不是“4-bit FP4”本身，而是：

> **FP4 payload + scale hierarchy**

#### MXFP4

OCP MX：

\[
v_i=X_bP_i
\]

每 **32 个元素**共享一个 **E8M0** scale。E8M0 本质上表示 power-of-two scale，因此 scale multiplication 很接近 exponent shift/add。citeturn2view0turn3view0

忽略 padding：

\[
\text{effective bits/value}
=
4+\frac{8}{32}
=4.25
\]

所以“MXFP4 是 4 bit”在 memory accounting 上也只是近似。

#### NVFP4

NVIDIA NVFP4 使用更小的 micro-block：

\[
16\text{ values}
\rightarrow
1\times E4M3\text{ local scale}
\]

并再配一个 tensor-level FP32 second-level scale。Transformer Engine 与 NVIDIA Blackwell 文档都描述了这种 two-level scaling。citeturn0search2turn15search25

仅算 first-level scale：

\[
4+\frac8{16}=4.5
\text{ bits/value}
\]

代价比 MXFP4 多一点 metadata，但 16-element block 对 outlier 的适应更强。

### Block Floating Point、Shared Exponent、Block Scaling 与 MX 不应混为一谈

传统 **Block Floating Point（BFP）**通常是：

```text
shared exponent E
+
private fixed-point mantissa_i
```

即 block 内：

\[
x_i\approx m_i 2^E
\]

而 MX 的更一般定义是：

\[
x_i\approx X_b P_i
\]

其中 \(P_i\) 自己还可以是 E4M3、E5M2、E2M1 等 floating element。

所以：

```text
BFP ⊂ shared-scale family

MX = standardized block-scaled element format
```

“block scaling”“shared exponent”“microscaling”高度相关，却不是严格同义词。OCP MX 标准显式区分 shared scale 与 private element data，并定义了 dot-product semantics。citeturn2view0turn1search2

**为什么 FP4 可能比 INT4 更适合部分神经网络分布？**

INT4 在 block 内是均匀 grid：

```text
... -3s -2s -s 0 s 2s 3s ...
```

FP4 的 level 非均匀，较多 representational structure 集中在较小 magnitude，同时还能表达不同 exponent region；TensorRT 也明确区分 INT 均匀 quantization 与 FP8/FP4 的非均匀 representable levels。citeturn12search15

但不能推导成“FP4 永远胜 INT4”。当 block 很小、scale 选得好、weight 分布适合均匀量化时，INT4 仍可能非常好。真正决定胜负的是：

\[
\text{codebook}
+
\text{block size}
+
\text{scale type}
+
\text{outlier treatment}
+
\text{kernel}
\]

而不是格式名字。

这也解释 **FP4 为什么极端依赖 block size**：block 越大，一个 outlier 控制越多元素的 scale；block 越小，则量化误差下降，但 metadata、scale load、layout 与硬件成本上升。

2026 年 AMD MI355X 已原生支持 MXFP6/MXFP4，AMD 还公开展示 W\_MXFP4 × A\_MXFP6 的混合方案：用 4-bit weight 保留 bandwidth 优势，但给 activation 额外 2 bits 以恢复 accuracy；其公开实验中，该组合在测试 workload 上接近 MXFP4 throughput、accuracy 更靠近 FP8。这说明未来很可能不是单一 FP4，而是更细的 **W4A6、W4A8、mixed block format**。citeturn16search1turn16search9

## 量化算法：它们究竟分别在修复什么数值问题

从算法角度理解量化，最有价值的问题不是“它是 GPTQ 还是 AWQ”，而是：

> **它认为 naive quantization 的主要误差来自哪里？**

### PTQ、QAT、weight-only 与 W+A 的坐标系

**PTQ（Post-Training Quantization）**：已经有训练好的模型，通过 calibration/optimization 得到 quantized checkpoint，不重新完整训练。

**QAT（Quantization-Aware Training）**：训练或 fine-tuning 时模拟 quantization：

\[
w\rightarrow Q(w)\rightarrow \hat w
\]

forward 感受到量化误差，参数因此学会适应。

**Weight-only quantization**：

\[
W_qA_{BF16}
\]

主要目标是减少 weight HBM traffic，非常适合 autoregressive decode 中矩阵趋向 GEMV/GEMM-small-M 的 memory-bound 场景。

**Weight + Activation quantization**：

\[
W_qA_q
\]

除了节省 memory bandwidth，还可以直接使用更高 throughput 的 low-precision Tensor Core MMA，因此在 compute-bound prefill / high-batch 场景潜力更大。

### 主流算法对比

| 方法 | 它解决的数值问题 | 典型 precision | Calibration / Training | Outlier 方法 | Kernel 加速现实 |
|---|---|---|---|---|---|
| LLM.int8() | activation emergent outliers | W8A8 + FP16 outlier path | PTQ | mixed-precision decomposition | 有专门实现才真正快 |
| SmoothQuant | activation 比 weight 难量化 | W8A8 | training-free calibration | 把难度迁移到 weights | INT8 GEMM 非常硬件友好 |
| GPTQ | weight rounding 导致 layer output error | W4A16 / W3A16 | one-shot calibration | Hessian/second-order proxy | Marlin 等成熟后很实用 |
| AWQ | 少量 salient weights 对 output 影响巨大 | W4A16 | activation calibration，无 backprop | protect salient channels | 很适合 fused weight-only kernels |
| ZeroQuant | W+A fine-grained quant + q/dq overhead | INT8/低 bit | calibration/KD variants | fine granularity | 强调 backend fusion |
| SpQR | 极少 weight outlier 不适合低 bit | ~3–4 bit + sparse high precision | PTQ | isolate outliers | irregular sparse path 增加复杂度 |
| QuIP | 2-bit 下矩阵高度 coherent | ~2–4 bit weight | PTQ | incoherence transform | specialized codebook/kernel |
| QuIP# | 极低 bit rounding/codebook error | ≤4 bit | calibration + light tuning | Hadamard + lattice codebook | kernel specialization要求高 |
| OmniQuant | clipping/scaling 参数选不好 | W4A16/W4A4 等 | calibration optimization | learnable equivalent transform | 最终是否快取决于 export dtype |
| AutoRound | fixed RTN rounding 不是最优 | 2–8 bit | 短 calibration optimization | optimize rounding + clipping | 可导出 GPTQ/AWQ等成熟格式 |
| FP8 PTQ | range/rounding error | W8A8 FP8 | static/dynamic calibration | scaling | Hopper+ 原生 |
| Rotation 类 | outlier/anisotropy | INT4/FP4等 | calibration | Hadamard/rotation 平坦化分布 | 需 fusion 才不被 transform 吃掉 |

LLM.int8()、SmoothQuant、GPTQ、AWQ、ZeroQuant、SpQR、QuIP/QuIP# 的上述核心机制分别来自其原始论文；OmniQuant 与 AutoRound 也分别优化 calibration-time 参数和 rounding/clipping，而非重新做完整模型训练。citeturn6search0turn6search3turn6search5turn6search2turn7search23turn9search0turn7search1turn7search9turn5search1turn8search3

### GPTQ 到底在做什么

最 naive 的 weight quantization 是：

\[
\hat W=Q(W)
\]

希望：

\[
W\approx\hat W
\]

但神经网络真正关心的是 layer output：

\[
XW\approx X\hat W
\]

即最小化：

\[
\|XW-X\hat W\|^2
\]

GPTQ 使用近似二阶信息，在逐列/逐块 quantization 时补偿已经造成的误差，而不是每个 weight 独立 nearest rounding。原始 GPTQ 工作证明大模型可以在 3/4-bit weight-only 下保持较好的质量，并报告了配套 kernel 的实际推理加速。citeturn6search5

因此 GPTQ 真正解决的是：

> **“weight 本身的 L2 rounding error 小”不等于“layer output error 小”。**

### AWQ 到底在做什么

AWQ 的洞察是，并非所有 weight 一样重要。通过 activation statistics 找到少量 salient weight channels，然后通过 equivalent scaling 保护它们，可以显著降低 W4A16 的质量损失；它不需要完整 backprop 或 layer reconstruction。citeturn6search2

因此：

```text
GPTQ:
利用二阶/output-error结构选择 rounding

AWQ:
利用 activation magnitude 识别重要 weight 并保护
```

二者都是 W4A16，但数值假设不同。

### SpQR、QuIP# 为什么越来越“像编码理论”

当 4 bit 继续降到 3、2 bit，传统：

\[
Q(x)=\mathrm{round}(x/s)
\]

越来越不足。

SpQR 认为关键是**极少数特别难量化的 weight**：把它们抽出来用高精度稀疏保存，其余大量权重低 bit。原论文报告了低 perplexity degradation 和 >4× compression，但 irregular sparse metadata/kernel 会使工程实现比普通 dense INT4 复杂。citeturn9search0

QuIP/QuIP# 则认为低比特失败的重要原因之一是 matrix coherence / distribution geometry，因此先用 incoherence processing/Hadamard 类变换把信息“摊平”，再使用 optimized rounding 或 lattice codebook。citeturn7search1turn7search9

这是很重要的趋势：

> **越往 4 bit 以下走，问题越不只是“选 scale”，而开始变成 distribution shaping、rotation、codebook、error feedback 与 hardware layout 的联合设计。**

### “文件更小”与“推理更快”是两个完全不同的问题

假设 FP16 weight：

\[
W\in\mathbb{R}^{4096\times6144}
\]

参数数：

\[
25,165,824
\]

仅 payload：

```text
BF16/FP16 ≈ 50.3 MB
FP8       ≈ 25.2 MB
INT4      ≈ 12.6 MB
```

但 INT4 kernel 如果这样执行：

```text
load packed INT4
      ↓
unpack nibbles
      ↓
load group scales
      ↓
convert INT4 → FP16
      ↓
write temporary FP16
      ↓
call FP16 GEMM
```

那么它可能**显存小，却不快**。

真正好的 kernel 是：

```text
load packed INT4
      │
      ├── scale
      │
      ▼
register / MMA-fragment dequant
      │
      ▼
Tensor Core MMA
      │
      ▼
high-precision accumulator
      │
      ▼
fused bias/activation/quant epilogue
```

即 dequant 不落 HBM、不生成完整 FP16 temporary。

TensorRT 的 weight-only INT4 文档就明确区分 low-precision storage 与实际 GEMM compute；其 explicit quantization 路径允许两个 4-bit values packed 到一个 byte，而运行时可以在 GEMM 路径中 dequant，而不是说明“INT4 文件就等于 INT4 Tensor Core”。citeturn12search2

这也是 Marlin、CUTLASS、FlashInfer 等 kernel 存在的意义。

## Hardware × GEMM × Kernel：低 bit 为什么会变快，也为什么经常没那么快

低精度同时影响四个层面：

\[
\boxed{
\text{capacity}
+
\text{bandwidth}
+
\text{compute throughput}
+
\text{cache residency}
}
\]

### GPU × datatype 支持矩阵

下面的矩阵强调 **Tensor Core / matrix-core native compute**，而不是“软件能否存这种 dtype”。代表 GPU 的 peak 数字采用 dense matrix multiply 口径；不同 SKU、功耗和 sparsity 模式不能直接横比。Volta/Turing/Ampere/Hopper/Blackwell 的格式能力来自 NVIDIA 架构资料，2026 年 Google Cloud 规格表提供了 V100/T4/A100/H100/B200 等 representative dense peaks。citeturn15search0turn15search7turn15search8turn15search5turn17search2

| 架构 / 代表 GPU | FP64 TC | TF32 | FP16 | BF16 | FP8 | INT8 | INT4 | FP4 | 代表性低精度 dense peak |
|---|---|---|---|---|---|---|---|---|---|
| Volta / V100 | — | — | ◎ | — | — | — TC | — TC | — | FP16 mixed ≈125 TFLOPS |
| Turing / T4 | — | — | ◎ | — | — | ◎ | ◎ | — | FP16 65T / INT8 130T / INT4 260T |
| Ampere / A100 | ◎ | ◎ | ◎ | ◎ | — | ◎ | ◎ | — | FP16/BF16 312T / TF32 156T / INT8 624T |
| Ada / L4/L40S | —主流 | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | — | L4 FP8 dense ≈121T；L40S 更高 |
| Hopper / H100 | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | 非主力路径 | — | FP8/INT8 ≈1979T；FP16 ≈989.5T |
| Blackwell / B200 | ◎/产品相关 | ◎ | ◎ | ◎ | ◎ | ◎ | expanded | ◎ | FP4 ≈9 PFLOPS；FP8 ≈4.5 PFLOPS |
| Blackwell Ultra / B300 | 产品相关 | ◎ | ◎ | ◎ | ◎ | ◎ | expanded | ◎ | FP4 进一步强化 |
| Rubin / SM107 | 公共资料持续展开 | ◎族系 | ◎族系 | ◎族系 | ◎ | — | — | ◎ | NVIDIA 公布 NVFP4 Transformer Engine 达 50 PFLOPS/GPU 级别 |

H100 官方产品规格也显示 FP8 Tensor Core peak 约为 FP16/BF16 的 2×；Blackwell 系统规格进一步给出 FP4 相对 FP8 又约 2× 的 peak-rate 层级。Rubin 的公开 Tensor Core 页面截至 2026 年已宣传增强的 NVFP4 Transformer Engine，而 CUTLASS 4.4 已有 Rubin SM107 初始 FP8/FP4 Tensor Core MMA 支持；这里应把“公开 SDK 已支持”与“整个平台所有 dtype、所有 kernel production-ready”区分开。citeturn15search2turn15search14turn15search20turn14search10

AMD 方向，MI355X/CDNA4 公开支持 MXFP4/MXFP6，并具有 288 GB HBM3E 与 8 TB/s 带宽；ROCm 已展示 gfx950 直接消费 FP4 payload 与 E8M0 scales 的 scaled MFMA，而不是先完整 dequant 回 BF16。citeturn16search1turn16search5

Intel Gaudi 3 具有原生 E4M3/E5M2 FP8、BF16/FP16/FP32 MME 路径；Intel 的公开部署资料给出 8 个 MME、64 个 TPC 和 128 GB HBM2e。citeturn16search3

Google TPU7x/Ironwood 截至 2026 年 8 月已在 Google Cloud 公开：每 chip BF16 peak 2307 TFLOPS、FP8 peak 4614 TFLOPS、192 GiB HBM 和约 7.38 TB/s bandwidth，体现同样的 8-bit matrix-compute 趋势。citeturn17search0turn17search3

### 为什么 bit width 能影响这么多性能指标

假定需要从 HBM 读取 \(N\) 个 weights：

\[
T_{\mathrm{memory}}
\approx
\frac{N\cdot b/8}{BW_{\mathrm{HBM}}}
\]

从 BF16：

\[
b=16
\]

变 FP8：

\[
b=8
\]

理想 weight traffic 减半。

变 4 bit：

\[
b=4
\]

理想 payload traffic 再减半。

但真实还有：

\[
B_{\mathrm{real}}
=
B_{\mathrm{payload}}
+
B_{\mathrm{scale}}
+
B_{\mathrm{zero}}
+
B_{\mathrm{metadata}}
+
B_{\mathrm{padding}}
\]

因此 4-bit block format 不等于准确的 “4 bits/parameter”。

更低 bit 还意味着同样 L2/cache 能容纳更多 weights：

\[
\text{effective cache capacity in parameters}
\propto\frac1{\text{bits/weight}}
\]

也能增加单卡 KV capacity，从而提高：

- maximum context；
- concurrent sequences；
- continuous batching；
- batch size；
- 因 batch 增大导致的 Tensor Core utilization。

### memory-bound 和 compute-bound 必须先分清

Roofline 的核心是：

\[
P
\le
\min
\left(
P_{\mathrm{peak}},
BW\cdot AI
\right)
\]

其中 arithmetic intensity：

\[
AI=
\frac{\text{operations}}
{\text{bytes transferred}}
\]

**LLM decode、小 batch** 中，矩阵经常趋向 GEMV/small-M GEMM。每生成一个 token，要把大量 weight 从 HBM 读一遍，而每个 weight 参与的复用不多，因此通常偏 memory-bound。

此时：

\[
BF16\rightarrow INT4
\]

最大的价值通常是**少读 4× weight bytes**，不是 Tensor Core 的 “TOPS 多了 4×”。

相反在长 prompt prefill 或 large-batch serving 中：

\[
M=B\times T
\]

明显增大，weights 被更多 token 重用，AI 上升，GEMM 更 compute-bound。此时 FP8/FP4 如果有 native Tensor Core，可以真正利用更高 peak FLOPS。

所以：

```text
decode / batch≈1:
memory bandwidth > raw FLOPS

prefill / large batch:
Tensor Core throughput越来越重要
```

Google TPU 和 NVIDIA GPU 的官方架构资料也都明确强调低精度既减少 HBM pressure，也增加 matrix compute throughput。citeturn17search3turn15search8

### GEMM kernel 中低精度到底发生在哪里

以 scaled FP8 为例：

```text
HBM
│
├─ A: FP8 payload
├─ B: FP8 payload
├─ scale_A
└─ scale_B
        │
        ▼
      SMEM
        │
        ▼
 Tensor Core MMA
        │
  FP32 accumulator
        │
        ▼
   epilogue
   × scales
   + bias
   + activation
        │
        ▼
    BF16 / FP8 C
```

最优情况下：

- scale load 被 tile 化；
- dequant/scale 与 MMA 融合；
- accumulator 留在 register；
- bias、SiLU/GELU、requant 在 epilogue 完成；
- intermediate 不回 HBM。

CUTLASS 就是描述和生成这类 tile-level mainloop/epilogue 的核心 CUDA 模板库；Blackwell 当前 CUTLASS 已提供 NVFP4、MXFP8/MXFP6、mixed block-scaled GEMM，2026 年还增加 Rubin FP8/FP4 Tensor Core MMA。citeturn0search4turn14search10

**cuBLASLt** 更像 vendor-optimized GEMM dispatch/API；**CUTLASS** 给你更底层的 kernel building blocks；**Triton** 允许用 Python-like DSL 写定制 tile kernel；**Transformer Engine** 在更高层管理 FP8/FP4 recipe、scales 和 Transformer modules；**FlashInfer** 针对 LLM serving 的 attention、MoE、quantized GEMM 等路径；**Marlin** 等则解决 packed low-bit weight-only GEMM 的硬件利用率问题。vLLM/SGLang 的价值之一正是根据 checkpoint、GPU 与 shape 选择这些不同 kernel backend。citeturn10search0turn10search10turn11search4

### native support 与 software emulation 的分界

真正 native：

```text
low-bit HBM payload
        ↓
native load/layout
        ↓
hardware low-bit MMA
```

software-emulated：

```text
low-bit HBM payload
        ↓
unpack/dequant instructions
        ↓
FP16/BF16 values
        ↓
FP16/BF16 Tensor Core
```

后者仍能由于减少 HBM traffic 获益，但没有完整享受到 low-bit MMA peak，而且 unpack/dequant 会占：

- integer/FP ALU；
- registers；
- shared memory；
- instruction bandwidth；
- latency。

这就是为什么某些 INT4 模型 **比 FP16 小 4×，但 tokens/s 甚至更差**。

一个非常好的现实例子是 Mistral 的官方 NVFP4 模型卡：其明确提醒，在非 Blackwell GPU 上可以通过 vLLM/Marlin fallback 运行 FP4 checkpoint、获得 memory gain，但不一定获得相对 FP8 的 speedup。这恰好说明“格式可加载”和“native execution”不是一回事。citeturn19search12

## LLM 实际 Precision Map：KV Cache、MoE、模型文件与训练/推理

### 一张典型 Transformer Decoder precision map

以下不是某个统一标准，而是一个 **Hopper/Blackwell FP8/FP4 serving system 的合理典型设计**。实际 checkpoint 与 engine 可能不同；Transformer Engine、TensorRT-LLM 和当前公开 quantized checkpoints 都体现了这种 operator-specific mixed precision。citeturn0search13turn10search1turn18search27

```text
Token IDs
   │
   ▼
Embedding
weight: BF16 / FP16 / quantized
output: BF16
   │
   ▼
RMSNorm
input: BF16
square/reduce: FP32
output: BF16
   │
   ▼
QKV GEMM
weight: FP8 / INT4 / FP4
activation: BF16 → dynamic FP8/FP4
MMA: low precision
accumulator: FP32
output: BF16
   │
   ├────────────► K/V → optional FP8 / INT8 / FP4 KV cache
   │
   ▼
RoPE
BF16 / FP32-sensitive math
   │
   ▼
Q × Kᵀ
BF16 / FP8 / quantized-KV path
higher precision accumulation
   │
   ▼
scale by 1/sqrt(d)
   │
   ▼
Softmax
max: FP32
exp: FP32/BF16 implementation
sum: FP32
prob output: BF16/FP16
   │
   ▼
Attention Prob × V
V: BF16 / FP8 / INT8...
accumulator: FP32
output: BF16
   │
   ▼
Output Projection
low-bit weight / activation
FP32 accumulation
BF16 output
   │
   ▼
RMSNorm
FP32 reduction
   │
   ▼
MLP / MoE
gate/up/down weights: FP8 / INT4 / FP4
activation: BF16 → low-bit
accumulator: FP32
   │
   ├─ if MoE:
   │    Router logits: BF16/FP32
   │    top-k: higher precision
   │    Experts: low-bit grouped GEMM
   │
   ▼
Residual add: BF16 / FP32-sensitive path
   │
   ▼
Final Norm
   │
   ▼
LM Head
large GEMM may be low-bit
logits: BF16 / FP32
   │
   ▼
Sampling / argmax
```

因此一句：

> “DeepSeek-V3 是 FP8”

实际上至少可能意味着：

```text
某些 weights       FP8
某些 activations   FP8
scales             FP32/BF16
GEMM accumulators  FP32
Norm               BF16/FP32
attention internals mixed
router             higher precision
checkpoint metadata additional
```

DeepSeek-V3 的 FP8 framework 就使用 fine-grained scaling：activations 可采用约 1×128 tile granularity，weights 采用 128×128 block granularity，而不是整个模型统一一个 scale。citeturn18search0turn18search4

### KV Cache 是完全独立的量化问题

对于 standard attention/GQA，KV cache 每 token 的 payload 大致是：

\[
M_{KV/token}
=
2
\times
L
\times
N_{KV}
\times
D_h
\times
\text{bytes}
\]

其中前面的 2 分别代表 K 与 V。

举例：

```text
layers       = 32
KV heads     = 8
head_dim     = 128
dtype        = BF16 = 2 bytes
```

则：

\[
2\times32\times8\times128\times2
=131072\text{ bytes}
\]

即：

\[
128\ KiB/token
\]

单 sequence 128K context：

\[
128\ KiB\times131072
\approx16\ GiB
\]

只算 KV payload。

换 FP8：

\[
\approx8\ GiB
\]

理论 INT4/FP4 payload：

\[
\approx4\ GiB
\]

再加 scales、block tables、alignment 等。

这就是长上下文场景中，KV precision 可能比 weight quantization 更影响 concurrency 的原因。

权重 quantization 与 KV quantization 不同：

```text
Weight:
一次离线 calibration
固定
跨所有 request 重复使用

KV:
每个 token 在线生成
值随 prompt/context 改变
不断增长
每一 attention step 都会重复读取
```

因此 KV quantization 更依赖：

- dynamic scale；
- per-token/per-head granularity；
- attention kernel 是否直接消费 quantized cache；
- dequant 是否 fused 到 attention；
- K 和 V 是否应该采用相同 quantizer。

截至当前 vLLM 文档，FP8 KV cache 已是正式功能；其内部接口也出现了 per-tensor FP8、per-token-head INT8/FP8 以及 NVFP4/TurboQuant 等更细 quantization modes，说明 KV quantization 正从“只 FP8”快速演化成独立技术栈。citeturn20search0turn20search9turn20search11

粗略工程对比：

| KV dtype | Payload vs BF16 | Accuracy 风险 | Runtime overhead | 适合 |
|---|---:|---|---|---|
| BF16 | 1× | 最低 | 低 | baseline |
| FP16 | 1× | 低，但 range 较小 | 低 | FP16 stack |
| FP8 | ~0.5× | 较低～中 | scale/dequant | H100+ 长 context |
| INT8 | ~0.5× | 中 | dynamic quant/dequant | 有专用 attention kernel |
| INT4 | ~0.25× | 高 | 明显 | 极长 context / memory pressure |
| FP4/NVFP4 | ~0.25× + scales | 高但 fine scaling 可改善 | 依赖 native kernel | Blackwell 新路径 |

“payload ratio”是按 bit width 的理论值；实际 allocation 会受 scales/paging/layout 影响。vLLM 官方目前特别强调 FP8 KV 对可缓存 token 数与长上下文 throughput 的价值。citeturn20search0

### MoE 为什么又有一张自己的 precision map

MoE：

\[
p=\mathrm{softmax}(r(x))
\]

\[
\mathcal E=\mathrm{TopK}(p)
\]

然后只执行选中的 experts。

如果两个 expert router logits：

\[
r_1=2.001,\quad r_2=2.000
\]

低精度扰动后：

\[
\hat r_1=1.999,\quad
\hat r_2=2.002
\]

就不是“输出有一点误差”，而是：

```text
Expert A  → Expert B
```

整个后续计算路径改变。

所以 router 是一个**离散敏感点**。

而 expert weights 恰好相反：它们占模型绝大部分 capacity，每 token 只激活少数 experts，是非常值得 FP8/FP4/INT4 化的部分。

典型 MoE precision map：

```text
hidden state: BF16
     │
     ▼
router GEMM: BF16/FP32-sensitive
     │
gate logits: BF16/FP32
     │
top-k selection
     │
     ├── token dispatch / All-to-All
     ▼
experts:
FP8 / INT4 / MXFP4 weights
     │
grouped GEMM
high precision accumulation
     │
combine
     ▼
BF16 hidden state
```

SGLang 当前 expert-parallel backend 已有 DeepGEMM FP8 blockwise，以及针对 Blackwell FP4/MXFP4/NVFP4 的 FlashInfer 等 grouped-GEMM 路径，体现了 MoE low precision 与 grouped GEMM/All-to-All 的共同优化。citeturn11search4

AMD 2026 的 MI355X 部署甚至在 MXFP4 MoE 模型上采用 **FP4 dispatch + FP8 combine** 以减少 All-to-All traffic，再次说明通信 dtype 本身也是 precision map 的组成部分。citeturn16search16

### 模型文件里的 dtype 不等于 GPU 执行 dtype

一个典型 INT4 GPTQ checkpoint：

```text
disk
│
├─ packed INT4 weight bytes
├─ scales
├─ zero points / metadata
└─ quant config
      │
      ▼
GPU memory
packed INT4 remains packed
      │
      ▼
kernel
load packed bytes
      │
unpack nibble
      │
apply group scale / zero
      │
      ├── native mixed low-bit MMA
      │       or
      └── dequant inside registers
              ↓
          FP16/BF16 MMA
      │
      ▼
FP32 / high precision accumulator
      │
      ▼
BF16 output
```

所以 safetensors/Hugging Face checkpoint 中看到 `torch.float16`、`torch.bfloat16`、float8 bytes 或 packed integers，只能说明**持久化表示的一部分**。实际 GPU execution 还由：

- `quantization_config`；
- checkpoint metadata；
- engine loader；
- architecture；
- kernel backend；
- GEMM compute type

共同决定。

TensorRT 的 explicit quantization 文档明确展示了低精度 weight storage 与运行时 dequant/GEMM 的分离；Hugging Face Transformers 当前也把 AWQ、GPTQ、AutoRound、bitsandbytes、MXFP4、FP8 等作为不同量化 integration，而非把它们等价成一个 PyTorch dtype。citeturn12search2turn8search7

## 当前模型与推理引擎：checkpoint 支持不等于 kernel 支持

### 代表模型案例

**DeepSeek-V3。** 这是 FP8 精度工程的标志性案例：其技术报告公开了大规模 FP8 mixed-precision training，并采用 fine-grained quantization 解决 per-tensor FP8 的 outlier/range 问题。其 activation 采用细粒度 tile scaling、weights 采用 block scaling，而高敏感算子仍使用更高精度。citeturn18search0turn18search4

**Qwen。** Qwen3 以及 2026 年 Qwen3.8 系列持续提供官方 FP8 checkpoints；官方模型卡明确标注 fine-grained FP8、block size 128，并直接声明 vLLM/SGLang/Transformers 等部署兼容性。citeturn18search14turn18search6turn18search2

**Llama。** Meta 的 Llama 4 model card 表明 Scout 原始发布为 BF16，并提供 on-the-fly INT4 路径；Maverick 同时发布 BF16 和 FP8 weights。NVIDIA 也为 Llama 3.1/3.3/4 提供 TensorRT Model Optimizer 生成的 FP8 checkpoints。Meta Llama 3.1 405B 的 FP8 方案并非“所有 tensor 全 FP8”，而是主要量化 Transformer linear operators；NVIDIA 对 405B checkpoint 的模型卡报告约 50% weight memory reduction，并在其 H200 测试中得到约 1.7× speedup。citeturn18search35turn18search7turn18search27

**Mistral/Mixtral 系。** 早期 Mixtral 更常作为 BF16 base + 社区 AWQ/GPTQ/INT4 的案例；到 Mistral 2025–2026 新模型，官方已经直接发布 FP8 与 NVFP4 checkpoints。例如 Ministral 3 8B 官方为 FP8；Mistral Small 4 119B 提供 NVFP4 checkpoint。citeturn19search0turn19search8

**GLM。** Z.ai 官方已有 GLM-4.5/4.5V/4.7 FP8 checkpoint；GLM-4.5 本身是 355B total、32B activated 的 MoE，因此尤其适合研究“低精度 expert + higher-precision router”的部署结构。citeturn22search4turn22search1turn22search13

**Gemma。** Google 当前 Gemma 4 同时覆盖 dense/MoE 与多种规模，但它更适合作为“高精度 base checkpoint → downstream engine quantization”的案例，而不是把某个统一官方 FP8/FP4 policy 当作 Gemma 系列定义。citeturn22search2

这些案例揭示了产业变化：**FP8 正逐渐成为官方 checkpoint format；FP4/MXFP4/NVFP4 则开始从第三方 PTQ 走向模型发布方直接提供。**

### inference engine × quantization 支持矩阵

截至 **2026-09-06**，必须区分四层“支持”：

```text
A. loader 能读 checkpoint
B. 有 correctness implementation
C. 有该 GPU 的 optimized kernel
D. production serving 已成熟
```

下面：

- **◎**：官方文档明确、有优化执行路径；
- **○**：支持但硬件/模型/backend 有限制；
- **△**：实验性、fallback、转换或版本依赖较强；
- **—**：当前不是主要官方路径。

矩阵基于各项目当前官方文档/代码状态，而不是仅看 Hugging Face 能否下载文件。citeturn10search0turn11search2turn10search1turn8search7turn13search1turn13search2turn12search19turn20search1

| Engine | BF16/FP16 | FP8 W8A8 | INT8 | AWQ | GPTQ | INT4/W4A16 | NVFP4 | MXFP4 | FP8 KV | INT8/更低 KV | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **vLLM** | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | ◎/○ | ◎/○ | ◎ | ○ | LLM Compressor、Marlin、Quark 等多 backend |
| **SGLang** | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | ◎ | ◎/○ | ○ | FP4/MXFP4 MoE backend 很积极 |
| **TensorRT-LLM** | ◎ | ◎ | ◎ | ◎/○ | ○ | ◎ | ◎ | ◎ | ◎ | NVFP4 KV 等模型依赖 | NVIDIA production-focused |
| **Transformers** | ◎ | ○ | ○ | ◎ | ◎ | ◎ | ○ | ◎/○ | backend-dependent | backend-dependent | “加载支持”强，性能取决于 backend |
| **llama.cpp** | ◎ | △ | GGUF 路径 | 需转换 | 需转换 | ◎ GGUF | △ | ◎ | evolving | evolving | CPU/Metal/Vulkan/CUDA 与 GGUF 生态不同 |
| **MLC** | ◎ | ◎ CUDA 路径 | ○ | △ | —主路径 | ◎ q4 | —主路径 | —/版本依赖 | limited | limited | compiler-generated kernels |
| **TensorRT** | ◎ | ◎ | ◎ | 不是算法级接口 | 不是算法级接口 | ◎ | ◎ FP4 | MX/block 路径依版本 | application-specific | application-specific | 通用 inference runtime |
| **DeepSpeed** | ◎ | △/生态依赖 | ○ | 非主路径 | 非主路径 | compression-related | —主路径 | —主路径 | —主路径 | —主路径 | 当前强项更偏训练/分布式 |

vLLM 当前量化文档列出了 AutoAWQ、GPTQModel、BitsAndBytes、LLM Compressor FP8/INT4/INT8、NVIDIA Model Optimizer、AMD Quark 与 quantized KV cache；Marlin 路径覆盖多种 low-bit checkpoint。citeturn10search0turn10search10turn20search17

SGLang 当前 server arguments 已直接列出 `fp8`、`mxfp8`、`gptq`、`marlin`、`awq`、`modelopt_fp4`、`nvfp4_online`、`mxfp4`、`auto-round`、`w8a8_fp8`、`w4afp8` 等；但其 roadmap 同样显示部分 vector/codebook quantization 仍在演进，所以不能把命令行出现一个名字等同于所有模型 production-ready。citeturn11search2turn11search3

TensorRT-LLM 2026 年量化矩阵已覆盖 NVFP4、MXFP4、FP8、FP8 KV 与多种 W4Ax，并明确按模型区分支持情况；最新 release notes 还出现 Blackwell CuteDSL NVFP4 grouped GEMM、MXFP8×MXFP4 MoE 等路径。citeturn10search1turn10search4

llama.cpp 已对 MXFP4 有原生生态支持，包括 GPT-OSS MXFP4 的 CUDA/Vulkan/Metal/CPU 路径；2026 年 CUDA MXFP4 KV/SM120 FP4 类优化仍可看到持续开发，因此尤其需要区分 merged stable path 和实验 PR。citeturn13search1turn13search14

MLC 当前官方量化文档则明确提供 q3/q4/AWQ 以及 CUDA FP8 E4M3/E5M2 W+A 模式，属于 compiler-centric 的另一种低精度实现路线。citeturn13search2

### 为什么引擎支持矩阵会快速失效

因为一个“quantization format”通常同时绑定：

```text
checkpoint schema
+ scale layout
+ group size
+ tensor layout
+ GPU architecture
+ kernel implementation
+ attention backend
+ MoE backend
```

举例：

```text
MXFP4 model
```

可能在：

- Blackwell：native block-scaled Tensor Core；
- Hopper：Marlin/fallback；
- AMD MI355X：gfx950 scaled MFMA；
- CPU：GGUF-oriented unpack/dequant；

得到完全不同 performance。

因此工程上不要维护：

```text
Model → dtype
```

而要维护：

```text
Model
× checkpoint format
× GPU
× engine version
× kernel backend
× workload shape
→ measured result
```

## Accuracy、性能 Benchmark 与真正的 Precision Engineering

### 数值误差必须分层测

**Tensor-level** 最基础：

绝对误差：

\[
e_i=\hat x_i-x_i
\]

相对误差：

\[
r_i=
\frac{|\hat x_i-x_i|}
{|x_i|+\epsilon}
\]

MSE：

\[
MSE=
\frac1N\sum_i(\hat x_i-x_i)^2
\]

cosine similarity：

\[
\cos(x,\hat x)
=
\frac{x^T\hat x}
{\|x\|\|\hat x\|}
\]

SQNR：

\[
SQNR=
10\log_{10}
\frac{\|x\|_2^2}
{\|x-\hat x\|_2^2}
\]

这些适合比较：

```text
BF16 GEMM output
vs
FP8/INT4/FP4 GEMM output
```

但仍不够。

对于 attention distribution：

\[
D_{KL}(P\|Q)
=
\sum_iP_i\log\frac{P_i}{Q_i}
\]

往往比简单 MSE 更能暴露 softmax 概率结构变化。

对于语言模型还应测：

logit difference：

\[
\Delta z=z_q-z_{ref}
\]

top-token margin：

\[
m=z_{(1)}-z_{(2)}
\]

perplexity：

\[
PPL=
\exp
\left[
-\frac1N
\sum_t
\log p(x_t|x_{<t})
\right]
\]

以及：

- token agreement；
- top-k agreement；
- long-context retrieval；
- task benchmark accuracy；
- reasoning/coding benchmark；
- calibration-set 与 out-of-distribution prompts。

### 为什么 kernel MSE 很小仍可能改变模型答案

假设：

```text
BF16 logits:
cat  = 10.001
dog  = 10.000

quantized:
cat  = 9.999
dog  = 10.002
```

整个 logit vector cosine similarity 可能仍接近 1，但 greedy token 已经从 `cat` 变成 `dog`。

下一 token 计算的条件从：

\[
p(x_{t+1}|...,cat)
\]

变成：

\[
p(x_{t+1}|...,dog)
\]

此后两条 trajectory 完全不同。

所以 autoregressive decoding 的误差传播是：

```text
small numeric error
      ↓
small logit perturbation
      ↓
top-1 crossing
      ↓
different token
      ↓
different context
      ↓
different KV cache
      ↓
all future hidden states diverge
```

这不是简单的 floating-point error amplification，而是**连续数值系统跨过离散 decoding boundary**。

因此“bitwise/tokenwise 完全一致”并不是合理的唯一量化标准；真正应判断的是 output distribution、PPL、benchmark quality 和实际任务成功率。

### 公平的 BF16/FP8/INT8/INT4/FP4 benchmark 应如何设计

首先固定：

```text
same model architecture
same tokenizer
same prompts
same output lengths
same sampling settings
same GPU(s)
same TP / PP / EP
same scheduler
same maximum context
same CUDA / ROCm stack
same engine commit/version
same attention backend
same speculative decoding setting
```

量化 checkpoint 如果算法不同，必须记录：

```text
quant algorithm
calibration dataset
number of calibration tokens
group/block size
scale dtype
symmetric/asymmetric
outlier policy
excluded layers
KV dtype
```

否则你测到的是“算法 A + kernel A”对“算法 B + kernel B”，不是简单 dtype 对比。

至少测以下三层。

**Capacity：**

\[
M_{\mathrm{total}}
=
M_{\mathrm{weights}}
+
M_{KV}
+
M_{\mathrm{runtime}}
+
M_{\mathrm{workspace}}
+
M_{\mathrm{allocator}}
\]

不能只报 Hugging Face 文件大小。

**Latency：**

- TTFT：Time To First Token；
- TPOT：Time Per Output Token；
- ITL：Inter-token latency；
- P50/P95/P99；
- prefill latency；
- decode latency。

**Throughput：**

- tokens/s/GPU；
- request/s；
- output tokens/s；
- batch throughput；
- concurrency at SLO。

再加 hardware metrics：

\[
\text{Tensor Core utilization}
\]

\[
\text{HBM bandwidth utilization}
\]

\[
\text{L2 hit rate}
\]

\[
\text{power}
\]

\[
E_{\mathrm{token}}
=
\frac{\text{Joules}}
{\text{tokens}}
\]

最终商业指标：

\[
\boxed{
\text{cost/token}
=
\frac{
\text{GPU cost/time}
+
\text{power}
+
\text{infra overhead}
}{
\text{tokens/time}
}
}
\]

这才是 low precision 的终局指标。

### 推荐的一组 benchmark matrix

例如对同一模型：

| Case | Weight | Activation | KV | Accum | 目的 |
|---|---|---|---|---|---|
| Baseline | BF16 | BF16 | BF16 | FP32 | reference |
| FP8 | FP8 | FP8 | BF16 | FP32 | compute + weight BW |
| FP8+KV | FP8 | FP8 | FP8 | FP32 | long context |
| INT8 | INT8 | INT8 | BF16 | INT32/FP32 | integer Tensor Core |
| W4A16 | INT4 | BF16 | BF16 | high precision | decode memory-bound |
| W4A8 | INT4 | FP8/INT8 | FP8 | high precision | mixed serving |
| FP4 | NVFP4/MXFP4 | FP4/mixed | BF16 | high precision | Blackwell native |
| FP4+KV | FP4 | FP4/mixed | quantized | high precision | max capacity |

然后至少跑四种 workload：

```text
short prompt / short output
long prompt  / short output
short prompt / long output
long prompt  / long output
```

以及 concurrency sweep：

```text
1 → 2 → 4 → 8 → 16 → 32 → 64 → ...
```

这样才能分别看到：

- single-request latency；
- memory-bound decode；
- compute-bound prefill；
- continuous batching；
- KV capacity ceiling。

## 工程决策树、统一认知框架与进一步阅读

### 一棵实际可用的 Precision Decision Tree

```text
                       拿到一个新 LLM
                             │
                             ▼
                 GPU / accelerator 是什么？
                             │
            ┌────────────────┼─────────────────┐
            │                │                 │
       Ampere/older       Hopper/Ada      Blackwell/MI355X+
            │                │                 │
            ▼                ▼                 ▼
      BF16/FP16 baseline   FP8 native?       FP4/MX native?
            │                │                 │
            │                yes               yes
            │                │                 │
            ▼                ▼                 ▼
   模型是否能放进 HBM？   先 benchmark FP8   compare FP8 vs FP4/MX
        │          │          │                 │
       yes        no          │                 │
        │          │          ▼                 ▼
        │       W8/W4      accuracy OK?      accuracy OK?
        │      weight-only    │    │            │    │
        │          │         yes   no           yes   no
        │          │          │    │            │    │
        ▼          ▼          ▼    ▼            ▼    ▼
 latency-sensitive?   FP8 deploy  BF16    FP4 candidate FP8/BF16
        │
    ┌───┴────┐
   yes       no
    │         │
    ▼         ▼
 low batch  high batch
 decode     throughput
    │         │
    ▼         ▼
检查 memory-   检查 Tensor Core
bound?         compute-bound?
    │              │
   yes            yes
    │              │
    ▼              ▼
W4A16/FP4      FP8/FP4 W+A
价值很高        native MMA价值高
    │              │
    └──────┬───────┘
           ▼
       context 很长？
        │       │
       yes      no
        │       │
        ▼       ▼
   quantize KV  KV BF16 baseline
        │
        ▼
注意 attention kernel 是否直接支持
        │
        ▼
      是 MoE？
        │
       yes
        │
        ▼
experts低精度
router/logits高精度
检查 grouped GEMM + All-to-All
        │
        ▼
kernel 是 native 还是 fallback？
        │
    ┌───┴────┐
 native    dequant→FP16
    │          │
    ▼          ▼
继续评估     不要相信 dtype 名字
    │
    ▼
Accuracy:
PPL / task / long context / token agreement
    │
    ▼
Performance:
TTFT / TPOT / throughput / HBM / power
    │
    ▼
              cost/token 最低？
                    │
               ┌────┴────┐
              yes        no
               │          │
               ▼          ▼
             deploy    换 precision /
                       kernel / engine
```

对 NVIDIA GPU，可以把它进一步简化为：

**V100/Turing：** FP16 是稳妥 Tensor Core baseline；Turing 上 INT8/INT4 有硬件路径，但现代 LLM kernel 兼容性必须验证。citeturn17search2turn15search7

**A100：** BF16 是很好的默认值；INT8 与 INT4 weight-only 对 memory-bound inference 很有价值，但 A100 没有 Hopper-style native FP8，因此“FP8 checkpoint”可能主要获得 storage benefit 或走 fallback，而不是 H100 的原生 W8A8 FP8 路径。citeturn15search8turn20search7

**H100/H200：** 优先认真评估 FP8。Hopper 原生支持 E4M3/E5M2 FP8 Tensor Core，且 FP8 peak 约为 FP16/BF16 的 2×；如果显存压力进一步提高，再比较成熟的 INT4/W4A16 kernels。citeturn15search5turn15search2

**B200/B300：** FP8 不再自动是终点。NVFP4/MXFP4 已成为真正 native 的工程选项，尤其适合大模型/MoE/高 concurrency；但必须同时跑 accuracy 与 kernel benchmark。citeturn15search25turn15search14

**MI355X：** MXFP4/MXFP6 已是 native datatype，而 ROCm 2026 的实例表明 mixed W4A6 可能成为 accuracy/throughput 的实用中间点。citeturn16search1turn16search9

### 面对未来 FP3、MXFP3、新 Tensor Core datatype，应如何自己判断

不要问：

> “FP3 比 FP4 好不好？”

按下面十五个问题依次检查。

**表示层**

1. **bit 如何分配？**

\[
b=1+e+m
\]

还是 integer/codebook/index？

2. **numerical range 多大？**

不要只看 payload。若有 scale，真正 range 是：

\[
R_{\mathrm{effective}}
=
R_{\mathrm{element}}
\times
R_{\mathrm{scale}}
\]

3. **precision 多高？**

看 ULP、relative error、codebook spacing，而非只看 bit width。

4. **是否有 subnormal、Inf、NaN、signed zero、saturation？**

E4M3FN 与 E4M3FNUZ 已证明两个看似相同的 “E4M3” 可以拥有不同 exponent bias/zero/NaN 规则。citeturn21search5

**Scaling 层**

5. scale 是：

```text
FP32?
BF16?
E8M0?
E4M3?
integer exponent?
```

6. granularity：

\[
1,\ 16,\ 32,\ 64,\ 128,\ channel,\ token?
\]

7. static/current/delayed？

8. 是否二级 scale？

例如 NVFP4：

\[
x
\approx
S_{\mathrm{tensor}}
S_{\mathrm{block}}
P_{FP4}
\]

而 MXFP4 是：

\[
x\approx S_{E8M0,b}P_{E2M1}
\]

两者 payload 都 4-bit，却是不同数值系统。citeturn0search2turn2view0

**算子层**

9. 哪些 tensors 使用？

```text
weight?
activation?
gradient?
KV?
router?
logits?
communication buffer?
```

10. accumulator 是什么？

这是常被 marketing table 隐藏、却最影响 numerical stability 的字段。

**硬件层**

11. accelerator 有 native MMA 吗？

问的是：

```text
native payload → Tensor Core
```

而不是：

```text
能不能把这种 dtype 存在显存
```

12. throughput 真提高多少？

理论：

\[
P_{low}/P_{BF16}
\]

只是 upper bound。

**Kernel 层**

13. 是否需要 unpack/dequant？

如果需要：

```text
是否 fused？
是否落 HBM？
scale 是否成为额外 load？
register pressure 如何？
tile shape 合适吗？
```

**模型层**

14. accuracy loss 在哪里出现？

不要只测 tensor MSE，要测：

\[
\text{PPL}
,\quad
\text{KL}
,\quad
\text{token agreement}
,\quad
\text{task accuracy}
\]

尤其分别看：

- attention；
- router；
- long context；
- rare outliers；
- early/late layers。

**系统层**

15. 最终是否降低：

\[
\boxed{\$/token}
\]

如果一个新的“3-bit”格式使文件小 25%，但：

- quant/dequant 多 30% instructions；
- kernel occupancy 降；
- accuracy 导致模型必须换大一档；
- engine 只能 experimental；
- TPOT 没改善；

那么它在生产系统里可能毫无价值。

### 最重要的 Takeaway

1. **bit width 不是 numerical precision。** exponent 决定 range，mantissa/codebook 决定 local precision；BF16 与 FP16 都是 16 bit，但数值性格完全不同。citeturn4search2turn4search8

2. **BF16 成功的关键不是“更准”，恰恰是 range 大。** 它比 FP16 mantissa 更少，但保留 FP32-like exponent range，因此训练更稳。citeturn4search6

3. **低精度成功的核心是 mixed precision，而非 low precision everywhere。** GEMM operands 可以很低，reduction/accumulator/Norm/Softmax/router 往往需要高得多的精度。citeturn14search12turn18search28

4. **accumulator precision 与 operand precision 必须分开看。** `FP8 × FP8` 不意味着“整个 GEMM 都是 8 bit”。

5. **FP8 的真正技术不是 E4M3/E5M2 本身，而是 scaling。** 没有合适 scale，一个 outlier 就足以浪费大半 representable range。citeturn0search11turn0search2

6. **scaling granularity 是低精度时代最重要的超参数之一。** 从 per-tensor 到 per-block，accuracy 通常改善，但 metadata 与 runtime complexity 上升。

7. **E4M3 与 E5M2 是 range/precision trade-off。** E4M3 精度更高，E5M2 range 更大；block scaling 会进一步改变这一选择。citeturn0search11turn0search5

8. **activation outlier 是 Transformer INT quantization 的核心难题。** LLM.int8 选择绕开 outlier；SmoothQuant 选择把 activation difficulty 转移给 weights。citeturn6search0turn6search3

9. **GPTQ、AWQ 的意义不在“都是 INT4”。** GPTQ 优化 output-error-aware rounding；AWQ 保护 activation-sensitive salient weights。citeturn6search5turn6search2

10. **FP4 并不是一个独立 4-bit 数字就够用了。** 实用 FP4 是 `FP4 payload + small block scale + sometimes second-level scale`。citeturn2view0turn15search25

11. **MXFP4 与 NVFP4 都是 4-bit，却不是同一种 numerical system。** MXFP4 是 32-value/E8M0 block；NVFP4 是 16-value/E4M3 micro-scale + tensor-level second scale。citeturn3view0turn0search2

12. **模型文件更小 ≠ 推理更快。** 真实速度取决于 low-bit payload 是否直达 native MMA、dequant 是否 fused，以及 shape 是否匹配优化 kernel。citeturn12search2turn19search12

13. **decode 与 prefill 对 precision 的收益机制不同。** decode 往往首先受 weight/KV bandwidth 限制；large-batch prefill 更能利用 FP8/FP4 的高 Tensor Core FLOPS。

14. **KV Cache 应作为独立 precision dimension。** 它是动态生成、持续增长、反复读取的数据结构，不能把 weight quantization 的策略直接照搬。citeturn20search0turn20search9

15. **MoE router 是高敏感控制流。** expert selection 跨过 top-k 边界后是离散变化，因此 expert weights 可以极低精度，而 router 往往不值得同样激进地量化。

16. **“FP8 模型”只是营销级简称。** 真正的 production specification 应写成 `W dtype + A dtype + scale granularity + accumulator + output + KV + sensitive-op exceptions`。

17. **硬件世代决定量化策略。** A100、H100、B200 上同一个 FP8/FP4 checkpoint 可能走完全不同的 kernel，因此 precision selection 必须从 accelerator capabilities 开始。citeturn15search8turn15search5turn15search25

18. **未来主线不是“bit 一直变少”，而是 heterogeneous precision。** MXFP6、W4A6、FP4+FP8 mixed paths 已经说明下一阶段更可能是每种 tensor、每个 block 按统计性质花不同 bit budget。citeturn16search9turn14search10

19. **数值评估不能停在 kernel MSE。** autoregressive decoding 中极小 logit perturbation 只要翻转 top-1，就会使后续上下文与 KV trajectory 完全分叉。

20. **最终目标从来不是“bit 最少”，而是 accuracy SLO 下的最低 cost/token。**

### 推荐进一步阅读：论文、标准、代码和官方资料

**数值表示与 mixed precision**

- NVIDIA《Floating Point and IEEE 754》与 CUDA Programming Guide：理解 IEEE arithmetic、rounding、compute mode 的工程入口。citeturn4search4turn4search0
- Micikevicius et al., **Mixed Precision Training**：FP16 + FP32 master weights + loss scaling 的原点。citeturn20academia19
- NVIDIA Ampere Architecture / Tuning Guide：TF32、BF16 和 Tensor Core compute semantics。citeturn15search1turn14search0

**FP8**

- NVIDIA Transformer Engine FP8 recipes：理解 E4M3/E5M2、current/delayed scaling、amax history。citeturn0search11turn0search2
- Transformer Engine block scaling 文档：理解 blockwise FP8 与 MXFP8。citeturn0search5turn0search20
- DeepSeek-V3 Technical Report：研究真正超大模型 fine-grained FP8 training 的代表案例。citeturn18search0

**INT quantization**

- **LLM.int8()**：activation outlier mixed-precision decomposition。citeturn6search0
- **SmoothQuant**：activation-to-weight difficulty migration。citeturn6search3
- **GPTQ**：second-order weight-only PTQ。citeturn6search5
- **AWQ**：activation-aware salient-weight protection。citeturn6search2
- **ZeroQuant / ZeroQuant-V2**：fine-grained W+A quantization、backend 与 compensation。citeturn7search23turn7search3
- **SpQR、QuIP、QuIP#**：理解从 4 bit 向 3/2 bit 后为什么问题逐渐变成 outlier isolation、rotation 与 codebook。citeturn9search0turn7search1turn7search9
- **OmniQuant、AutoRound**：理解 calibration-time learnable scaling/clipping/rounding。citeturn5search1turn8search3

**FP4 与 Microscaling**

- OCP **Microscaling Formats v1.0**：这是理解 MXFP8/MXFP6/MXFP4、E8M0、FP4 E2M1 和 block semantics 最重要的标准文档。citeturn2view0turn3view0turn3view1
- **Microscaling Data Formats for Deep Learning**：理解 MX 设计动机。citeturn1search2
- NVIDIA Transformer Engine **NVFP4BlockScaling**：理解 two-level scaling 与 NVFP4 training recipe。citeturn0search2
- NVIDIA Blackwell NVFP4 文档与 CUTLASS block-scaled GEMM：理解硬件如何真正消费 FP4 payload + scale。citeturn15search25turn0search4
- **Pretraining Large Language Models with NVFP4**：研究 4-bit training 的可行边界及与 FP8 baseline 的关系。citeturn5search3

**Kernel 与 serving**

- cuBLAS / cuBLASLt compute type 文档：建立“input dtype ≠ accumulator dtype ≠ output dtype”的 API 级认识。citeturn14search12
- CUTLASS 4.x：直接研究 FP8/NVFP4/MXFP4 mainloop、block scales、epilogue 与 Rubin/Blackwell MMA。citeturn14search4turn14search10
- vLLM Quantization / Quantized KV Cache 文档：研究 checkpoint format 到 serving kernel 的映射。citeturn10search0turn20search0
- SGLang Quantization 与 Expert Parallel backend：研究 MoE + FP8/FP4 grouped GEMM 的生产路径。citeturn11search2turn11search4
- TensorRT-LLM Quantization Guide：研究 FP8、NVFP4、MXFP4、quantized KV cache 与 ModelOpt 的 NVIDIA production stack。citeturn10search1

最终可以把整套知识体系压缩成下面这个公式：

\[
\boxed{
\begin{aligned}
\text{Useful Precision}
&=
f(
\text{representation},
\text{range},
\text{ULP},
\text{scaling},
\text{granularity},
\text{outliers},
\\
&\quad
\text{operator sensitivity},
\text{accumulation},
\text{hardware MMA},
\text{kernel fusion},
\text{memory traffic},
\\
&\quad
\text{engine support},
\text{model accuracy},
\text{serving workload}
)
\end{aligned}
}
\]

而工程终点不是：

\[
\boxed{\min(\mathrm{bits})}
\]

而是：

\[
\boxed{
\min\left(\frac{\$}{token}\right)
\quad
\text{s.t.}
\quad
\text{accuracy}\ge A_{\min},
\;
TTFT\le L_1,
\;
TPOT\le L_2
}
\]

一旦把 precision 理解成这个联合优化问题，即使下一代出现 **FP3、MXFP3、新 shared-scale codebook、adaptive block format 或新的 Tensor Core datatype**，判断方法也不会变化：先看 **表示与 scale**，再看 **operator 与 accumulator**，再看 **native MMA 与 kernel data path**，最后只相信 **accuracy × latency × throughput × memory × energy × cost/token 的端到端实测**。OCP MX、Blackwell FP4、MI355X MXFP4/MXFP6 与 Rubin 初始 FP8/FP4 支持，已经清楚地展示了这种从“dtype”走向“representation + scaling + hardware/software co-design”的方向。citeturn2view0turn15search25turn16search1turn14search10
