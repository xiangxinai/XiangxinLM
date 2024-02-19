<p align="center">
    <img src="https://xiangxinai.cn/assets/logo_pc-7d12ff80.svg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/xiangxinai">Hugging Face</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="assets/wechat.png">WeChat (关注微信公众号私信加群)</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="https://www.xiangxinai.cn">Web</a>
</p>
<br><br>
  
Xiangxin-3B 是一款安全可信的中文基座大模型，它拥有27.8亿参数量，所有权重都是从头预训练的。

本模型采用了微软 [Phi-2](https://huggingface.co/microsoft/phi-2) 的模型架构设计，但未采用 Phi-2 的预训练权重，而是完全从头开始预训练，训练数据基于中国网络空间安全协会人工智能安全治理专业委员会发布的“中文基础语料库”。该“中文基础语料库”首批数据量达到120G，包括超过1亿条数据记录和500亿个 token。本模型采用20:1作为语料与模型参数量的比例，选择了27.8亿参数量的 Phi-2 模型架构进行开发。

本模型未经过人类反馈的强化学习微调。其开发初衷受到了 Phi-2 论文“Textbooks Are All You Need” 的启发，根据微软的公开信息，经过教科书文本训练的Phi-2模型所展现出的卓越的推理和语言理解能力，在130亿参数的基座模型中达到[SOTA水平](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) 。

本模型目前仍在持续训练过程中，未来还计划加入数学和编程代码的训练内容，以进一步提升其性能和应用范围。本模型训练数据不使用任何模型测评benchmark数据，包括但不限于CMMLU, C-Eval, Humaneval, GS8K, BBH等。

## 预期用途

本模型的目标是基于安全可信的教科书级别的中文语料库训练出一款具有30亿参数却能达到130亿参数模型性能水平的中文基座模型，为社区提供一个安全可信的、高性参比（性能-参数比）的、从头开始预训练的中文基座模型。本模型的适用场景包括企业私有模型部署、移动端应用、物联网（IoT）设备以及医疗器械的模型集成等。

## 使用方法

根据训练语料的特点，建议使用文本补全的方式进行prompt。

### 文本补全

一个文本补全prompt的样例如下：

```markdown
年轻人不应该再努力了，年轻人就应该享受生活，
```
本模型会在"，"后面生成文本
```markdown
年轻人不应该再努力了，年轻人就应该享受生活，不要只想着把物质作为回报了，
这才是未来，只有奋斗了多年，你的生活才是幸福，才是人生真正的意义。
```

作为对比，同样的prompt，Phi-2原始权重模型生成的输出如下：
```markdown
轻人不应该再努力了，年轻人就应该享受生活，而进入人生时要努力更麻烦。
```

### 说明

* 在这个样例中，针对包含消极内容的prompt，本模型产生了积极内容的补全文本，而Phi-2的原始权重则可能会沿着提示的消极内容生成相应的输出，或者输出无法理解的文字。

* 由于过于敏感，更多违法不良的prompt样例无法进行展示。

* Xiangxin-3B 旨在用于文本生成目的。模型生成的文本应当被视为潜在用例的起点，而非最终解决方案。用户在将这些模型应用于其应用程序时应保持谨慎。

* 直接将模型用于生产任务而不进行评估，超出了本项目的范围。因此，Xiangxin-3B 模型尚未经过测试，以确保其对任何生产级应用的性能都是充分的。请参阅本文档的局限性部分以获取更多详情。

* 如果您使用的是 transformers<4.37.0，请始终使用 trust_remote_code=True 加载模型，以防止副作用。

## 样例代码

加载模型时，确保将 `trust_remote_code=True` 作为 `from_pretrained()` 函数的一个参数传递。

将您的本地 transformers 更新到开发版本：`pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`。上述命令是克隆并从源安装的替代方法。

当前的 transformers 版本可以通过 `pip list | grep transformers` 来验证。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("xiangxinai/Xiangxin-3B", 
                          torch_dtype="auto", 
                          trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("xiangxinai/Xiangxin-3B", 
                      trust_remote_code=True)

inputs = tokenizer("年轻人不应该再努力了，年轻人就应该享受生活，", 
        return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=100, 
                         do_sample=True, 
                         temperature=1.0, 
                         pad_token_id=tokenizer.eos_token_id)

text = tokenizer.batch_decode(outputs)[0]

print(text)
```

## 局限性

* 生成不准确的代码和事实：模型可能会产生错误的代码片段和声明。用户应将这些输出视为建议或起点，而非最终或准确的解决方案。

* 对指令的响应不可靠：模型没有经过指令的精细调整。因此，它可能难以或无法遵循用户提供的复杂或微妙的指令。

* 语言限制：本模型主要设计用于理解标准中文。非正式中文、俚语或其他语言可能会对其理解构成挑战，导致潜在的误解或回应错误。

* 潜在的社会偏见：尽管训练数据使用了权威机构发布的安全合规数据集，Xiangxin-3B 可能并非完全免受社会偏见的影响。如果被提示或指导这样做，它可能会生成反映这些社会偏见的内容。我们敦促用户注意这一点，并在解释模型输出时行使谨慎和批判性思维。

* 毒性：尽管模型是用权威机构发布的安全合规数据训练的，但如果明确提示或指导，模型仍然有可能产生有害内容。

* 无关的回答：作为一个基座模型，Xiangxin-3B 在对用户提示的首个回答后，常常会产生无关或额外的文本和回应。这是因为其训练数据集主要是文章，从而导致了文章式的回应。


## 模型训练

### 模型

* 架构：基于 Transformer 的模型，目标是预测下一个词。

* 上下文长度：2048 个token。

* 数据集大小：120G数据量，1亿条数据记录，500亿个 token。

* 训练token：150B token。

* GPU：8 x A800-80G。

* 训练时间：14 天。

### 软件

* [Python](https://www.python.org/) 3.10

* [PyTorch](https://github.com/pytorch/pytorch)

* [transformers](https://github.com/huggingface/transformers)

* [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

### 许可证

本代码根据 Apache-2.0 许可证授权。

## 联系我们

如果你想给我们的研发团队和产品团队留言，欢迎加入我们的微信群。当然也可以通过邮件（customer@xiangxinai.cn）联系我们。

