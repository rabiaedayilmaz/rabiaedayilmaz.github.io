# COMMUNITY PAPER READING SESSIONS - COHERE AI

## Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks

::::{grid}
:gutter: 1

:::{grid-item-card} Paper

[Read](https://arxiv.org/abs/2211.12588)
:::

Chain of Thoughts Prompting = CoT and Program of Thougts = PoT. CoT models are language models that are used to produce text describing reasoning and computation, so that answering a question. On the other hand, PoT is used to generate text and programming language, so that an answer to a question. PoT is evaluated on 5 math word and financial-QA dataset problems in both few-shot and zero-shot setting. Concluded that PoT has 12% performance gain overall. [Github](https://github.com/wenhuchen/Program-of-Thoughts) 

## Big Self-Supervised Models are Strong Semi-Supervised Learners

::::{grid}
:gutter: 1

:::{grid-item-card} Paper

[Read](https://arxiv.org/pdf/2006.10029.pdf)
:::

One of the used techniques for learning from few labeled examples while making best use of a large amount of unlabeled data is **unsupervised pretraining** *followed by supervised finetuning*. This paradigm uses unlabeled data in a task-agnostic way. In contrast to common approaches of semi-supervised learning for computer vision, they showed that it is surprisingly effective for semi-supervised learning on ImageNet. Key ingredient is the use of big (deep and wide) networks during pretraining and fine-tuning. Semi-supervised learning algorithm can be summarized in three steps: unsupervised pretraining of a big ResNet model using SimCLRv2, supervised fine-tuning on a few labeled examples, and distillation with unlabeled examples for refining and transferring the task-specific knowledge. In medical applications where acquiring high-quality labels requires careful annotation by clinicians, better semi-supervised learning approaches can potentially help save lives.