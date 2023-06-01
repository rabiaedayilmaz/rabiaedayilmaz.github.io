# Notes About Transformers United 2023 by Stanford University

## Introduction to Transformers by Andrej Karpathy

### Lecture
Attention Timeline: 

* 1990s : Prehistoric Era (RNNs, LSTMs) 
* 2014 : Simple Attention Mechanism
* 2017 : Beginning of Transformer (Attention is all you need)
* 2018 : Explosion of Transformers in NLP (BERT, GPT-3)
* 2018 - 2020 : Explosion into other fields (ViTs, Alphafold-2)
* 2021 : Start of Generative Era (Codex, Decision Transformer, GPT-x, DALL-E)
* 2022 : Present (ChatGPT, Whisper, Robotics Transformer, Stable Diffusion)

`````{admonition} Past, Present, and Future
:class: tip
Prehistoric Era -> Seq2Seq, LSTMs, GRUs } RNNs; good: encoding history; bad: long sequences, context.

Present -> unique apps: audio, art, music, storytelling; reasoning capabilities: common sense, logical, mathemetical; human alignment and interaction: reinforcement learning and human feedback; controlling toxicity, bias, and ethics.

Future -> video understanding and generation, GPT authors, generalist agents, domain specific foundation models: DoctorGPT, LawyerGPT.

Missing Ingredients -> external memory like Neural Turing Machines, finance and business; reducing computational complexity; enhanced human controllability; alignment with language models of human brain.
`````

Attention = "communication phase"
* soft, data-dependent message passing on directed graphs
* each node stores a vector
* there is a communication phase (attention) and then a compute phase (MLP)

* key -> what do i have?
* value -> what do i publicly reveal/broadcast to others?
* query -> what am i looking for?
---

`````{admonition} Attention!
:class: tip

in *communication phase* of the transformer: i) every head applies this, in parallel and ii) then every layer, in series

in encoder-decoder models: i) encoder is fully-connected cluster and ii) decoder is fully-connected to encoder positions, and left-to-right connected in decoder positions.
`````

* multi-head self-attention - heads copy&paste in parallel
* self-attention - layers copy&paste in serial
* cross-attention - queries are from the same node but keys and value are from external source, like encoder.
---
* Transformer -> flexible: chop everything up into pieces, add them into the mix, self-attend over everthing. it frees neural net computation from the burden of Euclidean space.

* Language Models a re Few-Shot Learners => Transformers are capable of in-context learning or meta learning.
* if previous NNs are special-purpose computers designed for a specific task, GPT is a general-purpose computer reconfigurable at run time to run natural language programs. Programs are given in prompts (a kind of inception). GPT runs the program by completing the doc. - Andrej Karpathy

### Recommended Readings

* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
---
## Language and Human Alignment by Jan Leike (OpenAI)

### Lecture

* Team AI -> incredibly strong players
* Team Human -> can pick which AI players join

Objectives of Team Human (TH): i) recruit AI players to TH (**Alignment**), and ii) write rules so TH doesn't lose (**Governance**).

How do we build AI systems that follow human intent? 1. Explicit intent (follow instructions, be an assistant), 2. Implicit intent (do what i mean, dont make stuff up, dont be mean etc)

Main technique: RLHF. 1. Train a reward model from comparisons, 2. Fine-tune a pretrained model with RL.

Proximal Policit Optimization (PPO) > Shrink and Finetune (SFT) > GPT

ChatGPT -> dialog is the universal interface, better at refusing harmful tasks, still halucinates, sensitive prompting, silly mistakes, free for now.

Training Cost: GPT-3 > InstructGPT RL > InstructGPT SFT.

Evaluation is easier than generation.
Scaling human supervision (main RLHF problem) -> what humans can evaluate is limited to a specific task difficulty. so you can not evaluate properly anymore.

### Recommended Readings

* [ChatGPT](https://openai.com/blog/chatgpt/)
* [InstructGPT](https://openai.com/blog/instruction-following/)
* [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

## Emergent Abilities and Scaling in LLMs by Jason Wei (Google Brain)

**Emergence:** a qualitative change that arises from quantitavie changes

### Lecture

* If you scale up the size of language model, measured either in compute, in dataset size or number of parameters, there is a sort of this predictable improvement in the test loss.


`````{admonition} Analogy
:class: tip

With a bit of uranium, nothing special happens; with a large amount of uranium packed densely enough, you get a nuclear reaction.

Given only small molecules such as calcium, you can't meaningfully encode useful information; gives larger molecules such as DNA, you can encode a genome.

An ability is emergent if it is not present in smaller models, but is present in larger models.
`````

Measuring the *size* of a model: training FLOPs, number of model parameters, training dataset size.

* A few-shot prompted task is emergent if it achieves random accuracy for small models and above-random accuracy for large models.

`````{admonition} Funny Quote from Video
:class: tip

I don't know, but do you guys remember in chemistry class, when you'd have moles. And it would be like 10 to 23, and then teacher would be like, oh, don't even think about how big this number is. That is like the number of floating point operations that goes into the pre-training of some of these models.
`````

* A prompting technique is emergent if it hurts performance (compared to baseline) for small models, and improves baseline for large models. => Chain-of-Thought prompting as an emergent prompting technique.

* Large models benefit from RLHF, but for smaller models hurt the performance :(

* Emergence = measure of model **scale**

* Emergent abilities can only be observed in large models
    * Their emergence can not be predicted by scaling plots with small models only
* Reflection
    * forming for viewing these abilities, which are not intentionally built in "why we should keep scaling; these abilities are hard to find otherwise"
    * tension between emergence (task general; bigger models) and many production tasks (task specific; compute constraints; in-domain data)     
    * haven't seen a lot of work on predicting future emergence 
        * why? too hard, only task-specific answers?

Chain of Thoughts, Self-consistency: majority vote } large models
* LMs acquire emergent abilities as they scaled up
* The ability for LM to do multi-step reasoning emerges with scale, unlocking tasks (CoT etc.)
* LM will continue to get bigger and better

* Looking forward: scaling, better prompting & characterization of LM abilities, applied work (therapy, creative writing, science), benchmark, compute efficient methods for better LMs.

### Recommended Readings

* [ChatGPT](https://openai.com/blog/chatgpt/)
* [InstructGPT](https://openai.com/blog/instruction-following/)
* [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

## Strategic Games by Noam Brown (FAIR)

### Lecture

* Scaling + training cost can make huge difference (Poker AI runs on 28 CPU cores)
* Monte Carlo Search Tree -> useful for deterministic perfect information games

`````{admonition} The Bitter Lesson by Richard Suttor
:class: tip

The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective... The two methods that seem to scale or bitrarily in this way are **search** and **learning**.
`````

* Generality: general way of scaling inference compute? not MCTS, not Counterfactual Regret Minimization
* Much higher test-time compute are willing to pay for a proof of Riemann Hypothesis? Or a new life-saving drugs?

`````{admonition} Multi-Agent Perspective
:class: tip

* in purely competitive games (chess, go, poker etc.) self-play is guaranteed to converge to an optimal solution
* real-world involves a mix of **cooperation** and competititon, where success requires understanding human behavior and conventions.
* language is the ultimate human convention at the heart of cooperation
* current language modeling approaches are still just **imitating** human-like text
* people use language as a tool for **coordinating** with other people

* grounded and intentional dialogue } goal
`````

Main Contributions:
* **controllable dialogue models** that condition on the game state and a set of **intended actions** for the speaker and recipient.
* a **planning engine** that **accounts for dialogue** and human being while playing better than humans
* **self-play reinforcement learning algorithms** that model human behavior and dialogue and learn to respond effectively to them
* an ensemble of **message filtering techniques** that filter both nonsensical and strategically unsound messages

Algorithm: policy netwrok + dialogue model
**piKL-Hedge:** a regret minimize for an objective that combines expected value and KL divergence from a human imitation policy.
V_piKL(pi) = V(pi) - lambda * KL(pi || pi_human)

Cicero Limitations and Future: 
* intent representation is just an **action** per player. how do we condition on and plan over more complex things like explaining its actions, asking questions, high-level strategies etc. ?
* limited understanding of long-term effects of dialogue because doesn't condition on dialogues
* more general way of scaling inference-time compute to achieve better performance
* diplomacy is an amazing testbed for multi-agent AI and grounded dialogue
* dialogue + action data available through RFP

### Recommended Readings

* [Human-level play in the game of Diplomacy by combining language models with strategic reasoning](https://www.science.org/doi/full/10.1126/science.ade9097)
* [Modeling Strong and Human-Like Gameplay with KL-Regularized Search](https://proceedings.mlr.press/v162/jacob22a.html)
* [No-Press Diplomacy from Scratch](https://proceedings.neurips.cc/paper/2021/hash/95f2b84de5660ddf45c8a34933a2e66f-Abstract.html)

## Robotics and Imitation Learning by Ted Xiao (Google Brain)

### Lecture

* foundation models enable **emergent capabilities** (emergence of more complex behavior not present in smaller models) and **homogenization** (generalization to combinatorially many downstreams use cases) 

`````{admonition} Ingredients for a Robotic Foundation Model
:class: tip

#1. Design Principles of ML Scaling
* high-capacity architectures, i.e. self-attention
* scaling params & compute & corpus size (tokens)
* dataset size matters more than quality

#2. Proliferation of Internet-Scale Models
* generative models in language, coding, vision, audio... experience emergent capabilities
* proliferation + accelaration mean these models will get better "on their own" overtime

#3. Robotics moves from Online to Offline
`````

* **Scaling**: High-capacity architectures (attention) and data interpretability (tokenization)
* **internet-scale models**: leverage foundation models, provide common sense, use language
* **offline robot learning**: collect tons of diverse interesting data, dont care about how the data is collected

**Recipe**: combine learge diverse offline dataset with high-capacity architectures by using language as a universal glue

* RT-1 Model, SayCan, InnerMonologue, DIAL (VLM-visual language model)

* the bottleneck for robotics was high level semantic planning -> LLMs yeee

### Recommended Readings

* [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
* [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)
* [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608)

## Common Sense Reasoning by Yejin Choi (U. Washington / Allen Institute for AI)

### Lecture

* smaller but better + knowledge is better

* **Maieutic Prompting**: logically consistent reasing with recursive explanations. like Socrates. weighted max-SAT solver. better than fine-tuned T5. dramatically enhance computational reasoning.
* **Symbolic Knowledge Distillation**: from general LMs to Causal Commonsense models. systematic generalization problem=solving a dataset without solving the underlying task. commonsense=pratical knowledge+reasoning. LMs != knowledge models.
* **Commonsense Morality**: machine ethics. **value pluralism**.

humans are better at understanding rather than generation, unlike GPT-3.

### Recommended Readings

* [Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations](https://arxiv.org/abs/2205.11822)
* [Symbolic Knowledge Distillation: from General Language Models to Commonsense Models](https://arxiv.org/abs/2110.07178)
* [Can Machines Learn Morality? The Delphi Experiment](https://arxiv.org/abs/2110.07574)

## Biomedical Transformers by Vivek Natarajan (Google Health AI)

### Lecture

Why transformers in biomedicine?: clinical notes, electronic medical records, proteins

* can effectively handle multimodal biomedical data
* can model complex, long-range interactions over sequences
* can easily be scaled to large biomedical datasets

What is missing? => benchmarks (no big-bench for med.) and evaluation framework (comprehensive)

* Medical question answering: comprehension, recall of medical knowledge, reasoning.

* automated metrics -> deeply unsatisfactory -> fail to capture the nuances of the real worl clinical applications

* instruction prompt tuning (prompt params are aligned with medical domain) + FlanPALM

* proteins and genomics

### Recommended Readings

* [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138)
* [ProtNLM: Model-based Natural Language Protein Annotation](https://web.stanford.edu/class/cs25/)
* [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x)

## In-Context Learning & Faithful Reasoning by Stephanie Chan (DeepMind) & Antonia Creswell (DeepMind)

## Neuroscience-Inspired Artificial Intelligence by Trenton Bricken (Harvard/Redwood Center for Theoretical Neuroscience/Anthropic) & Will Dorrell (UCL Gatsby Computational Neuroscience Unit/Stanford)
