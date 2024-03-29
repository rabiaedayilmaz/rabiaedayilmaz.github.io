# Transformers United 2023 Notes

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

`````{admonition} Past, Present, and Future
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

## Emergent Abilities and Scaling in LLMs by Jason Wei (Google Brain)

## Strategic Games by Noam Brown (FAIR)

## Robotics and Imitation Learning by Ted Xiao (Google Brain)

## Common Sense Reasoning by Yejin Choi (U. Washington / Allen Institute for AI)

## Biomedical Transformers by Vivek Natarajan (Google Health AI)

## In-Context Learning & Faithful Reasoning by Stephanie Chan (DeepMind) & Antonia Creswell (DeepMind)

## Neuroscience-Inspired Artificial Intelligence by Trenton Bricken (Harvard/Redwood Center for Theoretical Neuroscience/Anthropic) & Will Dorrell (UCL Gatsby Computational Neuroscience Unit/Stanford)
