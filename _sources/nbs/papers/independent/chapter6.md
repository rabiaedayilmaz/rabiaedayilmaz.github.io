# Neuroscience

## Learnable latent embeddings for joint behavioural and neural analysis

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.nature.com/articles/s41586-023-06031-6#Abs1)
:::

Mapping behavioural actions to neural activity is a fundamental goal of neuroscience. There is a gap of lacking nonlinear techniques that can explicitly and flexibly leverage **joint behaviour** and **neural data** to uncover **neural dynamics**. Here, presented a new encoding method, CEBRA, that jointly uses **behavioural and neural data** in a (supervised) hypothesis- or (self-supervised) discovery-driven manner to produce both consistent and high-performance latent spaces. The inferred latents can be used for decoding. Effects were shown on both calcium and electrophysiology datasets, across sensory and motor tasks and in simple or complex behaviours across species. Also, single- and multi-session datasets for hypothesis testing or can be used label free. CEBRA can be used for the mapping of space, uncovering complex kinematic features, for the production of consistent latent spaces across two-photon and Neuropixels data, and can provide rapid, high-accuracy decoding of natural videos from visual cortex.

**A central quest in neuroscience is the neural origin of behaviour.** PCA is linear, interpretable and great but cost of performance. UMAP and t-SNE are nonlinear and great but lack exploiting time information (but always available in neural recordings!). 

Nonlinear methods->high-performance decoding but lack of identifiability (critical property->learned representations are uniquely determined->consistency across animals/sessions).

CEBRA -> a new self-supervised learning algorithm for obtaining interpretable, consistent embeddings of high-dimensional recordings using auxiliary variables. Unlike SimCLR, no data augmentation, no generative model. CEBRA can be used to generate embeddings across multiple subjects and cope with distribution shifts among experimental sessions, subjects and recording modalities and it uses uses a new data-sampling scheme to train a neural network encoder with a contrastive optimization objective to shape the embedding space.

CEBRA uses user-defined labels (supervised, hypothesis-driven) or time-only labels (self-supervised, discovery-driven).