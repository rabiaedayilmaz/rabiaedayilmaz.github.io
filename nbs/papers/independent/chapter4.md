# Spiking Neural Networks - SNNs && Forward-Forward Network && Learning Without Backpropagation

## Deep Learning in Spiking Neural Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1804.08150.pdf)
:::

A review paper. Neurons in an ANN are characterized by a single, static, continuous-valued
activation. Yet biological neurons use discrete spikes to compute and transmit information, and the spike times, in addition to the spike rates, matter. Spiking neural networks (SNNs) are thus more biologically realistic than ANNs, and arguably the only viable option if one wants to understand how the brain computes. SNNs are also more hardware friendly and energy-efficient than ANNs, and are thus appealing for technology, especially for portable devices. However, training deep SNNs remains a challenge. Spiking neurons’ transfer function is usually non-differentiable, which prevents using backpropagation.

---

## Spiking Neural Networks and Their Applications: A Review

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9313413/)
:::

**Training of SNNs:** There are two main approaches to train SNNs: (i) training SNNs directly based on either supervised learning with gradient descent or unsupervised learning with STDP (ii) convert a pre-trained ANN to an SNN model. The first approach has the problem of gradient vanishing or explosion because of a non-differentiable spiking signal.

While the majority of existing works on SNNs have focused on the image classification problem. While SNNs have shown an impressive advantage with regard to energy efficiency, their accuracy performances are still low compared to ANNs on large-scale datasets such as ImageNet.

---

## Improved spiking neural networks for EEG classification and epilepsy and seizure detection

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://dl.acm.org/doi/10.5555/1367089.1367090)
:::

A complicated pattern recognition problem: epilepsy and epileptic seizure detection. Three training algorithms are investigated: SpikeProp (using both incremental and batch processing), QuickProp, and RProp. The result is a remarkable increase in computational efficiency.For the XOR problem, the computational efficiency of SpikeProp, QuickProp, and RProp is increased by a factor of 588, 82, and 75, respectively. EEGs from three different subject groups are analyzed. RProp is the best training algorithm because it has the highest classification accuracy among all training algorithms specially for large size training datasets with about the same computational efficiency provided by SpikeProp. classification accuracy of 92.5%.

---

## Training Deep Spiking Neural Networks Using Backpropagation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2016.00508/full)
:::

In this paper, we introduce a novel technique, which treats the membrane potentials of spiking neurons as differentiable signals, where discontinuities at spike times are considered as noise. This enables an error backpropagation mechanism for deep SNNs that follows the same principles as in conventional deep networks, but works directly on spike signals and membrane potentials. In the N-MNIST example, equivalent accuracy is achieved with about five times fewer computational operations.

---

## Gradients without Backpropagation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2202.08587.pdf)
:::

Backpropagation, or reverse-mode differentiation, is a special case within the general family of automatic differentiation algorithms that also includes the forward mode. We present a method to compute gradients based solely on the directional derivative that one can compute exactly and efficiently via the forward mode.They called it **forward gradient** by entirely eliminating the need for backpropagation in gradient descent. 

SOURCE CODE WILL BE RELEASED.

---

## Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.semanticscholar.org/paper/Error-driven-Input-Modulation%3A-Solving-the-Credit-a-Dellaferrera-Kreiman/7e9020792cbdb0731fb623735127e202eff198db)
:::

BP lacks biological plausibility in many regards, including the weight symmetry problem, the dependence of learning on non-local signals, the freezing of neural activity during error propagation, and the update locking problem. They proposed to replace the backward pass with a second forward pass in which the input signal is modulated based on the error of the network. We show that this novel learning rule comprehensively addresses all the above-mentioned issues and can be applied to both fully connected and convolutional models.

---

## The Forward-Forward Algorithm: Some Preliminary Investigations

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
:::

The Forward-Forward algorithm replaces the forward and
backward passes of backpropagation by two forward passes, one with positive
(i.e. real) data and the other with negative data which could be generated by the
network itself. Each layer has its own objective function which is simply to have
high goodness for positive data and low goodness for negative data. The sum of the
squared activities in a layer can be used as the goodness but there are many other
possibilities, including minus the sum of the squared activities. If the positive and
negative passes can be separated in time, the negative passes can be done offline,
which makes the learning much simpler in the positive pass and allows video to
be pipelined through the network without ever storing activities or stopping to
propagate derivatives.

The Forward-Forward algorithm (FF) is comparable
in speed to backpropagation but has the advantage that it can be used when the precise details of
the forward computation are unknown. It also has the advantage that it can learn while pipelining
sequential data through a neural network without ever storing the neural activities or stopping to
propagate error derivatives. Somewhat slower than backpropagation and does not generalize quite as well on several of the toy problems. The two areas in which the forward-forward algorithm may be superior to backpropagation are as a model of learning in cortex and as a way of making use of very low-power analog hardware without resorting to reinforcement learning. Inspired by Boltzmann machines ( unsupervised contrastive learning ) and Noise Contrastive Estimation. Contrastive learning is the core idea.

The Boltzmann machine can be seen as a combination of two ideas:
1. Learn by minimizing the free energy on real data and maximizing the free energy on negative data generated by the network itself.
2. Use the Hopfield energy as the energy function and use repeated stochastic updates to sample global configurations from the Boltzmann distribution defined by the energy function.

FF can be viewed as a special case of a GAN in which every hidden layer of the discriminative
network makes its own greedy decision about whether the input is positive or negative so there is no
need to backpropagate to learn the discriminative model. No need to backpropagate to
learn the generative model, it just reuses the representations learned by the discriminative model.

---

## How important is weight symmetry in backpropagation?

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://dl.acm.org/doi/10.5555/3016100.3016156)
:::

Gradient backpropagation (BP) requires symmetric feedforward and feedback connections—the same weights must be used for forward and backward passes. This "weight transport problem" (Grossberg 1987) is thought to be one of the main reasons to doubt BP's biologically plausibility.

---

## Signal Propagation: A Framework for Learning and Inference In a Forward Pass

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2204.01723)
:::

Author has a [blog article](https://amassivek.github.io/sigprop):

Two constraints of backpropagation on the training network are: (1) the addition of feedback weights that are symmetric with the feedforward weights; and (2) the requirement of having these feedback weights for every neuron.

Spiking neural networks are similar to biological neural networks and used for neuromorphic chips.There are two problems for learning in spiking neural networks. First, the learning constraints under backpropagation are difficult to reconcile with learning in the brain, and hinders efficient implementations of learning algorithms on hardware. Second, training spiking networks results in the dead neuron problem (response is always nothing).

Works on Forward Learning: Error Forward Propagation (2018) and Forward Forward (2022).

**Spatial Credit Assignment** How does the learning signal reach every neuron?
Broadly, there are two approaches to the learning phase. The first approach computes a global learning signal (left middle figure) and then sends this learning signal to every neuron. The second approach computes a local learning signal (right figure) at each neuron (or layer). The first approach has the problem of having to coordinate sending this signal to every neuron in a precise way. This is costly in time, memory, and compatibility. The second approach does not encounter this problem, but has worse performance.

**Global signal:** feedforward network, gradient backpropagation, random feedback.
**Local signal:** perturbation learning, local losses.

**Temporal Credit Assignment** How does the global learning signal reach multiple connected inputs (aka every time step)?

> There are two popular methods to answer this question: Backpropagation through time, and forward mode differentiation (FMD does step 1 (inference) and step 2 (learning) together (alternating)).

Paper:

They proposed a new learning framework, signal propagation (sigprop), for propagating a learning signal and updating neural network parameters via a forward pass, as an alternative to backpropagation. In sigprop, there is only the forward path for inference and learning. So, there are no structural or computational constraints necessary for learning to take place, beyond the inference model itself, such as feedback connectivity, weight transport, or a backward pass, which exist under backpropagation based approaches. That is, sigprop enables global supervised learning with only a forward path. This is ideal for parallel training of layers or modules. In biology, this explains how neurons without feedback connections can still receive a global learning signal. They used sigprop to train continuous time neural networks with Hebbian updates, and train spiking neural networks with only the voltage or with biologically and hardware compatible surrogate functions.

BP is computationally inefficient for memory and time, and bottleneck parallel learning. Calculates the
contribution of each neuron to the network’s output error. Also, BP is not similar how the brain does not have comprehensive feedback connectivity, neither neural feedback, and feedback and feedforward connectivity would need to have weight symmetry.

Sigprop has the following desirable features:

First, inputs and learning signals use the same forward path, so there are no additional structural or computational requirements for learning, such as feedback connectivity, weight transport, or a backward pass.

Second, without a backward pass, the network parameters are updated as soon as they are reached by a forward pass containing the learning signal. Sigprop does not block the next input or store activations. So, sigprop is ideal for parallel training of layers or modules. 

Third, since the same forwardpass used for inputs is used for updating parameters, there is only one type of computation. And performs global learning signal.

**Feedback Alignment:** uses fixed random weights to transport error gradient information back to hidden layers, instead of using symmetric weights.

**Local Learning:** layers are trained independently by calculating a separate loss for each layer using an auxiliary classifier per layer.

**Target Propagation:** generates a target activation for each layer instead of gradients by propagating backward through the network. It requires reciprocal connectivity and is forwardpass and backwardpass locked. In contrast, sigprop generates a target activation at each layer by going forward through the network.

**Equilibrium Propagation:** is an energy based model using a local contrastive Hebbian learning with the same computation in the inference and learning phases. It is a continuous recurrent neural network that minimizes the difference between two fixed points: when receiving an input only and when receiving the target for error correction.

**Error Forward Propagation:** is for closed loop control systems or autoencoders. In either case, the output of the network is in the same space as the input of the network. These works calculate an error between the output and input of the network and then propagate the error forward through the network, instead of backward, calculating the gradient as in error backpropagation. Error forward propagation is backwardpass locked and forwardpass locked. It also requires different types of computation for learning and inference.

---

## TRAINING SPIKING NEURAL NETWORKS USING LESSONS FROM DEEP LEARNING

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2109.12894.pdf)
:::

If our brains dissipated as much heat as state-of-the-art deep learning models, then natural selection would have wiped humanity out long before we could have invented machine learning.

There are several persistent themes across these theories, which can be distilled down to ‘the three S’s’: spikes, sparsity, and static suppression (event driven processing).

Much like the artificial neuron model, spiking neurons operate on a weighted sum of inputs. Rather than passing the result through a sigmoid or ReLU nonlinearity, the weighted sum contributes to the membrane potential U(t) of the neuron.

Input encoding (rate coding, latency, delta modulation) + output decoding (rating coding, latency, population coding).

Objective functions: cross-entropy loss (cross-entropy spike rate/spike count, maximum membrane/membrane potential) and mean square error (mean square spike loss/spike count, mean square membrane/membrane potential).

SNN training methods: shadow training, backpropagation using spikes, local learning rules.

Where inference efficiency is more important than training efficiency, and if input data is not time-varying, then shadow training could be the optimal way to go.

Hybrid approach: 
* Taking the gradient only at spike times, unbiased error but not training dead neuron.
* Surrogate gradient descent, biased error, training dead neurons.

Long term temporal dependencies: adaptive thresholds, axonal delays, membrane dynamics, multistable neural activity.

---

## Memristor-Based Binarized Spiking Neural Networks: Challenges and applications

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/9693512/)
:::

Nature has engineered the most efficient computational processor, and yet the blueprint of the brain remains a mystery. How does the brain achieve within 20 W what it takes data centers hundreds of thousands of watts to process? Emerging memory technologies, such as memristors/ resistive random-access memory (RAM), are reducing the gap between the physical and algorithmic layers of computing from a bottom-up approach, while SNNs draw inspiration from the brain’s spikebased computational paradigm, providing top-down integration opportunities.

Representing information as digital spiking events can improve noise margins and tolerance to device variability.

Restricting neuron activations to single-bit spikes also alleviates the significant analog-todigital converter overhead that mixed-signal approaches have struggled to overcome.

Power Consumption Order (descending): SRAM, Arithmetic Units, ADC, Routers, Crossbar.

Even if activations and weights are bounded in precision, time can be thought of as continuous and provides an alternative dimension to encode information in.

In general, large-scale networks appear to be far more tolerant of quantization errors than constrained models are.

The leaky integrate-and-fire (LIF) neuron model is commonly used in conjunction with large-scale network models. Computationally inexpensive and easily trainable with the BPTT algorithm. 

There is a justified concern that advances in hardware will struggle to keep up, and the benefits derived from modern deep learning may soon saturate.

A floating-point version of the weights is stored, and a quantized version of the weights is used during the forward pass to calculate the loss at the output. During error backpropagation, the gradient is computed with respect to each layer’s activations and binary weights. The gradients are used to update the highprecision weights during the update step. This process is repeated for all time steps.

Network structure of this work: conv5 > avg. pool2 > conv5 > avg. pool2 > fully connected.

---

## Attention Spiking Neural Networks - <3

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2209.13929)
:::

We first present our idea of attention in SNNs with a plug-and-play combined module kit, termed the Multi-dimensional Attention (MA) module. Then, a new attention SNN architecture with end-to-end training called ”MA-SNN” is proposed, which infers attention weights along the temporal dimension, channel dimension, as well as spatial dimension separately or simultaneously. Based on the existing neuroscience theories, we exploit the attention weights to optimize membrane potentials, which in turn regulate the spiking response in a data-dependent way. At the cost of negligible additional parameters, MA facilitates vanilla SNNs to achieve sparser spiking activity, better performance, and energy efficiency concurrently.

To our best knowledge, this is for the first time, that the SNN community achieves comparable or even better performance compared with its ANN counterpart in the large-scale dataset. Compared with counterpart Res-ANN-104, the performance gap becomes -0.95/+0.21 percent and has 31.8×/7.4× better energy efficiency.

Design philosophy is clear, exploiting attention to regulate membrane potentials, i.e., focusing on important features and suppressing unnecessary ones, which in turn affects the spiking activity.

---

## Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch - <3

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2021.773954/full)
[Github](https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time)
:::

SNNs convey temporally-varying spike activation through time that is likely to induce a large variation of forward activation and backward gradients, resulting in unstable training.
To address this training issue in SNNs, we revisit Batch Normalization (BN) and propose a temporal Batch Normalization Through Time (BNTT) technique.
Different from previous BN techniques with SNNs, we find that varying the BN parameters at every time-step allows the model to learn the time-varying input distribution better. Specifically, our proposed BNTT decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes.

---

## Neuromorphic deep spiking neural networks for seizure detection

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://iopscience.iop.org/article/10.1088/2634-4386/acbab8/meta)
:::

EEG public datasets.
Vast majority of model are on cloud computing resources. But, edge devices can securely analyze sensitive medical data in a real-time and personalised manner, for seizure detection etc. Novel spiking ConvLSTM unit for a surrogate gradient-based deep spiking neural network (SNN). Computational overhead and energy consumption are significantly reduced. Hardware-friendly, low-power neuromorphic system. First feasibility study using a deep SNN for seizure detection on several reliable public datasets.

Three problems stand with the deep SNN (dSNN) conversion approach: (a) SNN conversion has not yet been optimized for sequential neural networks, and thus, temporal data is represented as an image. Forecasting models become prone to the future data leakage problem; (b) the SNN is an approximation of the deep neural network (DNN), and thus, the non-spiking network sets an upper-bound on performance, and (c) as the initial DNN is trained using error backpropagation, online learning for patient adaptation is no longer an option on resource-constrained hardware.

Converting raw EEG signals to the frequency domain enables better performance seizure for seizure identification.

---

>  Embedded Machine Learning and breast cancer - a wide range of low-power devices, including wearables, smartphones, and other IoT devices - future?