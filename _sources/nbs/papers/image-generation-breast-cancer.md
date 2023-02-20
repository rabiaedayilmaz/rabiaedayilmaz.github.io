# Image Generation - Breast Cancer - Papers

## BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2204.11425)
:::

```{figure} ../../assets/papers/bci-1.png
---
name: directive-fig
---
Samples from BCI Dataset
```

Human epidermal growth factor receptor 2 (HER2) is important for formulating a precise treatment of breast cancer. Evaluation of HER is performed with immunohistochemical techniques (IHC), however IHC is very costly to perform. Thus, in this paper, a breast cancer immunohistochemical (BCI) benchmark is proposed for the first time. The goal is to synthesize IHC data from the paired hematoxylin and eosin (HE) stained images. BCI dataset contains 4870 paired images with different expression levels (0, 1+, 2+, and 3+). Furthermore, a pyramid pix2pix universal image translation method is used. This paper, for the first time investigates this problem and tries to solve it.

Breast cancer is a common type in woman and leading cause of death. Accurate diagnosis and treatment are key factors to survival. Histopathological checking is a gold standard to identify cancer. It is done by staining tumor materials and getting hematoxylin and eosin (HE) slices that later will be observed by pathologists through the microscope or analyzing the digitized whole slice images (WSI). After diagnosis, preparing precise treatment is an essential step. For this step, expression of specific proteins are checked, such as HER2. Over expression of HER2 indicates tendency to aggressive clinical behaviour. However, to conduct evaluation of HER2 is really expensive. Therefore, it is  aimed to create HER2 images from IHC-stained slices.

IHC 0: no stain, IHC 1+: barely perceptible, IHC 2+: weak to moderate complete staining, and IHC 3+: complete and intense staining. 

```{figure} ../../assets/papers/bci-2.png
---
name: directive-fig
---
Examples of slices and HER2 expressions
```

```{figure} ../../assets/papers/bci-3.png
---
name: directive-fig
---
How the Dataset is Formed
```

The data scanning equipment is Hamamatsu NanoZommer S60. For each pathological tissue sample, the doctor will cut two tissue from it, one for HE staining and the other one for HER2 detection. Thus, there will be differences between two tissue samples and furthermore, samples will be stretched or squeezed during slice preparation. To align both images, registration process is followed and projection transformation that is done by mapping squares of images and moreover, elastix registraion is applied. After these steps, post-processing is applied to remove black border between blocks and fill it with surrounding content. Lastly, the blank or not-well aligned ares are filtered out.

```{figure} ../../assets/papers/bci-4.png
---
name: directive-fig
---
The Model: pix2pix
```

The L1 loss is directly calculates the difference between ground truth and generated image. Multi-scale loss is formed for scale transformation that applies low-pass filter to smooth the image and down-smapling smoothed image.

```{figure} ../../assets/papers/bci-5.png
---
name: directive-fig
---
The Overall Objective Function
```

The method used in this paper is outperformed.

```{figure} ../../assets/papers/bci-6.png
---
name: directive-fig
---
Benchmark Results
```

```{figure} ../../assets/papers/bci-7.png
---
name: directive-fig
---
HER2 Visualizations
```
The accuracy of the outcomes of the model is evaluated by pathologists and achieved 37.5% and 40.0% accuracy performance. Briefly, this is a challenging task and there is a need for more effective model.

---

## Breaking the Dilemma of Medical Image-to-image Translation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2110.06465)
:::

Supervised Pix2Pix and unsupervised Cycle-consistency are two models dominates the field of medical image-to-image translation. But both of them are not ideal. Moreover, it requires paired and well-pixel aligned images that makes it really challengable especially in medical field and not always feasible due to respiratory motion or anatomical changes between times of acquired paired images. Cycle-consistency works well on unpaired or misaligned images. However, accuracy performance is not optimal and may produce multiple solutions. To break this dilemma, in this paper, RegGAN is proposed for medical image-to-image translation. It is based on theory of "loss-correction". Misaligned target images are considered as noisy labels and generator is trained with an additional registration network. The main goal is to search for a common solution both for image-to-image translation and registration tasks. In this paper, it is demonstrated that RegGAN can be easily combined with these models and improve their performance. The key outcome of this paper is that they demonstrated using registrations improves significantly the performance of image-to-iamge translation because of adaptively eliminating the noise.

```{figure} ../../assets/papers/reggan-1.png
---
name: directive-fig
---
Comparison of Models
```

```{figure} ../../assets/papers/reggan-2.png
---
name: directive-fig
---
Correction Loss
```

```{figure} ../../assets/papers/reggan-3.png
---
name: directive-fig
---
Comparison of Results
```

---

## Diagnostic Strategies for Breast Cancer Detection: From Image Generation to Classification Strategies Using Artificial Intelligence Algorithms

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9322973/)
:::

Breast cancer is the leading death of women worldwide and according to World Health Organization (WHO) approximately 16% of diagnosed as malignant is the reason of that. Thus, early stage detection is important to have highest chance for survival. Breast cancer develops when any lump begins an angiogenesis process that causes the development of new blood vessels and capillaries from the existent vasculature. 
Its mortality rate is 69% in emergent countries. In emergent countries, late diagnosis increases this rate.

There are several technologies used to obtain breast tissue:

### Mammograpgy
It is used to screen breast tissue to detect abnormalities indicate cancer or another related diseases. It is recommend since it has 85% sensibility. Mammography uses low-doses of X-ray to form a picture of the internal breast tissue. To achieve this, breast is compressed by two platelets to mitigate the dispersion of the rays and obtain better picture without using high-doses of X-ray. Specialists look for the different zones like shape, size, contrast, edges, bright spots. The most common symptoms are calcifications and masses. Recently, Breast Tomosynthesis (BT) that allows 3D reconstruction and Contrast-Enhanced Mammography (CEM) that improves image resolution by injecting a contrast agent have been proposed. 
### Ultrasound
It is non-invasive and non-irradiating technique and useswaves to create images from breast. In order to create images, a transducer sends high-frequency sound waves (>=20kHz) and measures the reflected ones. The image is constructed by reflected wave sound from the internal tissues. Ultrasound has three purposes: i) assessing and determining the abnormality condition like solid or fluid-filled, ii) as an auxiliary scree tool when patient has dense breasts and mammography is not reliable enough, and iii) a guide to develop a biopsy in the suspected abnormality. 
To analyze ultrasound images several computer-aided diagnose (CAD) systems are proposed and their common objective is to improve resolution of the image. Another proposed method is micro-bubbles that are injected into the abnormalities detected at first sight.
Elastography is the technique to measure the tumor displacement when compressedusing a spatial transducer.
### Magnetic Resonance Imaging (MRI)
Breast MRI (BMRI) uses a magnetic field and radio waves to create a detailed image. Generally, 1.5T magnet with a contrast (usually gadolinium) is used. When the magnet is turned on, the magnetic field temporarily realigns the water molecules, so when radio waves are applied the emitted radiation is captured using specific-designed coils that are located at breast positions. These coils transform the captured radiation into electrical signals. The main goal is to get images of breast symmetry and the possible changes in the parenchymal tissue (reflection of the proportion of glandular tissue to fatty tissue). One of the problems of BMRI is that false-positive (specifity) rate, since this technique can detect low-size masses that are benign. To mitigate tihs issue, nanomaterials have been developed to stick to the cancer masses but not the benign ones as well as contrast agents.

There are other approaches such as microwave radiation, CT, PET etc.

Additionally, there is a recent image generation technique: Infrared Thermography (IRT). Temperature is an indicator of health. In breast cancer, when tumor exists it makes use of nutrients for its growth (angiogenesis) that results in increase of metabolism thus the temperature around the tumor. 

---

## Ea-GANs: Edge-Aware Generative Adversarial Networks for CrossModality MR Image Synthesis

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/8629301)
:::

Medical image synthesis is important topic. It maps from given source-modality image to unknown target-modality. There is wide range area of usage: virtual dataset creation, missing image imputation, image super-resolution etc. Currently there are two approaches: atlas-based methods that calculates atlas-to-image transformation in paired images (mostly healthy patient atlas data is available) and learning-based methods.

Magnetic resonance imaging (MRI) is widely used protocol and each MR modality (T1-w, T2-w, FLAIR etc.) reveals unique visual characteristics. To benefit from complementary information from multiple imaging modalities, *cross-modality MR synthesis* has gained attention. But most existing methods onyl focus on minimizing pixel/voxel-wise intensity difference but ignore textural details of the image content structure that affects the quality of synthesized image. So, in this paper a cross-modality MR image synthesis method is proposed. It is edge-aware generative adversarial network (EA-GAN). There, the edge information that represents textural structure and depicts the boundaries of different objects is integrated. Two learning strategies are proposed: gEA-GAN (generator-induced) and dEA-GAN (discriminator-induced). gEA-GAN integrates the edge information via its generator and dEA-GAN does same via both generator and discriminator, so that edge similarity is also learned adversarialy. Proposed EA-GANs are 3D based and utilize hierarchical features to capture contextual information. dEA-GAN outperforms and SOTA method for cross-modality MR image synthesis (07/2019) and it is generalizable. 

The edge maps are computed by using Sobel operator since it is simple and derivative can easily be computed.

```{figure} ../../assets/papers/eagan-1.png
---
name: directive-fig
---
Sobel Filter and Edges
```

```{figure} ../../assets/papers/eagan-2.png
---
name: directive-fig
---
Objective of Generator in gEA-GAN
```

```{figure} ../../assets/papers/eagan-3.png
---
name: directive-fig
---
Objective of Discriminator in gEA-GAN
```

```{figure} ../../assets/papers/eagan-4.png
---
name: directive-fig
---
Final Objective Function of gEA-GAN
```

```{figure} ../../assets/papers/eagan-5.png
---
name: directive-fig
---
Objective of Generator in dEA-GAN
```

```{figure} ../../assets/papers/eagan-6.png
---
name: directive-fig
---
Objective of Discriminator in dEA-GAN
```
Similarly, the final objective function is summation of generator and discriminator objectives.
Architecture consists of three modules: generator, discriminator and edge detector.
```{figure} ../../assets/papers/eagan-7.png
---
name: directive-fig
---
Architecture of EA-GANs
```

```{figure} ../../assets/papers/eagan-8.png
---
name: directive-fig
---
Architecture fo Generator
```

```{figure} ../../assets/papers/eagan-10.png
---
name: directive-fig
---
Results on BRATS2015 Dataset
```

```{figure} ../../assets/papers/eagan-11.png
---
name: directive-fig
---
Comparison with Pix2Pix on Different 2D Datasets
```

---

## Generative Adversarial Networks An Overview

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8253599)
:::

GANs provide a way to learn deep representations without extensively annotated data. They can be used in various tasks: image synthesis, semantic image editing, style transfer, image superresolution, and classfication. It is an emerging technique for both semisupervised and supervised learning. They achieve this by implicitly modeling high-dimensional distributions of the data. And can be used for various down-stream tasks such as semantic iamge editing, data augmentation, style transfer, image retrival etc. There are several GANs  architectures:

### Fully Connected GANs 
The first GAN architecture that used fully connected neural networks for both generator and discriminator.
### Convolutional GANs
DCGANs (deep convolutional gans) are dominant in this group. They make use of strided and fractionally stridede convolutions that allows spatial downsampling and upsampling.
### Conditional GANs
The conditional setting is both generator and discriminator networks are class-conditional. They have more advantage for multimodal data generation.
### GANs with Inference Models
The generator consists of two networks: the encoder(inference network) and the decoder. Both of them jointly trained to fool discriminator. The discriminator receives pairs of (x, z) vectors has to determine which pair constitute genuine tuple consisting of real image sample and its encoding or fake image sample and corresponding latent-space input to the generator.
In encoding-decoding model output is called as reconstruction.
### Adversarial Autoencoders
Autoencoders are composed from encoder and decoder. They learn nonlinear mappings in both directions.

There a couple of symptomps that GANs might suffer from:
* difficulties in getting the pair to converge
* the generative model collapsing to generator very similar samples for different inputs
* the discriminator loss converging quickly to zero so no reliable path for gradient updates to the generator

However, there are several training tricks: batch normalization, to minimize the number of fully connected layers, leaky ReLU between intermediate layers rather than ReLU, feature matching, minibatch discrimination, heuristic averaging, one-sided label smoothing.
For image-to-image translation models are pix2pix and cyclegan.

---

## Image-to-Image Translation with Conditional Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1611.07004.pdf)
:::

In this paper, a general solution for image-to-image translation by using conditional adversarial networks is investigated and pix2pix model is proposed. Moreover, not only learning the mapping from input image to target image but also learning a loss function to train this mapping. This paper shows that there is no-need no longer for hand-engineer own mappings. CNNs learn to minimize a loss function despite a lot of effort goese into designing effective losses. Briefly, we still need to teel CNN what wish to minimize. If a naive approach, like Euclidean distance, is taken then it will tend to produce blurry results. The reason is because Euclidean distance is minimized by averaging all the outputs that causes blurring. Notably, coming up with loss functions that force the CNN to do what we really want, like output sharp, realistic images etc., is an open problem and requires expert knowledge.  

*Structured losses for image modeling* Image2image translations are usually considered as per pixel classification or regression. These formulations treat output space as unstructured. And condiitonal GANs learn unstructured loss. Structured losses penalize the joint configuration of the output. In the literature, there so many methods that considers this problem: conditional random fields, the SSIM metric, feature matching, nonparametric losses, the convolutional pseudo-prior and losses based on matching covariance statistics.

pix2pix uses a U-Net  based architecture for generator and convolutional PatchGAN classfier for discriminator.

PatchGAN is also called as markovian discriminator. L2 and L1 produce generate blurry results so they will encourage low-level frequencies. To model high-frequencies, restricting attention on local patches is enough. This is how PatchGAN is proposed. Because it penalizes structure at the scale of the patches. Such a discriminator effectively models the image as a Markov random field by assuming independence between pixels seperated more than patch diameter. Lastly, PatchGAN can be considered as a form of texture/style loss. 

```{figure} ../../assets/papers/pix2pix-1.png
---
name: directive-fig
---
Final Objective Function
```

## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1703.10593.pdf)
:::

This method is for learning to translate an image from a source X domain to target image domain Y without using paired images. It has adversarial loss and cycle consistency loss. For the discriminator part, PatchGAN is used.

## Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2002.10964.pdf)
:::

GANs are awesome but often requires numerous training data and heavy computational resources. To overcome this issue several transfer learning approaches are proposed in GANs. However, they are prone to overfitting or limited learning small distribution shifts. In this paper, it is demonstrated that simple fine-tuning of GANs with frozen lower layers of the discriminator performs well.

---

## GANs for medical image analysis 

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1809.06222.pdf)
:::

This is a review paper about GANs that are used in medical image field. This paper categorized into seven parts: synthesis, segmentation, reconstruction, detection, de-noising,registration, and classification.

There are several GANs: DCGAN, Markovian GAN, conditional GAN, CycleGAN, auxiliary classifier GAN, Wasserstein GAN, least squares GAN.
### DCGAN
It adresses instability of GAN architecture and increases the resolution. In this model, both generator and discriminator follow a deep convolutional network by exploiting spatial kernels and hierarchical features. Batch-normalization and leaky-ReLU are included. But mode-collapse issue could not resolved completely.
### cGAN
In the original GAN paper, there was no explicitcontrol on the actual data. To address this issue, conditional GANs are proposed, thus they will incorporate additional information like class labels. In the cGAN, the generator is presented with random noise z jointly with some prior information c.
```{figure} ../../assets/papers/medganrev-1.png
---
name: directive-fig
---
cGAN Framework
```
Another conditional GAN framework is Markovian GAN (MGAN). It is proposed for fast and high-quality style transfer in images. Highly takes advantage of VGG19 for feature extraction.

Another successful variation of conditional GAN is pix2pix. The generator utilizes U-Net while discriminator uses a fully convolutional neural network similar to MGAN. It is showed that in the U-Net, the *skip connections* are beneficial for global coherence. Unlike original GAN, it requires image pairs. This allows the usage of L1 loss to stabilize training.
### CycleGAN
For image transformation between two domains, the model should extract characteristic features of both domains and discover underlying. To provide these criterias CycleGAN is proposed. The two GANs are chained together and a cyclic loss function forces them to reduce the space between their possible mapping functions. 
### AC-GAN
Auxiliary classifier GAN (AC-GAN) is proposed. Unlike the cGAN, they do not provide prior information. Instead the discriminator can be additionally tasked with respectively classifying its input. More precisely, discriminator is edited such that after a few layers it splits into a standard discriminator and auxiliary network that aims to classify samples into different categories. According to the authors, this partially allows to use pre-trained discriminators and appears to stabilize the model.
### WGAN
In the previous frameworks, the distributions of generated and real data are matched by means of the Jensen-Shannon (JS) divergence. This divergence measure causes vanishing gradients and makes the saddle-point optimization non-feasible that are underlying failures of GAN models. 

In Wasserstein-GAN (WGAN) that uses the Earth Mover (ME) or Wasserstein-1 distance as a more optimal divergence measure to avoid vanishing gradients.The downside of WGAN is slow optimization.
```{figure} ../../assets/papers/medganrev-2.png
---
name: directive-fig
---
Cycle and AC GANs
```
### LSGAN
Least-squares GANs tried to tackle with the training instability. Similar to WGAN, the loss function is modified to avoid vanishing gradients.

---

## GANs for Medical Image Synthesis: An Empirical Study

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2105.05318.pdf)
:::

In this paper, various GAN-architectures are tested from DCGAN to style-based GANs on three medical imaging modalities and organs: cardiac cine-MRI, liver CT and RGB retina images. Generating realistic-looking medical images by FID (Fréchet Inception Distance score) standards passed the Truing test, however segmentation results were not much satisfying.

```{figure} ../../assets/papers/emprical-1.png
---
name: directive-fig
---
Used GANs
```

There are three main issues about GANs: convergence, vanishing gradients and mode collapse.
There are several GANs: DCGAN, LSGAN, WGAN and WGAN-GP, HingeGAN, SPADE GAN(improvement of pix2pix, SOTA 2021) and StyleGAN.
There are several evaluation metrics for GANs: Peak Signal to Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), Inception Score (IS), Frechet Inception Distance (FID but can not detect if GAN just memorizes the training set and suffers from high bias). GANs are highly sensitive to hyperparameters but hyperparameter space research takes 500 GPU-days. So, regarding to earlier studies parameters are chosen. SPADE GAN and StyleGAN did most, respectively.

```{figure} ../../assets/papers/emprical-2.png
---
name: directive-fig
---
Selected Hyperparameters
```

---

## High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1711.11585.pdf)
:::

In this paper, a new method is presented that synthesizes high-resolution photo realistic images from semantic label maps  using cGANs. Generally results of cGANs are limited to low-resolution and far from realistic. However, in this paper 2048x1024 results are generated with a novel adversarial loss, new multi-scale generator and discrimiantor architectures. And, two interactive frameworks are represented: object instance segmentation information is enabled to remove/add objects or changing the category and a method to generate diverse results given the same input. pix2pix model is used as baseline. It is improved by using a coarse-to-fine generator, a multi-scale discriminator and a robust adversarial network objective function.

The coarse-to-fine generator has two parts: G1 (global generator) and G2 (local enhancer).

```{figure} ../../assets/papers/high-res-1.png
---
name: directive-fig
---
Generator Architecture 
```

The full objective function is combination of GAN loss and feature matching loss (related to perceptual loss). Lambda controls the importance of each loss.

```{figure} ../../assets/papers/high-res-2.png
---
name: directive-fig
---
Objective Function
```

Existing image synthesis methods utilizes semantic label maps, despite this fact, in this paper instance label maps are used. It improved performance. Mapping from semantic map is one-to-many problem. But authors proposed low-dimensional feature channels as the input to the generator. To generate low dimensional features, an encoder network E to find low-dimensional feature vectors. To ensure that features are consistent within in each instance, an instance-wise average pooling layer to the output of the encoder to compute average feature for object instances. After training the encoder, it runned on all instances in the training images and record the obtained features. Then, k-clustering is performed for each semantic category. These features are used as input to the generator.

```{figure} ../../assets/papers/high-res-3.png
---
name: directive-fig
---
Overall Network
```

---

## Synthetic Medical Images from Dual Generative Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1709.01872.pdf)
:::

In this paper, a novel two-stage pipeline generating synthethic medical images from a pair of generative adversarial networksmethod is proposed that is called SynthMed. Tests are conducted on retinal fundi images. Also a hierarchical generation process to divide complex image generation task into two parts is proposed: geometry and photorealism.

Stage-1 GAN: Produces segmentation masks that represent the variable *geometries* of the dataset. It is based on DCGAN. Cross-entropy loss is used.
Stage-2 GAN: Translate the masks produced in Stage 1 into *photorealistic* images. It is based on cGAN. 

Stacking GANs is effective since inhibits unstable nature of GANs. The lack of detail is unacceptable in medical image generation.

---

## High-resolution medical image synthesis using progressively grown generative adversarial networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1805.03144.pdf)
:::

In this paper, GANs are applied to medical images and generated synthetic data: fundus and multi-modal MR glioma. PGGANs are explored in medical images and showed that it is successful in being realistic and diverse. 

```{figure} ../../assets/papers/ggan-1.png
---
name: directive-fig
---
PGGAN Architecture
```

PGGAN trains the network in a step-wise style.

---

## Image synthesis with adversarial networks: A comprehensive survey and case studies

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2012.13736.pdf)
[Extensive Collection of Reviewed Papers](https://github.com/pshams55/GAN-Case-Study)
:::

This is a comprehensive review paper that summarizes synthetic image generation methods and discuss the categories (image-to-image translation, fusion image generation, label-to-image mapping). 

```{figure} ../../assets/papers/gan-rev-1.png
---
name: directive-fig
---
GAN Architectures Chronology
```

Here, GAN types will be reviewed.

* **Convolutional GANs:** Moving from FC to CNN is appropriate for image data. However, experiments showed that training with CNNs is hard because of: non-convergence, diminished gradient, unbalance between discriminator&generator, model collapse, and hyperparameter selections. One solution is using Laplacian pyramids adversarial networks where a real image is converted into a multi scale pyramid image and convolutional GAN is trained to produce multi scale and multi level feature maps where final map is combination of all. The Laplacian pyramid is a linear invertible image demonstration containing band-pass images and a low-frequency residual.

```{figure} ../../assets/papers/gan-rev-2.png
---
name: directive-fig
---
Laplacian Pyramid
```

* **Conditional GANs:** Proposed for image-to-image translation problem. It just not learns the mapping from input image to output image but also adopts a loss function to train this mapping. This provides opportunity to apply same generic method to problems that need complex loss formulations. There proposed InfoGAN (uses mutual info. so semantically meaningful), BAGAN (class conditioning in hidden space, similar to infogan but has two outputs) and ACGAN (similar infogan but no c conditional var. and added external classifier and optimized loss func.).

* **Autoencoder GANs:** Autoencoders learn a deterministic mapping via the encoder and decoder. They are generally for learning non-linear mappings in both directions. Images generated by autoencoder gans are blurry but accurate and efficient.

```{figure} ../../assets/papers/gan-rev-3.png
---
name: directive-fig
---
Autoencoder GAN Loss
```

There is also BiGAN, AGE, BEGAN etc. 

* **Progressive and Classifier GAN:** Idea came from progressive neural nets. It has high performance as it can receive additional leverage via lateral connections to earlier learned features. This architecture is widely used to extracy complex features and is stable. They have significant performance in tasks such as img2img translation, text2img synthesis.

* **Adversarial Domain Adaption:** ADDA, CycleGAN, CyCADA, DiscoGAN, AugGAN, DualGAN.

### Synthetic image generation methods

* **Single Stage Methods:** These type of GANs follow a generator G and a discriminator D architecture. They have simple architecture and no additional connections. DCGAN, ControlGAN, ClusterGAN.

* **Multi Stage Methods:** They use multiple generators and discriminators. Generators are in charge of different tasks. The idea behind this approach is to distinct an image into different portions, like foreground-background, style-structure. There, generators work in sequential or parallel. StructureGAN, CR-GAN, StarGAN, StarGAN-VC, StackGAN, AttenGAN, MC-GAN.

```{figure} ../../assets/papers/gan-rev-4.png
---
name: directive-fig
---
Various GANs
```

---

## Medical Image Synthesis for Data Augmentation and Anonymization Using Generative Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://link.springer.com/chapter/10.1007/978-3-030-00536-8_1)
:::

In  this paper, 3D GANs are used for data augmentation and then, the augmented data used on brain tumors dataset. It is observed that the performance got better. pix2pix is used and adopted to translate label-to-MRI (synthetic image generation) and MRI-to-label(image segmentation).

---

## Medical Image Synthesis with Generative Adversarial Networks for Tissue Recognition

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/8419363)
:::

DCGAN, WGAN and BEGAN are applied and compared on thyroid images. WGANs and BEGANs outperformed.

---

## Multimodal Unsupervised Image-to-Image Translation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)
:::

In computer vision unsupervised image-to-image translation is important. But they fail to generate diverse outputs from a given source domain image. To address this issue, Multimodal Unsupervised Image-to-Image Translation (MUNIT) framework is proposed. They assumed that the image representation can be decomposed into a content code that is domain invariant, and a style code that captures domain-specific properties. To translate an image to another domain, they recombined its content code with a random style code sampled from the style space of the target.

```{figure} ../../assets/papers/gan-rev-5.png
---
name: directive-fig
---
Proposed Framework
```

---

## ResViT: Residual vision transformers for multi-modal medical image synthesis

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2106.16031.pdf)
:::

CNNs are desgined to perform local processing with compact filters and this inductive bias compromises learning of contextual features. In this paper, ResVit method is proposed that leverages the contextual sensitivity of vision transformers. ResVit's generator employs a central bottleneck comprising novel aggregated residual transformer (ART) blocks that synergistically combine residual convolutional and transformer modules. Residual connections in ART blocks promote diversity in captured representations, while a channel compression module distills task-relevant information. A weight sharing strategy is used among ART to relief computational burden. A unified implementation is introduced to avoid the need to rebuild separate synthesis models for varying source-target modality configurations. Experiments are conducted for synthesis of multi contrasted MRI and CT from MRI images.

Medical imaging has a crucial role in healtcare since it enables in vivo examination of pathology. In many scenarios it is desirable to have multi modal protocols from multiple scanners(MRI, CT etc.) or multiple acquisitons from a single scanner(multi-contrast MRI). Complementary information empower physicians as causing high confidence and accuracy. But numerous factors such as uncooperative patient and excessive scan times prohibit ubiquitous multi modal imaging. Therefore, there is an increasing need for synthesizing unacquired images in multi modal protocols from the subset of available images, bypassing costs related to additional scans.

Vision transformers are highly promising since attention operators that learn contextual features can improve sensitivity for long-range interactions and focus on critical image regions for improved generalization to atypical anatomy such as lesions. But adopting vanilla transformers in this pixel level outputs is challenging because of the computational burden and limited localization. Thus, recent studies consider hybrid architectures or computation-efficient attention operators to adopt transformers in medical imaging tasks.

ResVit combines the sensitivity of vision transformers to global context, the localization power of CNNs, and the realism of adversarial learning. ResVit's generator follows an encoder-decoder architecture with a central bottleneck to distill task-critical information. The encoder and decoder contain CNN blocks to leverage local precision of convolution operators. The bottleneck comprises novel aggregated residual transformer (ART) blocks to preserve local and gloabal context, with a weight sharing strategy to minimize model complexity.

```{figure} ../../assets/papers/resvit-1.png
---
name: directive-fig
---
ResVit Model
```

---

## StarGAN v2: Diverse Image Synthesis for Multiple Domains

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/1912.01865.pdf)
:::

A good image model should learn i) diversit of generated images and ii) scalability over multiple domains. StarGAN deals with both.

```{figure} ../../assets/papers/stargan-1.png
---
name: directive-fig
---
Four Modules of StarGAN
```

---

## Unsupervised Multi-Modal Medical Image Registration via Discriminator-Free Image-to-Image Translation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2204.13656.pdf)
:::

In this paper, a novel translation-based unsupervised deformable image registration approach to convert the multi-modal registration problem to mono-model one is proposed. Concretely, this approach incorporates a discriminator-free translation network to facilitate the training of the registration network and a patchwise contrastive loss  to encourage the translation network to preserve object shapres. Thus, main idea is to reduce the inconsistency and artifacts of the translation by removing discriminator. Moreover, replacing adversarial loss with novel two losses (local alignment and global alignment) is proposed so that an unsupervised method requiring no ground truth deformation or pairs of aligned images for training. Local alignment loss is for capturing detailed local texture information and global alignment loss is for focusing on the overall shape. Four variants of the approach evaluated on a public dataset. According to experiment results, it achieved SOTA performance (04/2022).

```{figure} ../../assets/papers/multi-1.png
---
name: directive-fig
---
Overview of Method
```

```{figure} ../../assets/papers/multi-2.png
---
name: directive-fig
---
More Detailed Overview of Method
```

---

## WHEN, WHY, AND WHICH PRETRAINED GANS ARE USEFUL?

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2202.08937.pdf)
:::

The goal of this work is to scrutinize the process of GAN finetuning. There are three points of this work: i) pretrained checkpoint affects model's coverage, ii) pretrained generators and discriminators are important and iii) a simple recipe to select an appropriate GAN checkpoint that is most suitable for finetuning is described.

For iii., it is considered that a starting checkpoint optimal if it provides the lowest FID score or its FID score differs from the lowest by most 5%.