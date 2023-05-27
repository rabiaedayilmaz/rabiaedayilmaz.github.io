# Image Generation/GANs/Instance Segmentation - Breast Cancer(mostly) - Papers

## BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2204.11425)
:::

Human epidermal growth factor receptor 2 (HER2) is important for formulating a precise treatment of breast cancer. Evaluation of HER is performed with immunohistochemical techniques (IHC), however IHC is very costly to perform. Thus, in this paper, a breast cancer immunohistochemical (BCI) benchmark is proposed for the first time. The goal is to synthesize IHC data from the paired hematoxylin and eosin (HE) stained images. BCI dataset contains 4870 paired images with different expression levels (0, 1+, 2+, and 3+). Furthermore, a pyramid pix2pix universal image translation method is used. This paper, for the first time investigates this problem and tries to solve it.

Breast cancer is a common type in woman and leading cause of death. Accurate diagnosis and treatment are key factors to survival. Histopathological checking is a gold standard to identify cancer. It is done by staining tumor materials and getting hematoxylin and eosin (HE) slices that later will be observed by pathologists through the microscope or analyzing the digitized whole slice images (WSI). After diagnosis, preparing precise treatment is an essential step. For this step, expression of specific proteins are checked, such as HER2. Over expression of HER2 indicates tendency to aggressive clinical behaviour. However, to conduct evaluation of HER2 is really expensive. Therefore, it is  aimed to create HER2 images from IHC-stained slices.

IHC 0: no stain, IHC 1+: barely perceptible, IHC 2+: weak to moderate complete staining, and IHC 3+: complete and intense staining. 

The data scanning equipment is Hamamatsu NanoZommer S60. For each pathological tissue sample, the doctor will cut two tissue from it, one for HE staining and the other one for HER2 detection. Thus, there will be differences between two tissue samples and furthermore, samples will be stretched or squeezed during slice preparation. To align both images, registration process is followed and projection transformation that is done by mapping squares of images and moreover, elastix registraion is applied. After these steps, post-processing is applied to remove black border between blocks and fill it with surrounding content. Lastly, the blank or not-well aligned ares are filtered out.

The L1 loss is directly calculates the difference between ground truth and generated image. Multi-scale loss is formed for scale transformation that applies low-pass filter to smooth the image and down-smapling smoothed image.

The method used in this paper is outperformed.

The accuracy of the outcomes of the model is evaluated by pathologists and achieved 37.5% and 40.0% accuracy performance. Briefly, this is a challenging task and there is a need for more effective model.

---

## Breaking the Dilemma of Medical Image-to-image Translation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2110.06465)
:::

Supervised Pix2Pix and unsupervised Cycle-consistency are two models dominates the field of medical image-to-image translation. But both of them are not ideal. Moreover, it requires paired and well-pixel aligned images that makes it really challengable especially in medical field and not always feasible due to respiratory motion or anatomical changes between times of acquired paired images. Cycle-consistency works well on unpaired or misaligned images. However, accuracy performance is not optimal and may produce multiple solutions. To break this dilemma, in this paper, RegGAN is proposed for medical image-to-image translation. It is based on theory of "loss-correction". Misaligned target images are considered as noisy labels and generator is trained with an additional registration network. The main goal is to search for a common solution both for image-to-image translation and registration tasks. In this paper, it is demonstrated that RegGAN can be easily combined with these models and improve their performance. The key outcome of this paper is that they demonstrated using registrations improves significantly the performance of image-to-iamge translation because of adaptively eliminating the noise.

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


Similarly, the final objective function is summation of generator and discriminator objectives.
Architecture consists of three modules: generator, discriminator and edge detector.

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

Another conditional GAN framework is Markovian GAN (MGAN). It is proposed for fast and high-quality style transfer in images. Highly takes advantage of VGG19 for feature extraction.

Another successful variation of conditional GAN is pix2pix. The generator utilizes U-Net while discriminator uses a fully convolutional neural network similar to MGAN. It is showed that in the U-Net, the *skip connections* are beneficial for global coherence. Unlike original GAN, it requires image pairs. This allows the usage of L1 loss to stabilize training.
### CycleGAN
For image transformation between two domains, the model should extract characteristic features of both domains and discover underlying. To provide these criterias CycleGAN is proposed. The two GANs are chained together and a cyclic loss function forces them to reduce the space between their possible mapping functions. 
### AC-GAN
Auxiliary classifier GAN (AC-GAN) is proposed. Unlike the cGAN, they do not provide prior information. Instead the discriminator can be additionally tasked with respectively classifying its input. More precisely, discriminator is edited such that after a few layers it splits into a standard discriminator and auxiliary network that aims to classify samples into different categories. According to the authors, this partially allows to use pre-trained discriminators and appears to stabilize the model.
### WGAN
In the previous frameworks, the distributions of generated and real data are matched by means of the Jensen-Shannon (JS) divergence. This divergence measure causes vanishing gradients and makes the saddle-point optimization non-feasible that are underlying failures of GAN models. 

In Wasserstein-GAN (WGAN) that uses the Earth Mover (ME) or Wasserstein-1 distance as a more optimal divergence measure to avoid vanishing gradients.The downside of WGAN is slow optimization.
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

There are three main issues about GANs: convergence, vanishing gradients and mode collapse.
There are several GANs: DCGAN, LSGAN, WGAN and WGAN-GP, HingeGAN, SPADE GAN(improvement of pix2pix, SOTA 2021) and StyleGAN.
There are several evaluation metrics for GANs: Peak Signal to Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), Inception Score (IS), Frechet Inception Distance (FID but can not detect if GAN just memorizes the training set and suffers from high bias). GANs are highly sensitive to hyperparameters but hyperparameter space research takes 500 GPU-days. So, regarding to earlier studies parameters are chosen. SPADE GAN and StyleGAN did most, respectively.

---

## High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1711.11585.pdf)
:::

In this paper, a new method is presented that synthesizes high-resolution photo realistic images from semantic label maps  using cGANs. Generally results of cGANs are limited to low-resolution and far from realistic. However, in this paper 2048x1024 results are generated with a novel adversarial loss, new multi-scale generator and discrimiantor architectures. And, two interactive frameworks are represented: object instance segmentation information is enabled to remove/add objects or changing the category and a method to generate diverse results given the same input. pix2pix model is used as baseline. It is improved by using a coarse-to-fine generator, a multi-scale discriminator and a robust adversarial network objective function.

The coarse-to-fine generator has two parts: G1 (global generator) and G2 (local enhancer).

The full objective function is combination of GAN loss and feature matching loss (related to perceptual loss). Lambda controls the importance of each loss.

Existing image synthesis methods utilizes semantic label maps, despite this fact, in this paper instance label maps are used. It improved performance. Mapping from semantic map is one-to-many problem. But authors proposed low-dimensional feature channels as the input to the generator. To generate low dimensional features, an encoder network E to find low-dimensional feature vectors. To ensure that features are consistent within in each instance, an instance-wise average pooling layer to the output of the encoder to compute average feature for object instances. After training the encoder, it runned on all instances in the training images and record the obtained features. Then, k-clustering is performed for each semantic category. These features are used as input to the generator.
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

Here, GAN types will be reviewed.

* **Convolutional GANs:** Moving from FC to CNN is appropriate for image data. However, experiments showed that training with CNNs is hard because of: non-convergence, diminished gradient, unbalance between discriminator&generator, model collapse, and hyperparameter selections. One solution is using Laplacian pyramids adversarial networks where a real image is converted into a multi scale pyramid image and convolutional GAN is trained to produce multi scale and multi level feature maps where final map is combination of all. The Laplacian pyramid is a linear invertible image demonstration containing band-pass images and a low-frequency residual.

* **Conditional GANs:** Proposed for image-to-image translation problem. It just not learns the mapping from input image to output image but also adopts a loss function to train this mapping. This provides opportunity to apply same generic method to problems that need complex loss formulations. There proposed InfoGAN (uses mutual info. so semantically meaningful), BAGAN (class conditioning in hidden space, similar to infogan but has two outputs) and ACGAN (similar infogan but no c conditional var. and added external classifier and optimized loss func.).

* **Autoencoder GANs:** Autoencoders learn a deterministic mapping via the encoder and decoder. They are generally for learning non-linear mappings in both directions. Images generated by autoencoder gans are blurry but accurate and efficient.

There is also BiGAN, AGE, BEGAN etc. 

* **Progressive and Classifier GAN:** Idea came from progressive neural nets. It has high performance as it can receive additional leverage via lateral connections to earlier learned features. This architecture is widely used to extracy complex features and is stable. They have significant performance in tasks such as img2img translation, text2img synthesis.

* **Adversarial Domain Adaption:** ADDA, CycleGAN, CyCADA, DiscoGAN, AugGAN, DualGAN.

### Synthetic image generation methods

* **Single Stage Methods:** These type of GANs follow a generator G and a discriminator D architecture. They have simple architecture and no additional connections. DCGAN, ControlGAN, ClusterGAN.

* **Multi Stage Methods:** They use multiple generators and discriminators. Generators are in charge of different tasks. The idea behind this approach is to distinct an image into different portions, like foreground-background, style-structure. There, generators work in sequential or parallel. StructureGAN, CR-GAN, StarGAN, StarGAN-VC, StackGAN, AttenGAN, MC-GAN.

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

---

## StarGAN v2: Diverse Image Synthesis for Multiple Domains

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/1912.01865.pdf)
:::

A good image model should learn i) diversit of generated images and ii) scalability over multiple domains. StarGAN deals with both.

---

## Unsupervised Multi-Modal Medical Image Registration via Discriminator-Free Image-to-Image Translation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2204.13656.pdf)
:::

In this paper, a novel translation-based unsupervised deformable image registration approach to convert the multi-modal registration problem to mono-model one is proposed. Concretely, this approach incorporates a discriminator-free translation network to facilitate the training of the registration network and a patchwise contrastive loss  to encourage the translation network to preserve object shapres. Thus, main idea is to reduce the inconsistency and artifacts of the translation by removing discriminator. Moreover, replacing adversarial loss with novel two losses (local alignment and global alignment) is proposed so that an unsupervised method requiring no ground truth deformation or pairs of aligned images for training. Local alignment loss is for capturing detailed local texture information and global alignment loss is for focusing on the overall shape. Four variants of the approach evaluated on a public dataset. According to experiment results, it achieved SOTA performance (04/2022).

---

## WHEN, WHY, AND WHICH PRETRAINED GANS ARE USEFUL?

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2202.08937.pdf)
:::

The goal of this work is to scrutinize the process of GAN finetuning. There are three points of this work: i) pretrained checkpoint affects model's coverage, ii) pretrained generators and discriminators are important and iii) a simple recipe to select an appropriate GAN checkpoint that is most suitable for finetuning is described.

For iii., it is considered that a starting checkpoint optimal if it provides the lowest FID score or its FID score differs from the lowest by most 5%.

## BACH: Grand challenge on breast cancer histology images

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/1808.04277.pdf)
:::

---

## Breast cancer histopathological image classification using attention high-order deep network

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/7727519)
:::

This paper introduces a novel attention high-order deep network (AHoNet) to capture more discriminant deep features for breast cancer pathological images by simultaneously embedding attention mechanism and high-order statistical representation into a residual convolutional network. AHoNet gains the optimal patient-level
classification accuracies of 99.29% and 85% on the BreakHis and BACH database, respectively.

AHoNet ->  efficient channel attention module with non-dimensionality reduction + local cross-channel
interaction to achieve local salient deep features + matrix power normalization (more robust global feature presentation)

---

## Breast Cancer Histopathological Image Classification using Deep Learning

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/document/8965027)
:::

The paper employs deep learning to classify breast cancer histopathological image into benign and malignant categories. The Inception v1 convolutional neural network is mainly adopted. Spatial Pyramid Pooling and special global average pooling are added to the network to ensure that images can be imported in the original aspect ratio format. The experimental results show that the convolutional neural network performs well in breast cancer image classification, and the Global Average Pooling effect is slightly better than the Spatial Pyramid Pooling.

---

## Deep transfer with minority data augmentation for imbalanced breast cancer dataset

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S1568494620306979)
:::

The imbalanced class distribution results in the degradation of performance. A novel learning strategy that involves a deep transfer network has been proposed in this paper. DCGAN is used in the
initial phase for data augmentation of the minority class (benign) only. The dataset, with the class distribution now balanced, is applied as input to the deep transfer network. 

---

## Experimental Assessment of Color Deconvolution and Color Normalization for Automated Classification of Histology Images Stained with Hematoxylin and Eosin

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://pubmed.ncbi.nlm.nih.gov/33187299/)
:::

Here, it is investigated whether color preprocessing—specifically color deconvolution
and color normalization—could be used to correct such variability and improve the performance of
automated classification procedures and found that doing no color preprocessing was the best option in
most cases.

---

## Fusing of Deep Learning, Transfer Learning and GAN for Breast Cancer Histopathological Image Classification

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://link.springer.com/chapter/10.1007/978-3-030-38364-0_23)
:::

Biomedical image classification often deals with limited training sample due to the cost of labeling data. In this paper, they propose to combine deep learning, transfer learning and generative adversarial network (stylegan and pix2pix) to improve the classification performance. GANs made images noisy. :(

---

## GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S0925231218310749)
:::

Obtaining large datasets in the medical domain remains a challenge. They present methods for generating synthetic medical images using recently presented deep learning Generative Adversarial Networks (GANs). Furthermore, they show that generated medical images can be used for synthetic data augmentation, and improve the performance of CNN for medical image classification.

---

## Multiclass classifcation of breast cancer histopathology images using multilevel features of deep convolutional neural network

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.nature.com/articles/s41598-022-19278-2)
:::

They utilized six intermediate layers of the pre-trained Xception model to extract salient features from input images. They first optimized the proposed architecture on the unnormalized dataset, and then evaluated its performance on normalized datasets resulting from Reinhard, Ruifrok, Macenko, and Vahadane stain normalization procedures. Overall, it is concluded that the proposed approach provides a generalized state-of-the-art classifcation performance towards the original and normalized datasets. Also, it can be deduced that even though the aforementioned stain normalization methods offered competitive results, they did not outperform the results of the original dataset.

---

## SYNTHETIC DATA AUGMENTATION USING GAN FOR IMPROVED LIVER LESION CLASSIFICATION

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/1801.02385)
:::

In this paper, they present a data augmentation method that generates synthetic medical images using Generative Adversarial Networks (GANs). We propose a training scheme that first uses classical data augmentation to enlarge the training set and then further enlarges the data size and its diversity by applying GAN techniques for synthetic data augmentation. And achieved a significant improvement of 7% using synthetic augmentation over the classic augmentation.

---

## Two-Stage Convolutional Neural Network for Breast Cancer Histology Image Classification

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://link.springer.com/chapter/10.1007/978-3-319-93000-8_81)
:::

Due to the large size of each image in the training dataset, we propose a patch-based technique which consists of two consecutive convolutional neural networks. The first “patch-wise” network acts as  an auto-encoder that extracts the most salient features of image patches while the second “imagewise” network performs classification of the whole image. The main contribution of this work is presenting a pipeline which is able to process large scale images using minimal hardware.

---

## GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1810.10863.pdf)
:::

Especially usage of machine learning in medical imaging has a major obstacle: lack of data (and results in overfitting) and experts to annotate. GANs are a possible remedy (observed between 1 and 5 percentage uplift). Synthetic data can reduce overfitting significantly. Boost up accuracy and generalizability.

Progressive Growing of GANs (PGGAN) are used to generate synthetic data. From a paper[14th reference], it is suggested that different GAN architectures produce results which are, on average, not significantly different from each other.

One major advantage that traditional augmentation has over GAN augmentation is the ability to extrapolate. GANs can provide an effective way to fill in
gaps in the discrete training data distribution and augment sources of variance
which are difficult to augment in other ways, but will not extend the distribution beyond the extremes of the training data. 

Extrapolate -> traidional augmentation, Intrapolate -> GANs

---

## Self-Ensembling With GAN-Based Data Augmentation for Domain Adaptation in Semantic Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Self-Ensembling_With_GAN-Based_Data_Augmentation_for_Domain_Adaptation_in_Semantic_ICCV_2019_paper.html)
:::

Semantic segmentation suffers from insufficient data. Possible solution: unsupervised domain adaptation. In this paper, a self-ensembling technique that is generally used for classification is proposed. However, heavily-tuned manual data augmentation used in self-ensembling is not useful. To overcome this limitation, proposed a novel framework: data augmentation via GANs + self-ensembling.

For lack of data, ther is a technique: data synthesis but it has domain shift (different distribution) problem so does not perform well. Unsupervised domain adaptation handles domain shift by transferring knowledge from the labeled dataset in the source domain to the unlabeled dataset in the target domain.

Proposed framework is called as Target-Guided and Cycle-Free Data Augmentation (TGCF-DA). The first method is to generate labeled augmented data. And then, two segmentation networks as the teacher and the student in order to implement the self-ensembling algorithm.

Self-ensembling is composed of a teacher and a student network. Student is compelled to produce consistent predictions provided by the teacher on the target data. Teacher is an ensembled model that averages students' weights. Predictions from the teacher on target data can be thought as the pseudo labels for the students. Self-ensembling proved its efficiency in classification and requires heavily-tuned manual-data augmentation. However, such data augmentation + geometric transformations are great for classification task, it is not suited to minimize the domain shift in semantic segmentation. But, two different geometric transformations on each input can cause spatial misalignment between the student and the teacher predictions. No worries! Here, a novel data augmentation framework is proposed.

---

## ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content/WACV2021/html/Olsson_ClassMix_Segmentation-Based_Data_Augmentation_for_Semi-Supervised_Learning_WACV_2021_paper.html)
:::

Uh, lack of data is a big problem, again. To resolve this issue, semi supervised methods are utilized. Here, a novel data augmentation mechanism, ClassMix, is proposed. It generates augmentations by mixing unlabelled samples. However, augmentation techniques proved their inefficiency in semi superviesd learning. Recent approaches try to overcome this issue by applying: i) adding perturbations on an encoded state of the network instead of the input, ii) using augmentation technique CutMix to enforce consistent predictions over mixed samples.
 
ClassMix is a segmentation based data augmentation strategy and describe how it can be used for semi supervised semantic segmentation. Entropy minimization + pseudo labelling. It creates augmented images and artificial labels.

---

## Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html?ref=https://githubhelp.com)
:::

In this paper, copy-paste augmentation (randomly paste objects in various scales onto image) is performed. Annatotating large datasets for instance segmentation is reaaaally expensive and time consuming. Copy-paste is similar to mixup  and cutmix but only copying the exact pixels corresponding to an object as opposed to all pixels in the object's bounding box. They do not use geometric transformations, Gaussian blurring.
Copy and paste approach is also used for weakly supervised instance segmentation.

Copy paste is a strong data augmentation technique: robustness to backbone initialization, robustness to training schedules, additive to large scale jittering augmentation and works well across backbones + image sizes. Large Scale Jittering (LSJ) + copy paste works freaking awesome, better than LSJ+[mixup](https://paperswithcode.com/method/mixup). Also, copy paste does not increase the training cost or inference time!!

---

## Improving Data Augmentation for Medical Image Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openreview.net/pdf?id=rkBBChjiG)
:::

Here, mixup augmentation technique and its performance on medical images is investigated. It boosts up the performance. In mixup, images from training set are such linearly combined that it is a linear combination of two training data. Also, they proposed mixmatch method that is like mixup but not totatlly random and its motivation is medical data is highly imbalanced. It seems okay technique but mixup is better.

---

## Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2012.08548.pdf)
:::

There, they found the problem with EQL that is imbalanced gradients between positives and negatives. New version is gradient guided reweighing mechanism that rebalances the training process for each category indepedently and equally. Aaand EQLv2 >> EQL. EQL makes imrpovements on long-tailed dataset although end to end and decoupled training approaches work still better.

---

## Seesaw Loss for Long-Tailed Instance Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.html)
:::

Seesaw loss is SOTA(2021). It improves performance of long tailed dataset as it is dynamic, ditribution-agnostic and self-calibrated. 

---

## ResNeSt: Split-Attention Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2004.08955v2.pdf)
:::

For visual recognition, featuremap attention and multi-path representation are important that utilize cross feature interactions and learning diverse representations. ResNeSt >> EffcientNet. Better speed accuracy trade off. Instance Segmentation part: this backbone is better.

---

## YOLACT: Real-Time Instance Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html)
:::

Training on only one GPU and achieves better accuracy, woho. They achieved this by breaking instance segmentation into two parallel subtasks: i) generating prototype masks and ii) predicting per instance mask coefficients. Then masks are produced by linearly combining the prototypes with the mask coefficients. Sicne this process does not depend on repooling, it produces very high quality masks and exhibits temporal stability. Finally, fast NMS is proposed. First real-time instance segmentation algorithm with competitive results. Design of network closely follow RetinaNet but... faster sonic boom. In fast nms, already removed detections supress other detections. But results in a little performance loss so added semantic loss additionally.

---

## Instance Segmentation of Indoor Scenes using a Coverage Loss

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://people.csail.mit.edu/dsontag/papers/SilSonFer_ECCV14.pdf)
:::

It is noted that the major limitation of semantic segmentation is that not being able to distinguish different objects in the same class. Here, a model is introduced that utilizes both semantic and instance segmentation simultaneously. Also ,a new higher-order loss function. However, searching over semantic and instance seg. space is computationally infeasible. So segmentation tree is used.

---

## blob loss: instance imbalance aware loss functions for semantic segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2205.08209.pdf)
:::

Sørensen–Dice coefficient can tackle class imbalance however not aware of instance imbalance. Here, a novel family of loss functions is proposed by primarily aimed at maximizing instance level detection metrics. This func. is investigated mainly on medical datasets. 

---
 
## Efficient end-to-end learning for cell segmentation with machine generated weak annotations

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.nature.com/articles/s42003-023-04608-5#data-availability)
:::

A new model architecture for end to-end training using such incomplete annotations and machine-generated annotations are used. Model is LACSS (Location assisted cell segmentation system).

Often, amount of annotated data is inversely correlated with model performance in weakly and self supervised learning. Here, they focused on a specific subtype of  weak annotations and designed anew model architectue for end-to-end training using such incomplete annotations. Results are competitive and sometimes surpass sota, so it is promising.

---

## A full data augmentation pipeline for small object detection based on generative adversarial networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S0031320322004782)
:::

Proposed a full pipeline to generate data with GANs for small obejct detection. It generates new population of objects on an image.

---

## A survey of semi‑ and weakly supervised semantic segmentation of images

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://link.springer.com/article/10.1007/s10462-019-09792-7)
:::

Semisupervised and weakly supervised learning are gradually replacing fully supervised learning because good results with a lower cost. 

---

## Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-weighting

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2005.02066)
:::

Unsupervised domain adaptation is important. Cycle Consistency Panoptic Domain Adaptive Mask R-CNN
(CyC-PDAM) architecture is proposed for unsupervised nuclei segmentation in histopathology images. Also, a reweighting mechanism to dynamically add trade off weights for the task specific loss functions.

---

## Self-Supervised Visual Feature Learning With Deep Neural Networks: A Survey

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/1902.06162)
:::

Nice survey, just read it :)

---

## Cut and Learn for Unsupervised Object Detection and Instance Segmentation

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/abs/2301.11320)
:::

Cut-and-LEaRn (CutLER) that is a simple approach for training unsupervised object detection and segmentation (instance seg. desene abicim) models. It leverages for self-supervised models to discover objects without supervision and amplify it to train a sota localization model WITHOUT ANY HUMAN LABELS. CutLER first uses MaskCut to generate coarse masks for multiple objects in an image. And then, learns detector on these masks using their loss function. It can be applied to wide range of applications (data agnostic). CutLER >> ViT because good at multiple objects(salient) not just focusing prominent object. SOTA methods are (01/2023) FreeSOLO and MaskDistil but they need in domain unlabeled data. However, CutLER does not. Moreover, contains zero-shot detector. CutLER is solely trained on ImageNet. CutLER = Vit + MaskCut + Detector.

---

## FreeSOLO: Learning to Segment Objects without Annotations

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/2202.12181.pdf)
:::

Self-supervised instance segmentation method so without annotations. It generates class agnostic masks.

---
