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

## Image-to-Image Translation with Conditional Adversarial Networks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://arxiv.org/pdf/1611.07004.pdf)
:::

In this paper, a general solution for image-to-image translation by using conditional adversarial networks is investigated and pix2pix model is proposed. Moreover, not only learning the mapping from input image to target image but also learning a loss function to train this mapping. This paper shows that there is no-need no longer for hand-engineer own mappings.