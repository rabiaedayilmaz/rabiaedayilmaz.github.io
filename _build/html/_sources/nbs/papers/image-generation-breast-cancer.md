# Image Generation - Breast Cancer - Papers

## BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix

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