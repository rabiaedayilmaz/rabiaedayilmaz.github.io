# GANs in Medical Image Synthesis

## Generative Adversarial Networks in Medical Image augmentation: A review

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S0010482522001743)
:::

They classified GANs to three categories: random latent vectors, image translation, classical transformations.

* Random latent vector: a latent vector is a low dimensional representation of an image. During augmentation phase, the GAN randomly samples random latent vectors within this distribution and obtains new images through the generator. Types: without external conditions, conditional, local conditional.

* Image Translation: To change modality of medical image. Supervised or unsupuervised.

* Classical transformation: To optimize and combine classical transformations.

There are several commonly used loss functions:

* Lpix = pixel-level supervision loss | difference in pixel-wise intensity values between synthetic and real images

* Lcyc = cycle consistency loss | pixel-wise intensity values between reconstructed and real images

* Ladv = adversarial loss | the dicriminative difference between the real or synthetic image

* Lfeat = image feature loss | differences between real or synthetic image features of the same type of image

* Ldist = image distribution loss | differences in data distribution between real or sythetic datasets

* Llab = label consistency loss | difference between the labels of the synthetic image and true labels

Commonly used evaluation metrics:

* Epix = pixel level accuracy eval. | requires mono-modal real images as ground truth for synthetic images

* Eset = dataset distribution overlap eval. | evaluate the overall fidelity of the synthetic dataset

* Edoc = radiologist ratings eval. | quantification of physicist

* Edown = indirect eval. of downstream task performance

* GANs in Medical Field: Brain -> Tumor contour segmentation/registration /classifcaiton - cGAN.

* GANs in Medical Field: Breast -> For classification NcGAN, cGAN, and local-cGAN.

---

## Data augmentation using generative adversarial networks (CycleGAN) to improve generalizability in CT segmentation tasks

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.nature.com/articles/s41598-019-52737-x)
:::

Medical data scarce and expensive to generate. cGAN is used to tranform contrast CT into non contrast CT. Then trained model is used to augment synthetic non-contrast CT images. dice score. Performance increased.

---

## Medical Image Synthetic Data Augmentation Using GAN

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://dl.acm.org/doi/10.1145/3424978.3425118)
:::

Medical image datasets are unbalanced and limited.Thus, deep learning model is prone to underfit or overfit. This paper used a synthetic data augmentation method based on GANs. Experimental results show that this method produces better performance than existing methods. PG-ACGAN (progressive growing auxiliary classifier gan) is proposed (based on ACGAN). It improves diversity and quality of generated samples. It has two parts: ACGAN (enables to specify the type of generated sample with label) and PCGAN (enables to learn the pixel features better with limited data). Accroding to SSIM (structural similarity) metric, ACGAN is better tho. 

---

## A Data Augmentation Strategy Combining a Modified pix2pix Model and the Copy-Paste Operator for Solid Waste Detection With Remote Sensing Images

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9904838)
:::

pix2pix model and copy paste method are integrated. For pix2pix, local-global discriminator (LGD) is desgined to ensure that generated images had detailed info and consistent color. It has better performance than bare pix2pix.

---

## Automated scoring of CerbB2/HER2 receptors using histogram based analysis of immunohistochemistry breast cancer tissue images

::::{grid}
:gutter: 1

:::{grid-item-card} To Read
[Paper](https://www.sciencedirect.com/science/article/pii/S1746809421005218?via%3Dihub)
:::

In treatment of cancer, chemotherapy is frequently used but damages a lot. Instead new method - targeted therapy. By IHC, increased CerbB2 protein on the cell membrane is demonstrated by using antibodies against this protein. Four scores 0, 1, 2 and 3. CerbB2 protein targete therapy is suitable for just 3. 

ITU MED datasets are introduced (helal be). They are the only publicly available CerbB2/HER2 scoring dataset(valhala öyle). Dataset creation: image acquisition, color deconvolution, hybrid cell detection module(cell nuclei and membrane detection), membrane histogram extraction, membrane intensity vectors, cell identification aaand tissue scoring.

Overall, here, a tissue scoring system for CerbB2/HER2 receptors in breast cancer tissue is proposed and published likely-public dataset (sent request hopefully i will receive it).