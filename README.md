
 
<h1> DepthDive: Enhanced Underwater Depth Estimation using Monocular Images</h1>

[Abhimanyu Bhowmik](https://github.com/abhimanyubhowmik), [Madhushree Sannigrahi](https://github.com/Madhushree2000), [Krittapat Onthuam](https://github.com/tamonmaru), 

[![report](https://img.shields.io/badge/Initial-Report-brightgreen)](https://doi.org/10.1109/BigData55660.2022.10020494) 
<!-- [![dataset](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/competitions/ieee-fraud-detection) 
[![dataset](https://img.shields.io/badge/Saved-Models-red)](https://github.com/abhimanyubhowmik/DBNex/tree/main/Data_and_Model) -->
[![slides](https://img.shields.io/badge/Presentation-Slides-yellow)](https://docs.google.com/presentation/d/1btUACelHTbZ4aX9lCH3yg1O-jxsZWn4l/edit?usp=sharing&ouid=103859519837437819731&rtpof=true&sd=true) 



> **Abstract:** *Accurate underwater depth estimation is vital for applications such as autonomous underwater vehicles, marine biology, and underwater archaeology. Traditional methods often rely on expensive and complex equipment, whereas monocular depth estimation offers a more cost-effective alternative. Despite significant advancements in terrestrial monocular depth estimation driven by deep learning, these models are inefficient in underwater environments due to challenges such as light attenuation, water turbidity, and data scarcity. This paper introduces DepthDive, a novel approach that adapts the Depth Anything Model (DAM) for underwater depth estimation using monocular images. The model is fine tuned via the parameter effcient fine tuning (PEFT), specifically low rank adaptation (LoRA). In addition, this work proposed a data sample filtering method to improve the quality of underwater depth dataset. Experimental results demonstrate that DepthDive significantly improves depth estimation accuracy in underwater environments, even with limited datasets, showcasing the potential of fine-tuning foundation models for specialized applications.*

<hr />

<h2>Analysis of Underwater Dataset</h2>

<h3>Context</h3>

We conducted an extensive survey to compile multiple small underwater datasets with reliable depth annotations. The table below provides a comprehensive comparison of various datasets, detailing their attributes such as camera type, size, image type, depth type, lighting conditions, depth range, and estimation methods.

<h3>Table</h3>

| Name | Camera | Size | Image Type | Depth Type | Lighting | Depth (m) | Estimation Method |
|------|--------|------|------------|-------------|----------|-----------|-------------------|
| SQUID [Berman et al., 2020](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html) | Stereo | 57 (Video) | Natural | Real (Metric) | Clear | 3-30 | AprilTags with size reference |
| Eiffel Tower [Boittiaux et al., 2023](https://www.seanoe.org/data/00810/92226/) | Mono | 18,082 | Natural | Real (Relative) | Dark | 1,700 | Structure-From-Motion (SFM) |
| NAREON [Dion´ısio et al., 2023](https://rdm.inesctec.pt/dataset/nis-2023-002) | Mono | 7,000 | Natural | Real (Relative) | Varying | 0.01 - 2.5 | Hybrid imaging system |
| FLSea VI [Randall et al., 2023](https://www.kaggle.com/datasets/viseaonlab/flsea-vi) | Mono | 22,451 | Natural | Real (Metric) | Varying | 0-12 | AprilTags with size reference |
| SeaThru [Akkaynak et al., 2019](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset) | Mono | 1,157 | Natural | Real (Metric) | Clear | 4-10 | Structure-From-Motion (SFM) |
| VAROS [Zwilgmeyer et al., 2021](https://zenodo.org/records/5567209) | Mono | 4,713 | Synthetic (Blender) | Real (Metric) | Dark | - | Information from Blender |
| ATLANTIS [Zhang et al., 2023](https://www.kaggle.com/datasets/zkawfanx/atlantis/data) | Mono | 3,200 | Synthetic (Generated) | Real (Relative) | Varying | - | Using MiDas |
| DRUVA [Varghese et al., 2023](https://github.com/nishavarghese15/DRUVA) | Mono | 20 (30 fps) | Natural | Generated (Relative) | Clear | 3-6 | Using USe-ReDI-Net |
| USOD 10k [Hong et al., 2023](https://github.com/LinHong-HIT/USOD10K) | Mono | 10,255 | Natural | Generated (Relative) | Varying | 5-60 | Using DPT |

<h4> Table Fields Description </h4>

- **Name**: The dataset name and reference.
- **Camera**: Type of camera used (Mono for Monocular, Stereo for Stereoscopic).
- **Size**: Number of images or videos in the dataset.
- **Image Type**: Whether the images are Natural or Synthetic.
- **Depth type**: Indicates if depth data is Real (either Metric or Relative) or Generated.
- **Lighting**: Describes the lighting conditions during image capture.
- **Depth (m)**: The range of depth values present in the dataset.
- **Estimation method**: Method used to estimate or generate depth information.

This table provides a quick reference for researchers and developers working with these datasets, allowing for easier comparison and selection based on specific project requirements.



<h2>Proposed Methodology</h2>


<!--
  ======================Global Architecture===========================
                          -->

<h3>Model Overview: Depth Anything Architecture</h3>

Utilising the same approach as MiDas, DepthAnything is a state-of-the-art monocular depth estimation model generally developed for general scene depth estimation. The model utilises both labelled and unlabelled datasets by adapting the teacher-student method. The teacher model learns the labelled dataset and predicts the pseudo-label of the unlabeled dataset. The student model is then able to learn from both datasets. The model excels in zero-shot depth estimation and is a potential baseline for underwater depth estimation. 


<br><br>

<div align="center">
<img src = "./Images/Main.png" width="100%">
<p>Overall view of the proposed model: DBNex </p>
</div>

<br>

<!--
  =========================Data Preprocessing========================= 
                          -->
 

<h3>Evaluation matrics</h3>


| Metric                            | Definition |
|-----------------------------------|------------|
| **Absolute Relative Error (AbsRel)** | \(\text{AbsRel} = \frac{1}{n} \sum_{i=1}^{n} \frac{\|d_i - \hat{d}_i\|}{d_i}\) |
| **Squared Relative Error (SqRel)** | \(\text{SqRel} = \frac{1}{n} \sum_{i=1}^{n} \frac{(d_i - \hat{d}_i)^2}{d_i}\) |
| **Root Mean Squared Error (RMSE)** | \(\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (d_i - \hat{d}_i)^2}\) |
| **Logarithmic RMSE**              | \(\text{RMSElog} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log d_i - \log \hat{d}_i)^2}\) |
| **Scale Invariant MSE in Log Scale (SiLog)** | \(\text{SiLog} = \sqrt{\mathbb{E}\left[ (\log(\hat{d}) - \log(d))^2 \right] - \left( \mathbb{E}\left[ \log(\hat{d}) - \log(d) \right] \right)^2}\) |
| **Peak Signal-to-Noise Ratio (PSNR)** | \(\text{PSNR} = 10 \log_{10} \left(\frac{\text{MAX}^2_d}{\sqrt{\text{MSE}}}\right)\) |
| **Structural Similarity Index (SSIM)** | \(\text{SSIM}(d, \hat{d}) = \frac{(2 \mu_d \mu_{\hat{d}} + C_1)(2 \sigma_{d \hat{d}} + C_2)}{(\mu_d^2 + \mu_{\hat{d}}^2 + C_1)(\sigma_d^2 + \sigma_{\hat{d}}^2 + C_2)}\) |
| **Pearson Correlation**           | \(\text{r} = \frac{\sum_{i=1}^{n} (d_i - \bar{d})(\hat{d}_i - \bar{\hat{d}})}{\sqrt{\sum_{i=1}^{n} (d_i - \bar{d})^2 \sum_{i=1}^{n} (\hat{d}_i - \bar{\hat{d}})^2}}\) |
| **$\delta_i$**                    | \(\delta_i = \text{percentage of } \left(\max\left(\frac{d}{\hat{d}}, \frac{\hat{d}}{d}\right) < 1.25^i\right)\) |

<!--
  ============================Model=========================
                          -->

<h3>Dataset Used </h3>


we analyzed almost all available underwater datasets to combine them for training our model. However, as shown in Table, many of these datasets were of poor quality. Benchmark datasets in the literature, such as SQUID and SeaThru , have unreliable depth maps with missing objects. These maps are typically generated using the Structure-from-Motion (SFM) technique, which often blurs distant objects and fails to capture the depth of moving objects. The most accurate ground truths are found in synthetically generated datasets like VAROS  and ATLANTIS. However, these synthetic datasets do not fully mimic real-world conditions, as they lack the presence of moving objects and the varying light and turbidity conditions found in actual underwater environments.

<h2> Data Sample Filtering </h2>

For real-world datasets, we often get inaccurate ground truth values. If we finetune our model on those data points, the model might learn biased distributions of depth maps. To avoid this, we developed a method, which can eliminate inaccurate ground truths, providing a better dataset for model fine-tuning. Our method includes converting RGB images to RMI input space, which takes into account underwater light characteristics of propagation. The red wavelength suffers more aggressive attenuation underwater, so the relative differences between {R} channel and {G, B} channel values can provide useful depth information for a given pixel. We take the maximum value of {B} and {G} channels and mask the pixels, which have zero depth values in the ground truth. The resultant images are given in figure.

<!--
  ====================================RESULTS===============================
                          -->

<h2>Results </h2>

<h3> Table 1: Performance Comparison of Depth Anything Model with VAROS Dataset </h3>


| | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SIlog ↓ | log10 ↓ | PSNR↑ | SSIM↑ | Person corr↑ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---|---------:|-------:|-------:|----------:|--------:|-------:|-----:|----:|-----------:|-----:|-----:|-----:|
| Without Training | 5.9228 | 52.6330 | 5.8734 | 1.8618 | 0.6001 | 0.7679 | 8.4031 | 0.7501 | 0.7093 | 0.0189 | 0.0404 | 0.0669 |
| 5 Epochs Training | 0.2336 | 0.2696 | 0.1581 | 0.2753 | 0.2195 | 0.0780 | 18.8912 | 0.9367 | 0.7885 | 0.7878 | 0.9220 | 0.9567 |


<h3> Table 2: Performance Comparison of Depth Anything Model with FlSeaVI Dataset </h3>


| | AbsRel | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SIlog ↓ | log10 ↓ | PSNR↑ | SSIM↑ | Person corr↑ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---|-------:|-------:|-------:|----------:|--------:|-------:|-----:|----:|-----------:|-----:|-----:|-----:|
| Without Training | 3.8750 | 36.1175 | 7.5801 | 1.4900 | 0.9353 | 0.5726 | 11.7313 | 0.7005 | -0.8213 | 0.0796 | 0.1608 | 0.2440 |
| 5 Epochs Training | 0.0762 | 0.4483 | 0.7114 | 0.3690 | 0.3629 | 0.0404 | 24.3683 | 0.9488 | 0.8658 | 0.9633 | 0.9753 | 0.9794 |


<h3> Table 3: Performance Comparison of Different Models on FLSeaVI and SeaThru Datasets</h3>


| Dataset | Model | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---------|-------|---------:|-------:|-------:|----------:|-----:|-----:|-----:|
| FLSeaVI | UW-Net\cite{gupta2019unsupervised} | 0.527 | 1.765 | 1.725 | 1.961 | 0.337 | 0.565 | 0.699 |
| FLSeaVI | Amitai et al\cite{amitai2023self} | 0.203 | 1.955 | 1.546 | **0.245** | 0.768 | 0.923 | 0.966 |
| FLSeaVI | **Ours** | **0.0762** | **0.4483** | **0.7114** | 0.3690 | **0.9633** | **0.9753** | **0.9794** |
| SeaThru (D3 and D5) | IDisc-KITTI\cite{piccinelli2023idisc} | 4.702 | 4.4288 | 5.891 | 1.192 | 0.093 | 0.241 | 0.359 |
| SeaThru (D3 and D5) | IDisc-Atlantis \cite{zhang2023atlantis} | 1.630 | 1.4279 | **1.371** | **0.354** | **0.553** | **0.850** | **0.955** |
| SeaThru (D3 and D5) | NewCRFs-KITTI\cite{yuan2022neural} | 2.874 | 1.5768 | 3.251 | 0.934 | 0.213 | 0.375 | 0.465 |
| SeaThru (D3 and D5) | NewCRFs-Atlantis \cite{zhang2023atlantis} | 1.683 | 1.4764 | 1.435 | 0.378 | 0.476 | 0.837 | 0.952 |
| SeaThru (D3 and D5) | **Ours** | **0.7925** | **0.9480** | 1.6575 | 0.8268 | 0.1797 | 0.4052 | 0.6128 |



<!-- <hr />

<h2>Cite our work</h2>

```bibtex

  @INPROCEEDINGS{10020494,
  author={Bhowmik, Abhimanyu and Sannigrahi, Madhushree and Chowdhury, Deepraj and Dwivedi, Ashutosh Dhar and Rao Mukkamala, Raghava},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)}, 
  title={DBNex: Deep Belief Network and Explainable AI based Financial Fraud Detection}, 
  year={2022},
  volume={},
  number={},
  pages={3033-3042},
  doi={10.1109/BigData55660.2022.10020494}}

```
<hr /> -->

<h2>Contact</h2>
For any queries, please contact: <a href="mailto:bhowmikabhimnayu@gmail.com">bhowmikabhimnayu@gmail.com</a>
