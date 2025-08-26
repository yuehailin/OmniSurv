# OmniSurv
## Bipartite Patient-Modality Graph Learning with Event-Conditional Modelling of Censoring for Cancer Survival Prediction
Accurately predicting the survival of cancer patients is crucial for personalized treatment. However, existing studies focus solely on the relationships between samples with known survival risks, ignoring the value of censored samples, which is inevitable in clinical practice. Furthermore, these studies may suffer performance degradation in modality-missing scenarios and even struggle during the inference process. In this study, we propose OmniSurv, a multimodal method for pan-cancer survival prediction. Specifically, to alleviate performance degradation in modality-missing scenarios, we design a bipartite graph to simulate the patient-modality relationship in various modality-missing scenarios and leverage a complete-missing alignment strategy to explore modality-agnostic features. Then, we design a plug-and-play event-conditional modeling of censoring (ECMC) that selects reliable censored data using dynamic momentum accumulation confidences, assigns more accurate survival times to these censored data, and incorporates them as uncensored data into training. Finally, we design a gradient-based adaptive loss weighting strategy that can dynamically adjust the weights of different loss components, ensuring their sufficient optimization. Comprehensive evaluations on 7 publicly cancer datasets (e.g., breast cancer, brain tumor, etc.) show that OmniSurv outperforms the best-performing baseline by 2.9% in mean C-index and exhibits excellent robustness under various modality-missing scenarios. With the plug-and-play ECMC module, 9 baselines realize a 1.2% average C-index gain on 7 datasets. Moreover, OmniSurv also offers explainable predictions by allowing clinicians to trace decision pathways and understand how different modalities contribute to survival prediction.

<img width="1111" height="832" alt="image" src="https://github.com/user-attachments/assets/976bb181-dc33-4d8e-9019-44ecca2bf51e" />



## 1. Data Acquisition
Pathological slide and clinical records are available at https://portal.gdc.cancer.gov/. Genomic profile is available at https://www.cbioportal.org/.
The extracted features of the clinical records, pathology and genetic will be uploaded as soon as possible.


## 2. Train
```
train-for-graph.py
```

## Acknowledge
This project is based on [HGCN](https://github.com/lin-lcx/HGCN) and [MUSE+](https://github.com/zzachw/MUSE). We have great thanks for these awesome projects.



