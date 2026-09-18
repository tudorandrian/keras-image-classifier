# Evaluation report

- Split: `test`, 4050 images, 10 classes, 64 px
- Accuracy: **0.9491** (always answering the largest class would score 0.1111)
- Macro F1: **0.9473**
- Model: 99,450 parameters, best epoch 20 of 20, trained in 1356.6 s, seed 0
- Environment: Keras 3.15.1 on jax, Python 3.13.15, Windows-10-10.0.19045-SP0

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| AnnualCrop | 0.942 | 0.938 | 0.940 | 450 |
| Forest | 0.974 | 0.993 | 0.983 | 450 |
| HerbaceousVegetation | 0.910 | 0.924 | 0.917 | 450 |
| Highway | 0.945 | 0.923 | 0.934 | 375 |
| Industrial | 0.981 | 0.960 | 0.970 | 375 |
| Pasture | 0.928 | 0.943 | 0.936 | 300 |
| PermanentCrop | 0.912 | 0.907 | 0.909 | 375 |
| Residential | 0.978 | 0.991 | 0.985 | 450 |
| River | 0.916 | 0.899 | 0.907 | 375 |
| SeaLake | 0.991 | 0.993 | 0.992 | 450 |

![Confusion matrix](confusion_matrix.png)
