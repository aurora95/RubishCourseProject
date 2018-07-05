# Plant Seedlings Classification

[Plant Seedlings Classification contest on Kaggle](https://www.kaggle.com/c/plant-seedlings-classification)
Determine the species of a seedling from an image
![](images/raw.png)
There are 12 classes in total:
![](images/classes.png)

|Author|Student number|
|------|---------|
|邢亚杰|N/A|
|李瑞麟|N/A|
| 王尧 |N/A|

## Collaboration
|file_name|collaborator|
|---------|------------|
|readme.md|王尧|
|basemodel.py|邢亚杰|
|download.sh|李瑞麟|
|model.py|邢亚杰|
|PSDataset.py|李瑞麟|
|submission.py|王尧|
|train.py|邢亚杰|
|utils/logger.py|王尧|
|utils/metrics.py|邢亚杰|
|utils/progress_bar.py|邢亚杰|

## Code map

```
$Root
│
│─ readme.md —— this file
│
|─ basemodel.py —— basemodel definition
│
|─ download.sh —— dataset download shell script
│
|─ model.py —— model definition
│
|─ PSDataset.py —— Dataset preprocessor
│   
|─ submission.py —— contest submission script
│   
|─ train.py —— trainning script
│   
└─ utils —— useful tools
   │
   |─ logger.py —— log printing and saving tool
   │
   |─ metrics.py —— statistic tool
   │
   └─ progress_bar.py —— progress bar printing tool
```

## Result

![](images/loss.png)
![](images/acc.png)