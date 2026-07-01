HealthSenseTS/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│
├── src/
│   ├── datasets/
│   │   └── wesad_loader.py
│   │
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   └── windowing.py
│   │
│   ├── models/
│   │   ├── lstm.py
│   │   ├── cnn1d.py
│   │   ├── transformer.py
│   │   └── autoencoder.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── checkpoints/
│
├── requirements.txt
└── README.md

project/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── configs/
├── outputs/
├── requirements.txt
└── README.md