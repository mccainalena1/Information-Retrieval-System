# Information Retrieval System

This repository contains an **Information Retrieval (IR) System** implemented as part of a graduate-level course project.  
The system is designed to index and retrieve documents from the **Cranfield dataset**, evaluate retrieval effectiveness, and provide experimental results.

---

## 📖 Project Overview

The Cranfield collection is a widely used benchmark dataset for IR evaluation.  
This project implements indexing, ranking, and query processing, and reports performance using standard IR evaluation metrics.  

- **Dataset**: Cranfield collection  
- **Core functionality**: Indexing, retrieval, and evaluation  
- **Output**: Ranked retrieval results, figures, and a formal evaluation report  

For details of the assignment, see [docs/project1-2023_v3.pdf](docs/project1-2023_v3.pdf).  
For results and analysis, see [results/Evaluation.pdf](results/Evaluation.pdf).  

---

## 📂 Repository Structure
```
.
├── src/
│   ├── assignment1.py          # main retrieval pipeline
│   └── Untitled.ipynb          # exploratory notebook
├── data/
│   └── CranfieldDataset/       # Cranfield corpus
├── results/
│   ├── Evaluation.pdf          # final writeup
│   ├── Evaluation.docx
│   └── Figures/                # evaluation plots & screenshots
├── docs/
│   └── project1-2023_v3.pdf    # assignment description
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/mccainalena1/Information-Retrieval-System.git
cd Information-Retrieval-System

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the retrieval system:

```bash
python src/assignment1.py
```

- Ensure that the **Cranfield dataset** is located in `data/CranfieldDataset/`.
- Output figures and results are generated in the `results/` directory.

---

## 📊 Results

- Final evaluation report: [`results/Evaluation.pdf`](results/Evaluation.pdf)  
- Supplementary figures: [`results/Figures/`](results/Figures/)  

The system demonstrates indexing and retrieval effectiveness using the Cranfield benchmark.  
Performance metrics and observations are discussed in detail in the evaluation report.  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
© 2025 Alena McCain
