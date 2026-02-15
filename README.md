# RoD-TAL: A Benchmark for Answering Questions in Romanian Driving License Exams

[![Findings of EACL 2026](https://img.shields.io/badge/Findings%20of%20EACL-2026-blue)](https://2026.eacl.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.19666-b31b1b.svg)](https://arxiv.org/abs/2507.19666)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-RoD--TAL-yellow)](https://huggingface.co/datasets/unstpb-nlp/RoD-TAL)
[![Model](https://img.shields.io/badge/🤗%20Model-Embedding-orange)](https://huggingface.co/unstpb-nlp/multilingual-e5-small-RoD-TAL)

## Abstract

The intersection of AI and legal systems presents a growing need for tools that support legal education, particularly in under-resourced languages such as Romanian. In this work, we aim to evaluate the capabilities of Large Language Models (LLMs) and Vision-Language Models (VLMs) in understanding and reasoning about the Romanian driving law through textual and visual question-answering tasks. To facilitate this, we introduce RoD-TAL, a novel multimodal dataset comprising Romanian driving test questions, text-based and image-based, along with annotated legal references and explanations written by human experts. We implement and assess retrieval-augmented generation (RAG) pipelines, dense retrievers, and reasoning-optimized models across tasks, including Information Retrieval (IR), Question Answering (QA), Visual IR, and Visual QA. Our experiments demonstrate that domain-specific fine-tuning significantly enhances retrieval performance. At the same time, chain-of-thought prompting and specialized reasoning models improve QA accuracy, surpassing the minimum passing grades required for driving exams. We highlight the potential and limitations of applying LLMs and VLMs to legal education. We release the code and resources through the GitHub repository.

## Overview

RoD-TAL is a comprehensive benchmark designed to evaluate AI systems on Romanian driving license exam questions. The dataset includes:

- Multimodal Questions: Both text-based and image-based questions from Romanian driving exams
- Legal References: Annotated references to relevant Romanian traffic laws
- Expert Explanations: Human-written explanations for correct answers
- Multiple Tasks: IR, QA, Visual IR, and Visual QA evaluation tasks

## Getting Started

### Prerequisites

* Python 3.10+
* Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vladman-25/RoD-TAL.git
cd RoD-TAL
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Access

The RoD-TAL dataset is available on HuggingFace:
- **Dataset**: [unstpb-nlp/RoD-TAL](https://huggingface.co/datasets/unstpb-nlp/RoD-TAL)
- **Fine-tuned Embedding Model**: [unstpb-nlp/multilingual-e5-small-RoD-TAL](https://huggingface.co/unstpb-nlp/multilingual-e5-small-RoD-TAL)

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("unstpb-nlp/RoD-TAL")

# Load the embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("unstpb-nlp/multilingual-e5-small-RoD-TAL")
```

## Experiments

The repository contains Jupyter notebooks for reproducing all experiments from the paper:

### 1. Corpus Scraping
[`1_corpus_scrape.ipynb`](code/1_corpus_scrape.ipynb) - Downloads Romanian traffic-law documents from official sources, parses article-level text, cleans content, and exports structured corpus files.

### 2. Information Retrieval (IR)
[`5_1_experiments_ir.ipynb`](code/5_1_experiments_ir.ipynb) - Benchmarks text IR strategies (dense retrieval, reranking, query rewriting, and fine-tuned embeddings) using Recall/Precision/NDCG.

### 3. Question Answering (QA)
[`5_2_experiments_qa.ipynb`](code/5_2_experiments_qa.ipynb) - Evaluates LLM QA performance under no-RAG, retrieved-RAG, and ideal-RAG setups with Exact Match, Precision, Recall, and F1.

### 4. Visual Information Retrieval (VIR)
[`5_3_experiments_vir.ipynb`](code/5_3_experiments_vir.ipynb) - Evaluates visual retrieval strategies (question/caption/image-based reformulations) for legal-article and indicator retrieval with IR metrics.

### 5. Visual Question Answering (VQA)
[`5_4_experiments_vqa.ipynb`](code/5_4_experiments_vqa.ipynb) - Benchmarks multimodal QA strategies with o4-mini, varying image/caption usage and legal context (retrieved vs ideal).

### 6. Analysis I
[`6_analysis_1.ipynb`](code/6_analysis_1.ipynb) - Performs dataset profiling and per-category analysis across IR/QA/VIR/VQA, including targeted citation/hallucination diagnostics.

### 7. Analysis II
[`6_analysis_2.ipynb`](code/6_analysis_2.ipynb) - Conducts post-hoc error analysis of QA/VQA outputs, with emphasis on legal citation quality and reasoning behavior.

### 8. LLM Judge
[`7_judge.ipynb`](code/7_judge.ipynb) - Runs an LLM-as-judge pipeline for incorrect QA predictions, producing structured XML verdicts and cluster-based error analysis.

## Citation

If you use RoD-TAL in your research, please cite our paper:

```bibtex
@misc{man2025rodtalbenchmarkansweringquestions,
      title={RoD-TAL: A Benchmark for Answering Questions in Romanian Driving License Exams}, 
      author={Andrei Vlad Man and Răzvan-Alexandru Smădu and Cristian-George Craciun and Dumitru-Clementin Cercel and Florin Pop and Mihaela-Claudia Cercel},
      year={2025},
      eprint={2507.19666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19666}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.