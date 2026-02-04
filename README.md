# Serbian Speech-to-English Translation (SeamlessM4T)

Implementation of a **speech-to-text translation system**
that translates **Serbian audio directly into English text** using Meta's SeamlessM4T model.
The project focuses on **parameter-efficient fine-tuning with LoRA, model evaluation, and comparison**
for a low-resource speech translation scenario.

---

## Problem

Build a system that:
- takes **Serbian speech audio** as input
- generates **English text** as output
- avoids a cascaded ASR → MT pipeline
- uses **efficient fine-tuning** for limited compute resources
- works reliably on real-world news speech data

---

## Solution Overview

- Fine-tuned **SeamlessM4T-medium** using **LoRA (Low-Rank Adaptation)**
- Utilized a **high-quality parallel Serbian–English corpus** (Južne Vesti)
- Applied **8-bit quantization** for memory-efficient training
- Evaluated on a **held-out, domain-consistent test set**

---

## Data

### Training
- **Južne Vesti (train split)**  
→ Serbian news broadcasts with English translations

### Validation & Testing
- **Južne Vesti (validation & test splits only)**  
→ ensures fair evaluation on unseen, in-domain data

---

## Model & Training

- **Model**: `facebook/hf-seamless-m4t-medium`
- **Task**: Speech Translation (sr → en)
- **Framework**: HuggingFace Transformers + PEFT (LoRA)
- **Training setup**:
  - LoRA with r=16, alpha=32
  - 8-bit quantization
  - gradient accumulation
  - FP16 training
  - checkpointing and best-model selection

The model is explicitly configured for English output:
```python
model.generate(
    input_features=audio_features,
    tgt_lang="eng",
    generate_speech=False,
    max_new_tokens=225
)
```

---

## Evaluation & Testing

- **Validation Set (Južne Vesti validation split)**  
  - Used during training to monitor model performance and select the best epoch  
  - Metric used: **Word Error Rate (WER)**  
    → model checkpoint with lowest WER on validation set is saved as the best model

- **Test Set (Južne Vesti test split)**  
  - Used for final evaluation after training  
  - Metrics computed: **WER**, **BLEU**, **METEOR**  
    → provides a comprehensive assessment of transcription accuracy and translation quality

---

## Model Fine-tuning Comparison

We evaluated the impact of LoRA fine-tuning on SeamlessM4T-medium:

- **Models evaluated**:
  - `SeamlessM4T-medium` (baseline, zero-shot)
  - `SeamlessM4T-medium` (LoRA fine-tuned)

- **Comparison Metrics**:
  - **WER** – lower is better (transcription/translation accuracy)
  - **BLEU** – higher is better (translation quality)
  - **METEOR** – higher is better (semantic and lexical precision)

- **Findings**:
  - LoRA fine-tuning significantly improved performance compared to zero-shot baseline
  - Training with **<1% trainable parameters** achieved strong results
  - **8-bit quantization** enabled training on consumer GPUs (16GB VRAM)
  - This approach demonstrates the effectiveness of parameter-efficient fine-tuning for low-resource speech translation
