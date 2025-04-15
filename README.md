# Fine-Tuning WavLM for Text-Dependent Speaker Verification

This project demonstrates how to fine-tune the WavLM model for **text-dependent speaker verification (TDSV)** using the DeepMine dataset. It focuses on adapting a pretrained WavLM model to a TDSV task by training it with phrase-level speaker-labeled utterances. This is not a complete end-to-end speaker verification system, but a fine-tuning setup tailored for research and evaluation purposes.

The fine-tuned model is publicly available here: [aliyzd95/wavlm-deepmine-base-plus-sv](https://huggingface.co/aliyzd95/wavlm-deepmine-base-plus-sv)

## 🔍 Project Objective
- Fine-tune `wavlm-base-plus` specifically for **phrase-based speaker verification** (text-dependent).
- Adapt WavLM to handle speaker-specific variations across 10 predefined phrases.
- Evaluate speaker verification performance using EER (Equal Error Rate).

## 📂 Dataset: DeepMine
The DeepMine dataset is a Persian-English speech database collected through crowdsourcing. Users installed an Android application and recorded designated phrases in real-world environments across Iran. It includes both text-dependent and text-independent utterances.

- **Language**: Primarily Persian (Farsi), with English phrases
- **Collection Environment**: Real-life noise conditions
- **Utterance Types**:
  - 5 fixed Persian + 5 fixed English phrases
  - 
### 🔠 Phrase Mapping (Text-Dependent Samples)
```python
phrase_mapping = {
    "01": "صدای من نشان دهنده هویت من است.",
    "02": "صدای هر کس منحصر به فرد است.",
    "03": "هویت من را با صدای من تایید کن.",
    "04": "صدای من رمز عبور من است.",
    "05": "بنی آدم اعضای یکدیگرند.",
    "06": "My voice is my password.",
    "07": "OK Google.",
    "08": "Artificial intelligence is for real.",
    "09": "Actions speak louder than words.",
    "10": "There is no such thing as a free lunch.",
    "FT": "Free-Text"
}
```


## 🚀 Implementation Summary
- Framework: Hugging Face `transformers`, `datasets`, `torchaudio`, `librosa`
- Model: `WavLMForXVector` adapted for text-dependent verification
- Training: Hugging Face `Trainer` API
- Output: Cosine-similarity-based scoring for speaker verification

## 📉 Loss Function: Additive Margin Softmax (AM-Softmax)
**AM-Softmax** introduces a margin in cosine similarity space to improve speaker separation. It penalizes intra-class variation and rewards inter-class margins, making embeddings more discriminative — ideal for verification tasks.

## 📊 Evaluation Metric: Equal Error Rate (EER)
**EER** measures the operating point where the false acceptance rate (FAR) equals the false rejection rate (FRR). A lower EER indicates better system performance. We use cosine similarity between speaker embeddings for scoring.

## 🧠 Training Configuration
- Mixed precision (`fp16`) for speed and memory efficiency
- Gradient checkpointing + accumulation
- Cosine margin loss with adaptive scale
- Phrase-level batching and evaluation
---
Feel free to explore the notebook and pretrained model to build or adapt your own speaker verification research!
