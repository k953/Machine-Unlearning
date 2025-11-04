# LSTM_Unlearning

Machine-Unlearning

Short: Repository contains experiments and reference code for Machine Unlearning (BERT + LSTM examples). Purpose: demonstrate how to make NLP models “forget” specific data subsets (Df) and measure forgetting with KL-divergence / evaluation.

Repo overview (files)

README.md — (this file) instructions & summary.

bert_poisoned_to_forget.py — script / notebook for fine-tuning BERT (clean → poison → test).

kga_unlearner_test.py — main experiment pipeline that trains multiple BERT variants (D, Dn, Dr, Df), compares logits and computes KL divergence; includes starting point for unlearning (copy + modify).

story_rains_lstm.py — LSTM-based text-generation baseline (story generation) used as an alternate demo for unlearning on RNN models.

lecture22/ — example lecture files (not needed to run experiments).

articles_data_D.txt, new_data_Dn.txt, remaining_data_Dr.txt, to_be_forgotten_Df.txt (expected in Drive / local data folder) — data partitions used by scripts. (Not included in repo; add your own.)

Note: Filenames in your code refer to Google Drive paths (used originally in Colab). See Run / Setup below.

What this project demonstrates (plain)

Train a base BERT on dataset D (all data).

Train separate models on subsets:

Df (data to be forgotten),

Dr (remaining data — i.e., D without Df),

Dn (new/incremental data).

Compute and compare model outputs (logits / probability distributions) on test prompts.

Use KL divergence as a metric to quantify how much the base model differs from the clean model (Dr) — higher KL means more influence of Df.

Show/simple prototype methods to unlearn (reverse gradients, finetune on clean data, or KL minimization) and measure the dropping KL.

Why it matters

Machine unlearning relates directly to privacy requirements such as GDPR — Right to be Forgotten.

Goal: remove sensitive data influence from models without full retraining from scratch (saves time & compute).

Quick start (Colab / local) — stepwise

These steps assume you will run in Google Colab (recommended GPU) or local machine with Python + PyTorch. Replace Drive paths with local paths if needed.

1) Prerequisites

Python 3.8+

GPU recommended (for BERT finetuning)

Install required packages:

pip install -U pip
pip install accelerate transformers[torch] torch


If running in Colab you can put the pip lines in a cell.

2) Prepare your data

Place these text files (plain .txt) in your Drive or local repo folder:

articles_data_D.txt — full training data (D)

new_data_Dn.txt — new incremental samples (Dn)

remaining_data_Dr.txt — D without Df (Dr)

to_be_forgotten_Df.txt — samples you want the model to forget (Df)

Format: plain text; cleaning (lowercase / remove HTML / punctuation) is included in scripts — but ensure files are UTF-8 and reasonably sized for finetuning.

3) Mount Google Drive (Colab)

If using Colab, mount Drive so scripts can read/write:

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
# Then put your files in: /content/gdrive/My Drive/

4) Run the baseline training (BERT on D)

Open kga_unlearner_test.py or the notebook version and run the section that trains bert_finetuned_on_D. This will:

Clean articles_data_D.txt

Create a TextDataset and DataCollatorForLanguageModeling

Fine-tune bert-base-uncased for num_train_epochs (default 2; change as needed)

Save model to ./bert_finetuned_on_D and optionally to Drive

Tip: For faster iteration use fewer epochs / smaller block_size. For real experiments use more epochs and GPU.

5) Train the other variants (Dn, Dr, Df)

Run the corresponding sections in kga_unlearner_test.py to train:

bert_finetuned_on_Dn from new_data_Dn.txt

bert_finetuned_on_Dr from remaining_data_Dr.txt

bert_finetuned_on_Df from to_be_forgotten_Df.txt

Each produces a saved model directory.

6) Load models & evaluate

The script loads all saved model versions and runs a test prompt (e.g., text with [MASK]). It computes logits and probability distributions and then calculates KL divergence:

from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as F

# get outputs (logits) for a tokens_tensor
outputs = model(tokens_tensor)
logits = outputs.logits
probs = F.softmax(logits, dim=-1)

# compute KL as done in script:
loss = F.kl_div(input=logits_A, target=logits_B, log_target=True, reduction='batchmean')


Interpretation:

KL(D, Dr) high → D still contains Df influence (need unlearning).

KL(D, Df) low → D similar to Df (bad, influence strong).

After unlearning, KL(Unlearned, Dr) should be low.

7) Apply an unlearning method (prototype)

You can implement at least one of:

Retrain-from-scratch (baseline): retrain on Dr only (expensive but gold standard).

Fine-tune on Dr: keep original, run a few epochs on Dr to wash out Df.

Gradient-reversal on Df: run training on Df but subtract updates (reverse sign) to remove Df influence (prototype shown in comments).

KL-minimization: fine-tune the copied model to minimize KL between it and the Dr model (optimize logits similarity).

The repo includes a copy-based approach: out_put_unlearned_model = copy.deepcopy(bert_finetuned_on_D) — modify this copy and apply the chosen unlearning update loop.

Example commands (Colab cell / terminal)
# install
pip install accelerate transformers[torch] torch

# run (if script is executable)
python3 kga_unlearner_test.py


(In Colab, run the notebook cells in order instead.)

Evaluation suggestions

Use KL divergence between logits distributions for the same masked prompt(s).

Use multiple prompts (diverse contexts) to robustly test forgetting.

Track: KL_before, KL_after, perplexity on Dr and Df, top-k predictions for the mask word before/after.

Plot KL vs unlearning steps (epoch / iterations) to show convergence.

Interview-ready short lines (copy-paste)

“My project implements machine unlearning for BERT — I partition training data into D (all), Df (to forget), and Dr (remaining). I measure model drift using KL divergence and implement a gradient-reversal style update to remove Df influence without full retraining.”

“This is useful for GDPR ‘right to be forgotten’ compliance because it allows selective removal of user data effects from models.”

Tips & Caveats

Data: Do not commit private or sensitive files to GitHub. Use Drive or local private storage.

Compute: BERT finetuning needs GPU for practical speed.

Reproducibility: Fix random seeds, save tokenizers + model checkpoints, and log hyperparameters.

Ethics: Validate that unlearning does not harm model performance on retained data (Dr).

Suggested next steps (for the repo)

Add a small synthetic demo dataset and a one-cell Colab demo that runs full pipeline (train small BERT-like or DistilBERT on tiny data) so users can reproduce quickly.

Add visualization: KL vs iterations plot, top-5 predicted tokens before/after unlearning.

Add requirements.txt for easy installs.

Add a LICENSE (e.g., MIT) if you want to open-source.

Minimal requirements.txt (put in repo)
transformers>=4.0.0
torch
accelerate
regex
