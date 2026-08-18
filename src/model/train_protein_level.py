"""
Train the no-abundance ablation: protein_emb = ESM2 only (640-dim), text_emb
= whichever biobert_embeddings*.h5 variant --biobert-filename points at
(plain UniProt annotation, gene-name-masked, and/or GO-term-augmented — see
embed_biobert.py). No patient-level anything — abundance and clinical
description are both gone, so there is no patient-varying signal left at all.
Proteins are deduplicated globally across cohorts (src/model/dataset.py) and split
train/validation/test at the protein level (data/protein_splits.json), not
the patient level (see CLAUDE.md's "no-abundance ablation" notes for why).

BioBERT-unfreezing experiment (--unfreeze-biobert-layers): ESM2 always stays
fully frozen. --unfreeze-biobert-layers 0 (default) is the original,
fully-frozen-BioBERT baseline, unchanged -- text still comes straight from
the precomputed --biobert-filename .h5 file, no BioBERT forward pass at
training time at all. --unfreeze-biobert-layers N > 0 unfreezes the last N
BioBERT Transformer layers (total layer count read off the model itself,
never hard-coded) and switches the text side to a live tokenizer -> BioBERT
-> mean-pool forward pass so gradients can reach them, using the exact same
text a matching --biobert-filename would have been embedded from (inferred
from its naming convention -- see parse_biobert_text_flags in dataset.py).
BioBERT's unfrozen layers get their own optimizer param group at
--biobert-lr, separate from the projection heads' --lr.

Mixed precision (bf16 autocast): the live BioBERT forward/backward pass
(--unfreeze-biobert-layers > 0 only) runs under torch.autocast(dtype=
torch.bfloat16) on CUDA -- roughly half the compute cost of fp32 on Ampere+
tensor cores, no loss-scaler needed (bf16 keeps fp32's exponent range, just
fewer mantissa bits, so it doesn't need the numerical-stability workarounds
fp16 does). Weights and the optimizer step stay fp32 throughout; only the
forward/backward math for this pass runs in bf16. The frozen-BioBERT
baseline (--unfreeze-biobert-layers 0) never touches autocast at all --
identical to its original behavior.

There is no --cohorts flag: this script always trains on the full,
homology-aware-split human proteome (data/full_proteome/, see CLAUDE.md
section 11) -- the old per-cohort CPTAC concept doesn't apply here at all
(that's train.py / section 1-10, a different, ablation-baseline script).

Usage:
    uv run python src/model/train_protein_level.py \\
        --checkpoint data/runs/.../checkpoint_full_proteome_masked.pt \\
        --epochs 2000 \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --splits-file data/protein_splits_full_proteome.json
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModel
from src.model.dataset import ProteomicsDataset, parse_biobert_text_flags
from src.model.projection_head import ProjectionHead
from src.embeddings.textual.embed_biobert import MODEL_NAME, MAX_LENGTH, mean_pool

BATCH_SIZE = 512
MAX_LR = 1e-5
DEFAULT_BIOBERT_LR = 1e-5
MIN_RECOMMENDED_STEPS = 2000
SEED = 42


def infonce_loss(protein_vec, text_vec):
    logits = (protein_vec @ text_vec.T) * (1 / 0.07)
    labels = torch.arange(len(protein_vec), device=protein_vec.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def in_batch_recall_at_k(logits, ks=(1, 10)):
    """
    Cheap validation-only retrieval diagnostic reusing the similarity matrix
    already computed for the InfoNCE val loss -- ranks each row's diagonal
    (true match) against the other candidates in the same validation batch.
    Not a substitute for the project's real, closed-pool protein-level R@K
    (src/model/recall_at_k_protein_level.py) -- just a free, no-extra-pass
    signal for comparing unfreeze configurations during this experiment.
    """
    labels = torch.arange(logits.size(0), device=logits.device)
    ranks = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).float().argmax(dim=1)
    return {k: (ranks < k).float().mean().item() for k in ks}


def setup_biobert_encoder(device, unfreeze_last_n_layers: int):
    """
    Loads BioBERT, freezes everything, then unfreezes (requires_grad=True)
    only the last `unfreeze_last_n_layers` Transformer encoder layers.
    Frozen layers still run in the forward pass as normal -- only their
    requires_grad flag changes, nothing is skipped or wrapped in no_grad.
    Total layer count is read off the model itself (model.encoder.layer),
    never hard-coded, so this works for any BERT-family checkpoint.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.encoder.layer)
    unfrozen_indices = list(range(n_layers - unfreeze_last_n_layers, n_layers)) \
        if unfreeze_last_n_layers > 0 else []
    for idx in unfrozen_indices:
        for p in model.encoder.layer[idx].parameters():
            p.requires_grad = True

    return tokenizer, model, n_layers, unfrozen_indices


def encode_text_live(texts, tokenizer, model, device):
    """Tokenize + forward pass + mean-pool, identical to embed_biobert.py's
    embed_batch, except gradients are allowed to flow (no torch.no_grad())
    so the unfrozen layers can be trained."""
    encoded = tokenizer(list(texts), padding=True, truncation=True,
                         max_length=MAX_LENGTH, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    return mean_pool(outputs.last_hidden_state, encoded["attention_mask"])


def train(checkpoint_path: str, epochs: int,
          biobert_filename: str = "biobert_embeddings.h5",
          splits_file: str = "data/protein_splits.json",
          lr: float = MAX_LR, batch_size: int = BATCH_SIZE, hidden_dim: int = 512,
          output_dim: int = 256, on_epoch_end=None, early_stopping_patience: int | None = None,
          unfreeze_biobert_layers: int = 0, biobert_lr: float = DEFAULT_BIOBERT_LR):
    """
    No `cohorts` parameter -- this script always trains on the full,
    homology-aware-split proteome (ProteomicsDataset(use_abundance=False)
    always reads from data/full_proteome/, see dataset.py). The old
    per-cohort CPTAC concept only applies to the with-abundance ablation
    baseline (train.py, section 1-10), a separate script.
    on_epoch_end: optional callback(epoch, train_loss, val_loss) -> None,
    called after every epoch. Lets a caller (e.g. sweep_train.py) hook in
    external logging without this script needing to know that caller exists.
    early_stopping_patience: optional int. If set, stop once val_loss hasn't
    improved for this many consecutive epochs. Default None preserves the
    original always-run-all-epochs behavior for existing callers.
    unfreeze_biobert_layers: number of BioBERT Transformer layers to
    unfreeze, counted from the END (last N layers). Default 0 reproduces
    the original, fully-frozen baseline exactly -- text still comes from
    the precomputed biobert_filename .h5 file, no live BioBERT forward
    pass at all. N > 0 switches the text side to live encoding (raw text
    -> tokenizer -> BioBERT -> mean-pool), required so gradients can reach
    the unfrozen layers; ESM2 and the protein side are completely
    unaffected either way. biobert_lr: learning rate for BioBERT's
    unfrozen parameters (separate optimizer param group from the
    projection heads, which keep using `lr`); unused when
    unfreeze_biobert_layers == 0.
    Returns (best_checkpoint_path, best_val_loss).
    """
    # Fixed seed so repeated runs (e.g. comparing different --epochs values)
    # only differ because of epochs, not because of a different random weight
    # initialization or batch shuffle order each time.
    torch.manual_seed(SEED)

    with open(splits_file) as f:
        protein_splits = json.load(f)
    val_protein_ids = set(protein_splits["validation"])
    held_out_protein_ids = val_protein_ids | set(protein_splits["test"])

    live_text = unfreeze_biobert_layers > 0
    text_flags = parse_biobert_text_flags(biobert_filename) if live_text else None

    dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False, held_out_protein_ids=held_out_protein_ids,
        biobert_filename=biobert_filename, live_text=live_text, text_flags=text_flags,
    )
    print(f"Training data: data/full_proteome/  |  Training pairs (unique proteins): {len(dataset):,}")
    print(f"BioBERT source: {biobert_filename}  |  lr: {lr}  |  batch_size: {batch_size}  |  "
          f"hidden_dim: {hidden_dim}  |  output_dim: {output_dim}")

    shuffle_generator = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=shuffle_generator)

    val_dataset = ProteomicsDataset(
        cohorts=[], use_abundance=False, restrict_to_protein_ids=val_protein_ids,
        biobert_filename=biobert_filename, live_text=live_text, text_flags=text_flags,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation pairs (proteins never trained on): {len(val_dataset):,}")

    total_steps = epochs * len(dataloader)
    if total_steps < MIN_RECOMMENDED_STEPS:
        print(f"[warning] only {total_steps:,} total optimizer steps "
              f"({len(dataloader)} steps/epoch x {epochs} epochs) — "
              f"consider a larger --epochs value")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mixed precision only helps (and is only enabled for) the live BioBERT
    # forward/backward pass -- the frozen-BioBERT baseline never touches it.
    use_amp = live_text and device.type == "cuda"
    if live_text:
        print(f"[biobert_unfreeze_{unfreeze_biobert_layers}] bf16 autocast for live BioBERT fwd/bwd: {use_amp}")
    protein_head = ProjectionHead(input_dim=640, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    text_head = ProjectionHead(input_dim=768, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    run_name = f"biobert_unfreeze_{unfreeze_biobert_layers}"
    biobert_tokenizer = biobert_model = None
    biobert_n_layers = unfrozen_indices = None
    param_groups = [{"params": list(protein_head.parameters()) + list(text_head.parameters()), "lr": lr}]

    n_biobert_trainable = 0
    if live_text:
        biobert_tokenizer, biobert_model, biobert_n_layers, unfrozen_indices = setup_biobert_encoder(
            device, unfreeze_biobert_layers
        )
        biobert_trainable_params = [p for idx in unfrozen_indices for p in biobert_model.encoder.layer[idx].parameters()]
        n_biobert_trainable = sum(p.numel() for p in biobert_trainable_params)
        param_groups.append({"params": biobert_trainable_params, "lr": biobert_lr})

    n_heads_trainable = sum(p.numel() for p in param_groups[0]["params"] if p.requires_grad)
    print(f"[{run_name}] BioBERT layers: {biobert_n_layers if live_text else 'n/a (frozen, precomputed)'}  |  "
          f"unfrozen layer indices: {unfrozen_indices if live_text else []}  |  "
          f"biobert_lr: {biobert_lr if live_text else 'n/a'}  |  projection-head lr: {lr}")
    print(f"[{run_name}] trainable params -- projection heads: {n_heads_trainable:,}  |  "
          f"BioBERT: {n_biobert_trainable:,}  |  total: {n_heads_trainable + n_biobert_trainable:,}")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    def compute_text_vec(text_batch):
        if live_text:
            return text_head(encode_text_live(text_batch, biobert_tokenizer, biobert_model, device))
        return text_head(text_batch.to(device))

    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        protein_head.train()
        text_head.train()
        if live_text:
            biobert_model.train()
        total_loss = 0
        for protein_emb, text_batch in dataloader:
            protein_emb = protein_emb.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                protein_vec = protein_head(protein_emb)
                text_vec = compute_text_vec(text_batch)
                loss = infonce_loss(protein_vec, text_vec)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        protein_head.eval()
        text_head.eval()
        if live_text:
            biobert_model.eval()
        val_loss = 0
        val_recall = {1: 0.0, 10: 0.0}
        with torch.no_grad():
            for protein_emb, text_batch in val_dataloader:
                protein_emb = protein_emb.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    protein_vec = protein_head(protein_emb)
                    text_vec = compute_text_vec(text_batch)
                    logits = (protein_vec @ text_vec.T) * (1 / 0.07)
                    batch_val_loss = infonce_loss(protein_vec, text_vec)
                val_loss += batch_val_loss.item()
                batch_recall = in_batch_recall_at_k(logits)
                for k in val_recall:
                    val_recall[k] += batch_recall[k]
        val_loss /= len(val_dataloader)
        for k in val_recall:
            val_recall[k] /= len(val_dataloader)

        is_best = val_loss < best_val_loss
        print(f"[{run_name}] Epoch {epoch + 1}/{epochs}  "
              f"loss: {total_loss / len(dataloader):.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"val_R@1: {val_recall[1]:.4f}  val_R@10: {val_recall[10]:.4f}"
              + ("  <- best so far" if is_best else ""))

        if on_epoch_end is not None:
            on_epoch_end(epoch, total_loss / len(dataloader), val_loss)

        checkpoint = {"protein_head": protein_head.state_dict(),
                      "text_head": text_head.state_dict(),
                      "val_loss": val_loss,
                      "val_recall_at_1": val_recall[1],
                      "val_recall_at_10": val_recall[10],
                      "epoch": epoch + 1,
                      "total_epochs": epochs,
                      "lr": lr,
                      "batch_size": batch_size,
                      "hidden_dim": hidden_dim,
                      "output_dim": output_dim,
                      "biobert_filename": biobert_filename,
                      "splits_file": splits_file,
                      "run_name": run_name,
                      "unfreeze_biobert_layers": unfreeze_biobert_layers,
                      "biobert_n_layers": biobert_n_layers,
                      "unfrozen_layer_indices": unfrozen_indices,
                      "biobert_lr": biobert_lr if live_text else None}
        if live_text:
            # Only the unfrozen layers' weights need saving -- the frozen
            # embeddings + lower layers are always exactly the pretrained
            # BioBERT weights, reloadable via AutoModel.from_pretrained.
            checkpoint["biobert_unfrozen_state_dict"] = {
                idx: biobert_model.encoder.layer[idx].state_dict() for idx in unfrozen_indices
            }
        torch.save(checkpoint, out_path)
        if is_best:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(f"[{run_name}] Early stopping: no val_loss improvement for {early_stopping_patience} "
                  f"epochs (stopped after epoch {epoch + 1}/{epochs})")
            break

    return best_path, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="where to save checkpoint.pt")
    parser.add_argument("--epochs", type=int, required=True,
                        help="each epoch is only ~len(train_set)/512 steps (~20-160), "
                             "far fewer than the with-abundance model, so this needs to "
                             "be set deliberately — watch val_loss and stop once it plateaus")
    parser.add_argument("--biobert-filename", default="biobert_embeddings.h5",
                        help="e.g. biobert_embeddings_masked.h5 for the gene-name-masking ablation")
    parser.add_argument("--splits-file", default="data/protein_splits.json",
                        help="e.g. data/protein_splits_full_proteome.json for the full-proteome split")
    parser.add_argument("--lr", type=float, default=MAX_LR,
                        help=f"peak learning rate after warmup (default {MAX_LR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"default {BATCH_SIZE}")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="width of the projection heads' hidden layer (default 512)")
    parser.add_argument("--output-dim", type=int, default=256,
                        help="dimension of the final shared embedding space, where cosine "
                             "similarity is computed (default 256; CellWhisperer uses 1024)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="stop once val_loss hasn't improved for this many consecutive "
                             "epochs (default: disabled, always run all --epochs)")
    parser.add_argument("--unfreeze-biobert-layers", type=int, default=0,
                        help="number of BioBERT Transformer layers to unfreeze, counted from "
                             "the last layer backward (default 0 = fully frozen baseline, "
                             "identical to the original behavior -- text comes from the "
                             "precomputed --biobert-filename .h5 file). N > 0 switches to a "
                             "live BioBERT forward pass so gradients can reach the unfrozen "
                             "layers; the text variant encoded live is inferred from "
                             "--biobert-filename's naming convention (masked/family/location/"
                             "GO), so no separate flag is needed to pick it")
    parser.add_argument("--biobert-lr", type=float, default=DEFAULT_BIOBERT_LR,
                        help=f"learning rate for BioBERT's unfrozen layers, as a separate "
                             f"optimizer param group from the projection heads (default "
                             f"{DEFAULT_BIOBERT_LR}); unused when --unfreeze-biobert-layers 0")
    args = parser.parse_args()
    train(checkpoint_path=args.checkpoint, epochs=args.epochs,
          biobert_filename=args.biobert_filename, splits_file=args.splits_file,
          lr=args.lr, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
          output_dim=args.output_dim, early_stopping_patience=args.early_stopping_patience,
          unfreeze_biobert_layers=args.unfreeze_biobert_layers, biobert_lr=args.biobert_lr)
