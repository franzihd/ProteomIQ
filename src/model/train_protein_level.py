"""
Trains the protein and text projection heads (frozen ESM2 / BioBERT inputs)
with a symmetric InfoNCE loss.

Usage:
    uv run python src/model/train_protein_level.py \\
        --checkpoint data/runs/full_proteome/multiview_experiment/multiview.pt \\
        --epochs 100 --lr 1.16e-4 --batch-size 512 --hidden-dim 2048 --output-dim 1024 \\
        --early-stopping-patience 10 \\
        --biobert-filename biobert_embeddings_masked.h5 \\
        --view-biobert-filenames biobert_embeddings_masked.h5 \\
            biobert_embeddings_family_view.h5 biobert_embeddings_location_view.h5 \\
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
from src.model.dataset import ProteomicsDataset, parse_biobert_text_flags, GroupedBatchSampler
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


def multi_positive_infonce_loss(protein_vec, text_vec, protein_idx):
    """
    Multi-positive extension of infonce_loss, for ProteomicsDataset(
    multi_positive=True) + GroupedBatchSampler batches, where more than one
    row in a batch can legitimately share the same protein (its different
    text views). Plain infonce_loss's `labels = arange(N)` assumes exactly
    one positive per row (the diagonal) -- wrong here: rows i and j should
    both be pulled together whenever protein_idx[i] == protein_idx[j], not
    just when i == j.

    positive_mask[i, j] = (protein_idx[i] == protein_idx[j]) -- symmetric
    by construction, and always True for i == j (a row is trivially its own
    positive). Self is deliberately kept IN the positive set, unlike the
    textbook SupCon convention of excluding the anchor from its own
    positive/denominator sum: textbook SupCon assumes many same-class rows
    are available per batch, so excluding self costs nothing there. Here,
    282/13,515 proteins have only ONE available view (see dataset.py's
    multi_positive branch) -- excluding self for those rows would leave
    zero positives and a broken (divide-by-zero) loss term. Including self
    sidesteps that edge case entirely and costs nothing for proteins with
    2-3 views (self is just one additional guaranteed positive alongside
    the real ones).

    Symmetric like infonce_loss: both directions (protein->text via
    `logits`, text->protein via `logits.T`) use the same positive_mask
    (equality is symmetric, so mask.T == mask) and are averaged. Because
    GroupedBatchSampler duplicates a protein's ESM2 embedding across its
    2-3 rows (see that class's docstring), BOTH directions genuinely have
    multiple positives when a protein has more than one view in the batch
    -- not just the protein->text direction, unlike the earlier asymmetric
    (N-unique-proteins x N-texts) design that was proposed before this was
    actually implemented. That asymmetric design would avoid a small
    amount of redundant compute (re-embedding the same protein 2-3x per
    batch instead of once), but protein_head is a small MLP -- the redundant
    compute is negligible, and this symmetric, mask-based version is simpler
    to implement correctly and reuses infonce_loss's existing structure
    almost as-is.
    """
    logits = (protein_vec @ text_vec.T) * (1 / 0.07)
    positive_mask = (protein_idx.unsqueeze(1) == protein_idx.unsqueeze(0))

    def directional_loss(logits_dir):
        log_prob = logits_dir - logits_dir.logsumexp(dim=1, keepdim=True)
        n_positives = positive_mask.sum(dim=1)  # always >= 1, via self-inclusion
        return -(log_prob * positive_mask).sum(dim=1).div(n_positives).mean()

    return (directional_loss(logits) + directional_loss(logits.T)) / 2


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
          unfreeze_biobert_layers: int = 0, biobert_lr: float = DEFAULT_BIOBERT_LR,
          view_biobert_filenames: list[str] | None = None, multi_positive: bool = False):
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
    view_biobert_filenames: optional list of biobert_embeddings*.h5 filenames
    (e.g. [masked Function, Family-view, Location-view]) -- when given, the
    TRAINING dataset picks one at random per protein per __getitem__ call
    instead of reading `biobert_filename` (multi-view random-selection
    experiment; motivated by the finding that concatenating
    more distinguishing text, as M0-M3 did, helps closed-set self-ID tasks
    while hurting open-vocabulary generalization -- this varies WHICH text is
    shown instead of adding more content per example). VALIDATION deliberately
    keeps using the single, fixed `biobert_filename` (not randomized) so
    val_loss stays comparable across epochs and across runs -- augmentation
    applies to training only, same principle as random-crop-train/
    center-crop-val in vision. Not combinable with unfreeze_biobert_layers > 0
    (see ProteomicsDataset's own guard).
    multi_positive: when True (requires view_biobert_filenames), TRAINING uses
    every available view as its own row (not one random draw per protein --
    see ProteomicsDataset's multi_positive branch), batched via
    GroupedBatchSampler so a protein's views co-occur in the same batch, and
    trained with multi_positive_infonce_loss instead of infonce_loss. This is
    the CMC/SupCon-inspired extension of the plain multi-view random-selection
    experiment above.
    VALIDATION is unaffected either way: still the single, fixed
    biobert_filename, plain infonce_loss, same rationale as the
    view_biobert_filenames case (comparability across epochs/runs).
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
        view_biobert_filenames=view_biobert_filenames, multi_positive=multi_positive,
    )
    print(f"Training data: data/full_proteome/  |  Training pairs: {len(dataset):,}")
    if multi_positive:
        print(f"Multi-positive training text: {view_biobert_filenames}  |  "
              f"validation text (fixed): {biobert_filename}  |  lr: {lr}  |  batch_size: {batch_size}  |  "
              f"hidden_dim: {hidden_dim}  |  output_dim: {output_dim}")
    elif view_biobert_filenames:
        print(f"Multi-view training text: {view_biobert_filenames}  |  "
              f"validation text (fixed): {biobert_filename}  |  lr: {lr}  |  batch_size: {batch_size}  |  "
              f"hidden_dim: {hidden_dim}  |  output_dim: {output_dim}")
    else:
        print(f"BioBERT source: {biobert_filename}  |  lr: {lr}  |  batch_size: {batch_size}  |  "
              f"hidden_dim: {hidden_dim}  |  output_dim: {output_dim}")

    if multi_positive:
        # Plain shuffle=True would almost never co-locate a protein's 2-3
        # views in the same batch (see GroupedBatchSampler's docstring) --
        # the multi-positive loss has nothing to pull together if they
        # don't. batch_sampler replaces batch_size+shuffle entirely
        # (mutually exclusive in PyTorch's DataLoader).
        batch_sampler = GroupedBatchSampler(dataset, batch_size=batch_size, seed=SEED)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        shuffle_generator = torch.Generator().manual_seed(SEED)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=shuffle_generator)

    # Validation always uses the single fixed biobert_filename, never
    # view_biobert_filenames -- see the docstring note above.
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
        for batch in dataloader:
            if multi_positive:
                protein_emb, text_batch, protein_idx = batch
                protein_idx = protein_idx.to(device)
            else:
                protein_emb, text_batch = batch
            protein_emb = protein_emb.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                protein_vec = protein_head(protein_emb)
                text_vec = compute_text_vec(text_batch)
                loss = (multi_positive_infonce_loss(protein_vec, text_vec, protein_idx)
                        if multi_positive else infonce_loss(protein_vec, text_vec))

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
                      "biobert_lr": biobert_lr if live_text else None,
                      "view_biobert_filenames": view_biobert_filenames,
                      "multi_positive": multi_positive}
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
    parser.add_argument("--view-biobert-filenames", nargs="+", default=None,
                        help="multi-view random-selection experiment: space-separated list of "
                             "biobert_embeddings*.h5 filenames (e.g. "
                             "biobert_embeddings_masked.h5 biobert_embeddings_family_view.h5 "
                             "biobert_embeddings_location_view.h5) -- TRAINING picks one at "
                             "random per protein per sample instead of --biobert-filename; "
                             "VALIDATION still uses --biobert-filename alone, unrandomized. "
                             "Not combinable with --unfreeze-biobert-layers > 0")
    parser.add_argument("--multi-positive", action="store_true",
                        help="CMC/SupCon-inspired extension of --view-biobert-filenames: instead "
                             "of one random view per protein per sample, use every available view "
                             "as its own training row, batched (via GroupedBatchSampler) so a "
                             "protein's views co-occur, and trained with a multi-positive InfoNCE "
                             "loss that pulls all of a protein's co-occurring views together "
                             "instead of assuming exactly one positive per row. Requires "
                             "--view-biobert-filenames; VALIDATION is unaffected (still the "
                             "single fixed --biobert-filename, plain infonce_loss)")
    args = parser.parse_args()
    train(checkpoint_path=args.checkpoint, epochs=args.epochs,
          biobert_filename=args.biobert_filename, splits_file=args.splits_file,
          lr=args.lr, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
          output_dim=args.output_dim, early_stopping_patience=args.early_stopping_patience,
          unfreeze_biobert_layers=args.unfreeze_biobert_layers, biobert_lr=args.biobert_lr,
          view_biobert_filenames=args.view_biobert_filenames, multi_positive=args.multi_positive)
