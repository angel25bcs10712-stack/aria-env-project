# Training module — import train() only when explicitly needed
# to avoid triggering heavy ML imports (trl, torch, transformers)
# on every module load.
