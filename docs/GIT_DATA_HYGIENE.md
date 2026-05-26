# Git and data hygiene

This repository is meant to track source code, configs, docs, and lightweight examples only. Do not commit generated Isaac Sim frames, merged indices, checkpoints, training plots, or local virtual environments.

## Clean generated local data

Preview what would be deleted:

```bat
python tools\clean_generated_data.py
```

Actually delete generated data:

```bat
python tools\clean_generated_data.py --yes
```

This removes contents under `datasets/`, `results/`, Python caches, logs, checkpoints, and generated binary arrays. It recreates `datasets/.gitkeep` and `results/.gitkeep`.

## Stop tracking generated files already committed

`.gitignore` only affects new untracked files. If data was already committed or staged, remove it from the Git index while keeping your local files:

```bat
git rm -r --cached datasets results runs outputs checkpoints wandb tensorboard 2>nul
git rm --cached *.pt *.pth *.ckpt *.npz *.npy 2>nul
git add .gitignore .gitattributes datasets\.gitkeep results\.gitkeep tools\clean_generated_data.py docs\GIT_DATA_HYGIENE.md
git commit -m "Add gitignore and remove generated artifacts from git"
```

If you want to delete the local generated data too, run:

```bat
python tools\clean_generated_data.py --yes
```

## If large files were already pushed

Removing files in a normal commit does not remove them from Git history. For public or shared repos, prefer leaving history alone unless the files contain secrets. For private repos where you need to shrink history, use `git filter-repo` or BFG Repo-Cleaner, then force-push carefully.
