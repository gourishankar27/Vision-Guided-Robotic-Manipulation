# Isaac Franka workflow

## Windows command note

Use one-line commands in `cmd.exe`. The PowerShell continuation character is a backtick, but `cmd.exe` treats it as a normal argument.

`cmd.exe` one line:

```bat
python scripts\isaac_sim\convert_episode_to_training_npz.py --episode datasets\isaac_franka_clean\episode.npz --output datasets\isaac_franka_clean\index.jsonl
```

`cmd.exe` multiline:

```bat
python scripts\isaac_sim\convert_episode_to_training_npz.py ^
  --episode datasets\isaac_franka_clean\episode.npz ^
  --output datasets\isaac_franka_clean\index.jsonl
```

PowerShell multiline:

```powershell
python scripts\isaac_sim\convert_episode_to_training_npz.py `
  --episode datasets\isaac_franka_clean\episode.npz `
  --output datasets\isaac_franka_clean\index.jsonl
```

## Recommended flow

1. Collect an aligned Isaac Sim episode:

```bat
python scripts\isaac_sim\franka_reacher_collect.py --headless --output-dir datasets\isaac_franka_clean
```

2. Convert the episode to JSONL:

```bat
python scripts\isaac_sim\convert_episode_to_training_npz.py --episode datasets\isaac_franka_clean\episode.npz --output datasets\isaac_franka_clean\index.jsonl
```

3. Train the Isaac-to-PyTorch bridge in the lightweight PyTorch environment:

```bat
python train_isaac_franka.py --config configs\isaac_franka.yaml --index datasets\isaac_franka_clean\index.jsonl --epochs 5 --output-dir results\isaac_franka_smoke
```

The bridge model learns `RGB + q_t + ee_t + cube_t + target -> q_{t+1} - q_t` and `ee_{t+1}`. This confirms that the dataset is readable and that the visual/state model can train before replacing the prediction head with a differentiable Warp/robot dynamics backend.
