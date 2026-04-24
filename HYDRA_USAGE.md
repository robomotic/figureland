# Hydra Integration Guide

Figureland now uses Hydra for all configuration and logging.

## Basic Usage

### Generate single episode:
```bash
python main.py
```

### Generate 1000 episodes with parallel processing:
```bash
python main.py parallel_generation=true n_episodes=1000
```

### Enable debug logging:
```bash
python main.py hydra.verbose=true
```

### Change resolution and episode length:
```bash
python main.py resolution=[512,512] episode_length=200
```

### Use GPU vs CPU:
```bash
python main.py use_gpu=true
python main.py use_gpu=false
```

### Change physics parameters:
```bash
python main.py physics.gravity=19.6 physics.friction=0.1
```

### Multirun sweep over multiple parameters:
```bash
python main.py --multirun batch_size=8,16,32 physics.gravity=4.9,9.8,19.6
```

## Logging

Hydra automatically configures logging:
- Logs to both console and file
- Log files stored in `outputs/<date>/<time>/figureland.log`
- Full config saved in `.hydra/` directory for each run

## Configuration Structure

```
config/
├── config.yaml              # Main config
├── shapes/
│   └── default.yaml         # Shape generation config
├── physics/
│   └── default.yaml         # Physics engine config
├── rendering/
│   └── default.yaml         # Rendering config
├── parallel/
│   └── default.yaml         # Parallel generation config
├── output/
│   └── default.yaml         # Output format config
└── hydra/
    └── job_logging/
        └── custom.yaml      # Custom logging config
```

## Overriding Configs

All parameters can be overridden from command line:
```bash
# Use only squares and triangles
python main.py shapes.shape_types=[square,triangle]

# Change split ratios
python main.py parallel.train_ratio=0.7 parallel.val_ratio=0.2 parallel.test_ratio=0.1

# Export to MP4 only
python main.py output.formats=[mp4]
```
