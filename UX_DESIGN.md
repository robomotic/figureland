# User Experience Design Principles

Figureland follows these UX principles for both CLI and API usage:

## 1. Zero Configuration Defaults
- ✅ Sensible defaults work out of the box
- ✅ No required parameters to get started
- ✅ `figureland` command works immediately after install

## 2. Progressive Complexity
- ✅ Simple for basic use cases
- ✅ Full power available for advanced users
- ✅ Gradual learning curve

## 3. CLI Usability
- ✅ Standard `--help` for all commands
- ✅ Intuitive parameter names
- ✅ Predictable command structure
- ✅ Colorized output with progress bars
- ✅ Clear error messages
- ✅ Dry run support

## 4. API Design
- ✅ Consistent method names matching CLI
- ✅ Proper type hints
- ✅ Good default values
- ✅ Clear documentation strings
- ✅ Predictable return values

## 5. Error Handling
- ✅ Actionable error messages
- ✅ Validation before execution
- ✅ Graceful degradation
- ✅ Proper stack traces in debug mode

## 6. Logging & Observability
- ✅ Structured logging
- ✅ Appropriate log levels
- ✅ Progress indication for long operations
- ✅ Performance metrics
- ✅ Reproducible run logs

## Example Workflows

### Simplest Possible Usage
```bash
# Generate dataset with defaults
figureland
```

### Common Usage
```bash
# Generate 1000 episodes
figureland n_episodes=1000 parallel_generation=true
```

### Advanced Usage
```bash
# Custom config override
figureland resolution=[1024,1024] \
  episode_length=500 \
  batch_size=64 \
  physics.gravity=19.6
```

### Sweep Parameters
```bash
# Run parameter sweep
figureland --multirun batch_size=16,32,64 physics.gravity=4.9,9.8,19.6
```
