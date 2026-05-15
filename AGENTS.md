# AGENTS.md

## Cursor Cloud specific instructions

This is a research deep learning project (HMS-TENet) for EEG/EOG signal classification and regression using PyTorch. There is no web server, database, or external service to start.

### Project structure

- `model/network.py` — Main HMS-TENet network (entry class: `MutilModal`)
- `model/ResNet.py` — ResNet residual block module
- `model/data_processing.py` — PyTorch Dataset class for loading `.npy` data
- `dataprocessing/bands.py` — EEG signal preprocessing (bandpass filtering, differential entropy)

### Running the code

- No training script or `main.py` exists in the repo. The code provides model architecture definitions and data preprocessing utilities only.
- To import model modules, set `sys.path` to include `model/` or run from within that directory (e.g., `cd model && python3 -c "from network import MutilModal"`).
- The `network.py` imports `ResNet` with a relative import (`from ResNet import ResNet`), so it must be run from the `model/` directory or with `model/` on `sys.path`.

### Known issues

- `MutilModal.forward()` has a bug: `self.eeg_modal(e_data)` returns a tuple `(x, atts)` but the result is used directly as a tensor. To test the full pipeline manually, unpack the tuple: `e_data, atts = model.eeg_modal(eeg_data)`.

### Dependencies

Install via: `pip install torch numpy scipy tqdm matplotlib`

No `requirements.txt` file exists in the repository despite being referenced in the README.

### Testing

There are no automated tests or test framework in this repository. Verify the environment by importing modules and running model components with synthetic data (see README for dependency versions).
