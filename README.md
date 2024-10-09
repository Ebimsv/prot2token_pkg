# prot2token

**prot2token** is A multi-task framework for protein language processing using autoregressive language modeling

## Features

- Prepare models with specific configurations.

## Installation

You can install `prot2token` via `pip`:

```bash
pip install prot2token
```

## Usage

Once installed, you can use the prot2token package to run the test function, which clears the CUDA cache and prepares models.

Example usage:

```bash
prot2token-test
```

This will:

- Prepare the model using the prepare_models function in models.py.

## Sample Code

The `run.py` script contains the following code:

```python
import torch
import pandas as pd
from models import prepare_models

def test():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

    # Prepare models
    net, decoder_tokenizer, inference_configs = prepare_models(name="stability", device="cpu", compile_model=False)

    # Load samples from CSV
    samples = pd.read_csv("../data/inference_data/stability_inference.csv")["input"].tolist()

    # Run inference
    results = net.run(samples, merging_character="")
    print(results)

    torch.cuda.empty_cache()


if **name** == "**main**":
    test()
    print("done!")
```

## Dependencies

- **torch**: A deep learning framework by PyTorch.
- **numpy**: A library for numerical computations in Python.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

# Author

Name:
