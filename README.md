# d11m

Train a new model (overwrites the selected checkpoint):

```powershell
python -m d11m train "Ala ma kota." --context-size 4 --epochs 100
```

Generate text using the saved model:

```powershell
python -m d11m generate "Ala " --max-new-tokens 100
```

Both commands accept `--checkpoint path/to/model.pt`. The default is `model.pt`
in the project directory. CUDA is selected automatically when available.
Use `python -m d11m --help` or `<command> --help` for options.

Training defaults: context size 128, batch size 32, embedding dimension 128,
10 epochs, and 4 transformer layers. These are starting values for experiments,
not tuned settings. The default context requires at least 127 UTF-8 bytes of
training text (plus BOS and EOS). Use a smaller context for short examples.
