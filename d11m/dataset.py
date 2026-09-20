from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, tokens, *, context_size):
        self.tokens = tokens
        self.context_size = context_size

    def __len__(self):
        return len(self.tokens) - self.context_size

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError

        end_index = index + self.context_size

        input_tokens = self.tokens[index:end_index]
        target_tokens = self.tokens[index + 1:end_index + 1]

        return input_tokens, target_tokens
