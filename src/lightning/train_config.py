from pydantic.dataclasses import dataclass # as pydantic_dataclass

@dataclass
class LitTrainConfig:

    benchmark: bool=True  # Perform initial conv optimization sweep?
    accumulate_grad_batches: int=2
    max_epochs: int=100
    learning_rate: float=1e-4
    save_every_n: int=50
