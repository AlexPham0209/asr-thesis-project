from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:
        pass
