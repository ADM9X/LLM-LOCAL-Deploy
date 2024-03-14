from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):

    def __init__(self):
        super().__init__()
        src_dir = Path(__file__).absolute().parent.parent.parent
        self.data_dir = src_dir / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(f'"{str(self.data_dir)}" Directory does not exist')

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _predict(self, input):
        pass

    @abstractmethod
    def get_predict(self):
        pass
