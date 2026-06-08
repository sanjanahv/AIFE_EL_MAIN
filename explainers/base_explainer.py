import pandas as pd

class BaseExplainer:
    def explain(self, profiles: pd.DataFrame) -> dict:
        raise NotImplementedError

    def get_method_id(self) -> int:
        raise NotImplementedError

    def get_method_name(self) -> str:
        raise NotImplementedError
