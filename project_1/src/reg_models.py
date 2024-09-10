from project_1.src.data_handling import DataHandler


class RegModel:
    def __init__(self, data: DataHandler) -> None:
        pass

    def __fit_model(
        self, degree: int, ridge_lambda: float = 0, lasso_lambda: float = 0
    ):
        pass

    def MSE(self) -> float:
        pass

    def R2(self) -> float:
        pass


class OLSModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int):
        super().__fit_model(degree=degree)


class RidgeModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        super().__fit_model(degree=degree, ridge_lambda=lmbda)


class LassoModel(RegModel):
    def __init__(self, data: DataHandler) -> None:
        super().__init__(data)

    def fit_model(self, degree: int, lmbda: float):
        super().__fit_model(degree=degree, lasso_lambda=lmbda)
