from typing import Literal
import random

import numpy as np


Optimizer = Literal['gradient-decent', 'stochastic', 'mini-batch']
History = tuple[list[float], list[float]]


class Model:
    def __init__(self, *, weights_file: str | None = None) -> None:
        self.__weights: np.ndarray
        self.__bias: float
        self.__house: str
        self.__train: bool = True
        self.__parameters: dict[str, tuple[np.ndarray, float]] = {}
        if weights_file is not None:
            self.__train = False
            house_params: dict[str, np.ndarray] = np.load(weights_file, allow_pickle=False)
            for house, params in house_params.items():
                weights, bias = params[:-1], params[-1]
                self.__parameters[house] = weights, bias # type: ignore


    def __call__(self, *, house: str, input: int = 0) -> None:
        if self.__train:
            self.__weights = np.random.rand(input, 1)
            self.__bias = 0.0
            self.__house = house
            self.__adam_optimizer_setup() # INFO: setup adam optimizer
        else:
            self.__weights, self.__bias = self.__parameters[house]


    def __backup_parameters(self) -> None:
        self.__parameters[self.__house] = self.__weights.copy(), self.__bias


    def __adam_optimizer_setup(self) -> None:
        self.__t: int = 0
        self.__vw: np.ndarray = np.zeros_like(self.__weights)
        self.__sw: np.ndarray = np.zeros_like(self.__weights)
        self.__vb: float = 0.0
        self.__sb: float = 0.0


    def __adam_optimizer_backward(
            self, x: np.ndarray, gradient: np.ndarray, alpha: float
    ) -> None:
        self.__t += 1
        mv, ms, epsilon = 0.9, 0.99, 1e-7
        w_gradient = np.dot(x.T, gradient)
        b_gradient = np.sum(gradient)
        self.__vw = mv * self.__vw + (1 - mv) * w_gradient
        self.__vb = mv * self.__vb + (1 - mv) * b_gradient
        self.__sw = ms * self.__sw + (1 - ms) * np.square(w_gradient)
        self.__sb = ms * self.__sb + (1 - ms) * np.square(b_gradient)
        vw_hat = self.__vw / (1 - mv ** self.__t)
        vb_hat = self.__vb / (1 - mv ** self.__t)
        sw_hat = self.__sw / (1 - ms ** self.__t)
        sb_hat = self.__sb / (1 - ms ** self.__t)
        self.__weights -= alpha * vw_hat / np.sqrt(sw_hat + epsilon)
        self.__bias -= alpha * vb_hat / np.sqrt(sb_hat + epsilon)


    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        positives = np.clip(z, a_min=0, a_max=None)
        negatives = np.clip(z, a_min=None, a_max=0)
        sigmoid_of_positives = 1 / (1 + np.exp(-positives))
        sigmoid_of_negatives = np.exp(negatives) / (1 + np.exp(negatives))
        yHat = np.where(z > 0, sigmoid_of_positives, sigmoid_of_negatives)
        return yHat


    def __bce_gradient(self, y: np.ndarray, yHat: np.ndarray, batch_size: int) -> np.ndarray:
        return (yHat - y) / batch_size


    def __bce_loss(self, y: np.ndarray, yHat: np.ndarray) -> float:
        yHat = np.clip(yHat, 1e-7, 1 - 1e-7)
        loss = y * np.log(yHat) + (1 - y) * np.log(1 - yHat)
        loss = -1 * loss.mean()
        return float(loss)


    def __accuracy(self, y: np.ndarray, yHat: np.ndarray) -> float:
        yHat = yHat > 0.5
        accuracy = (y == yHat).mean()
        return float(accuracy)


    def __foreward(self, x: np.ndarray) -> np.ndarray:
        z = x.dot(self.__weights) + self.__bias
        yHat = self.__sigmoid(z)
        return yHat


    def __backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yHat: np.ndarray,
        alpha: float,
        batch_size: int
    ) -> None:
        zHat = self.__bce_gradient(y, yHat, batch_size)
        # self.__bias -= alpha * zHat.sum()
        # self.__weights -= alpha * np.dot(x.T, zHat)
        self.__adam_optimizer_backward(x, zHat, alpha) # INFO: backward adam optimizer


    def __gradient_decent(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float,
        epochs: int,
        data_size: int,
        verbose: int = 1,
    ) -> History:
        loss_history: list[float] = []
        accuracy_history: list[float] = []
        for i in range(epochs):
            # ------------ do foreward then backward ----------
            yHat = self.__foreward(x)
            self.__backward(x, y, yHat, alpha, data_size)
            # ------------ get loss and accuracy values -------
            loss = self.__bce_loss(y, yHat)
            accuracy = self.__accuracy(y, yHat)
            # ------------ hold loss and accuracy values ------
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            # ------------ print status -----------------------
            if verbose in (1, 2):
                print(f'epoch {i+1: 4}/{epochs}    loss={loss:.5f}     accuracy: {accuracy:.5f}')
        return loss_history, accuracy_history


    def __stochastic_gradient_decent(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float,
        epochs: int,
        data_size: int,
        verbose: int = 1,
    ) -> History:
        loss_history: list[float] = []
        accuracy_history: list[float] = []
        loss = accuracy = 0.0
        for i in range(epochs):
            for _ in range(data_size):
                # ------------ get random sample ------------------
                randint = random.randint(0, data_size - 1)
                rand_x = x[randint, :]
                rand_y = y[randint]
                # ------------ do foreward then backward ----------
                yHat = self.__foreward(rand_x)
                self.__backward(rand_x, rand_y, yHat, alpha, 1)
                # ------------ get loss and accuracy values -------
                loss = self.__bce_loss(rand_y, yHat)
                accuracy = self.__accuracy(rand_y, yHat)
                # ------------ hold loss and accuracy values ------
                loss_history.append(loss)
                accuracy_history.append(accuracy)
                # ------------ print status if verbose == 2 ---------
                if verbose == 2:
                    print(f'epoch {i+1: 4}/{epochs} sample: {randint+1: 4}   loss: {loss:.5f}   accuracy: {accuracy:.5f}')
            # ------------ print status if verbose == 1 ---------
            if verbose == 1:
                print(f'epoch {i+1: 4}/{epochs}   loss: {loss:.5f}   accuracy: {accuracy:.5f}')
        return loss_history, accuracy_history


    def __mini_batch_gradient_decent(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float,
        epochs: int,
        data_size: int,
        batch_size: int,
        verbose: int = 1,
    ) -> History:
        batchs_count = data_size // batch_size
        loss_history: list[float] = []
        accuracy_history: list[float] = []
        loss = accuracy = 0.0
        for i in range(epochs):
            for j in range(batchs_count):
                # ------------ get batch ------------------
                start = j * batch_size
                end = min(start + batch_size, data_size)
                rand_x = x[start: end, :]
                rand_y = y[start: end]
                # ------------ do foreward then backward ----------
                yHat = self.__foreward(rand_x)
                self.__backward(rand_x, rand_y, yHat, alpha, end - start)
                # ------------ get loss and accuracy values -------
                loss = self.__bce_loss(rand_y, yHat)
                accuracy = self.__accuracy(rand_y, yHat)
                # ------------ hold loss and accuracy values ------
                loss_history.append(loss)
                accuracy_history.append(accuracy)
                # ------------ print status if verbose == 2 ---------
                if verbose == 2:
                    print(f'epoch {i+1: 4}/{epochs} batch: {j+1: 3}/{batchs_count}   loss: {loss:.5f}   accuracy: {accuracy:.5f}')
            # ------------ print status if verbose == 1 ---------
            if verbose == 1:
                print(f'epoch {i+1: 4}/{epochs} (batchs: {batchs_count})   loss: {loss:.5f}   accuracy: {accuracy:.5f}')
        return loss_history, accuracy_history


    def save(self, filename: str) -> None:
        params = { house: np.append(w, b) for house, (w, b) in self.__parameters.items() }
        np.savez(filename, **params, allow_pickle=False)


    def test(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        yHat = self.__foreward(x)
        loss = self.__bce_loss(y, yHat)
        accuracy = self.__accuracy(y, yHat)
        return loss, accuracy


    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        yHat = self.__foreward(x)
        return yHat


    def predict(self, x: np.ndarray) -> np.ndarray:
        yHat = self.__foreward(x)
        yHat = (yHat > 0.5).astype(int)
        return yHat


    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float = 0.01,
        epochs: int = 1_337,
        batch_size: int = 100,
        optimizer: Optimizer = 'mini-batch',
        verbose: int = 1,
        ) -> History:

        data_size = x.shape[0]

        assert x.shape[0] == y.shape[0], 'x, and y must have same batch size'
        assert 0 < batch_size < data_size, 'Batch size cannot be greater than traning data size'

        match optimizer:
            case 'gradient-decent':
                history = self.__gradient_decent(
                    x, y, alpha=alpha, epochs=epochs, data_size=data_size, verbose=verbose
                )
            case 'stochastic':
                history = self.__stochastic_gradient_decent(
                    x, y, alpha=alpha, epochs=epochs, data_size=data_size, verbose=verbose
                )
            case 'mini-batch':
                history = self.__mini_batch_gradient_decent(
                    x, y, alpha=alpha, epochs=epochs, data_size=data_size,
                    batch_size=batch_size, verbose=verbose
                )
            # case _:
            #     raise ValueError(
            #         f"Unknown optimizer: `{optimizer}' supported ones:"
            #         '  gradient-decent, stochastic, mini-batch'
            #     )
        self.__backup_parameters()
        return history

