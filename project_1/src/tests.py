import numpy as np

from data_gen import SimpleTest
from data_handling import DataHandler
from reg_models import OLSModel, RidgeModel, LassoModel


def test_data_gen():
    # Test data generator
    data = SimpleTest(data_points=3)
    z = data.get_data()

    assert z.shape == (3, 3)
    assert z[2, 0] == 0
    assert z[1, 2] == 1


def test_data_handler():
    tol = 1e-9

    # Test data handler
    data = SimpleTest(data_points=10)  # Creates 10 * 10 = 100 data points

    handler1 = DataHandler(data, test_size=0.2)
    handler2 = DataHandler(data, test_size=0.4)
    handler1.create_cross_validation(kfolds=10)
    _ = handler2.preprocess(degree=2)

    # Test train size
    assert len(handler1.z_train) == 80
    assert len(handler2.z_test) == 40

    # Test X_matrix scaling
    assert abs(np.var(handler2.X_train[:, 1]) - 1) < tol
    assert abs(np.mean(handler2.X_train[:, 0])) < tol

    # Test CV size
    assert len(handler1.data_splits) == 10
    assert len(handler1.data_splits[1][0]) == 90  # Length of train in split
    assert len(handler1.data_splits[4][1]) == 10  # Length of test in split


def test_reg_models():
    tol = 1e-9
    np.random.seed(20)

    data = SimpleTest(data_points=21)  # Creates 10 * 10 = 100 data points
    handler = DataHandler(data, test_size=0.2)
    _ = handler.preprocess(degree=3)
    ols = OLSModel(handler)

    z1 = np.array([[0.0], [0.0], [1.0]])
    z2 = np.array([[0.0], [1.0], [3.0]])
    z3 = np.array([[2.0], [4.0], [0.0]])
    z4 = z3.copy()

    # Test MSE
    assert abs(ols.MSE(z1, z2) - (5 / 3)) < tol
    assert abs(ols.MSE(z1, z3) - (21 / 3)) < tol
    assert abs(ols.MSE(z2, z3) - (22 / 3)) < tol
    assert abs(ols.MSE(z3, z4)) < tol

    # Test R2
    assert abs(ols.R2(z3, z4) - 1) < tol
    assert abs(ols.R2(z1, z2) + 1 / 14) < tol
    assert abs(ols.R2(z1, z3) + (13 / 8)) < tol
    assert abs(ols.R2(z2, z3) + (7 / 4)) < tol

    # Test fit
    X = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )
    z = np.array([[0], [1], [2], [1], [2], [3], [2], [3], [4]])
    beta = ols.fit_model_on_data(X, z)
    assert (X @ beta - z).all() < tol

    # Test everything on simple model
    f = lambda x, y: x
    ols.get_preprocessed_data(degree=3)
    ols.fit_simple_model(deg=2)
    z_tilde = ols.predict(ols.X_test)
    assert (z_tilde - ols.z_test).all() < tol


test_reg_models()
