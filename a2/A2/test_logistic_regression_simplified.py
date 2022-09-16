'''
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 2 Auto Marker
B. Chan
'''

import _pickle as pickle
import numpy as np
import logistic_regression
import train_logistic_regression


def accuracy(y, p):
    """ This returns the accuracy of prediction given true labels.

    Args:
    - y (ndarray (shape: (N,1))): A Nx1 matrix consisting of true labels
    - p (ndarray (shape: (N,C))): A NxC matrix consisting N C-dimensional probabilities for each input.
    
    Output:
    - acc (float): Accuracy of predictions compared to true labels
    """
    assert y.shape[0] == p.shape[0], f"Number of samples must match"

    # Pick indicies that maximize each row
    y_pred = np.expand_dims(np.argmax(p, axis=1), axis=1)
    acc = sum(y_pred == y) * 100 / y.shape[0]

    return acc


class TestLogisticRegression():
    def __init__(self):
        with open('logistic_regression_soln.pkl', 'rb') as f:
            self.true_vals = pickle.load(f)

    def test_feature_map(self):
        try:
            x = self.true_vals["x_3"]
            x_mapped = train_logistic_regression.feature_map(x)
            return int(np.allclose(x_mapped, self.true_vals["x_3_mapped"]))
        except Exception as e:
            print(e)
            return 0

    def test_loss_and_grad_1(self, atol=1):
        # no regularization, 2-D, 20
        try:
            x1 = self.true_vals['x1']
            y1 = self.true_vals['y1']
            model = logistic_regression.LogisticRegression(x1.shape[1], 2)
            model.parameters = self.true_vals['params_1']
            [nll, grad] = model._compute_loss_and_gradient(x1, y1)
            m = 0

            if np.allclose(nll, self.true_vals['nll_1'], atol=atol):
                m += 1
            
            if np.allclose(grad, self.true_vals["grad_1"], atol=atol):
                m += 1
            return m, nll, grad

        except Exception as e:
            print(e)
            return 0, None, None

    def test_loss_and_grad_reg_1(self, atol=1):
        # with regularization, 2-D, 20
        try:
            x1 = self.true_vals['x1']
            y1 = self.true_vals['y1']
            model = logistic_regression.LogisticRegression(x1.shape[1], 2)
            model.parameters = self.true_vals['params_2']
            [nll, grad] = model._compute_loss_and_gradient(x1, y1, 10, 5)
            m = 0

            break_loop = False
            for comb_i in range(4):
                if np.allclose(nll, self.true_vals['nll_2'][comb_i], atol=atol):
                    break_loop = True
                    m += 1
                
                if np.allclose(grad, self.true_vals['grad_2'][comb_i], atol=atol):
                    break_loop = True
                    m += 1

                if break_loop:
                    break
            return m, nll, grad

        except Exception as e:
            print(e)
            return 0, None, None

    def test_loss_and_grad_2(self, atol=1):
        # no reg, 20, 4-D
        try:
            x2 = self.true_vals['x2']
            y2 = self.true_vals['y2']
            model = logistic_regression.LogisticRegression(x2.shape[1], 4)
            model.parameters = self.true_vals['params_3']
            [nll, grad] = model._compute_loss_and_gradient(x2, y2)
            m = 0

            if np.allclose(nll, self.true_vals['nll_3'], atol=atol):
                m += 1
            
            if np.allclose(grad, self.true_vals['grad_3'], atol=atol):
                m += 1
            return m, nll, grad
        
        except Exception as e:
            print(e)
            return 0, None, None

    def test_loss_and_grad_reg_2(self, atol=1):
        # with reg, 20, 4-D
        try:
            x2 = self.true_vals['x2']
            y2 = self.true_vals['y2']
            model = logistic_regression.LogisticRegression(x2.shape[1], 4)
            model.parameters = self.true_vals['params_4']
            [nll, grad] = model._compute_loss_and_gradient(x2, y2, 10, 5)
            m = 0

            break_loop = False
            for comb_i in range(4):
                print(self.true_vals['nll_4'][comb_i], self.true_vals['grad_4'][comb_i])
                if np.allclose(nll, self.true_vals['nll_4'][comb_i], atol=atol):
                    break_loop = True
                    m += 1
                
                if np.allclose(grad, self.true_vals['grad_4'][comb_i], atol=atol):
                    break_loop = True
                    m += 1

                if break_loop:
                    break
            return m, nll, grad
        
        except Exception as e:
            print(e)
            return 0, None, None


if __name__ == "__main__":
    tester = TestLogisticRegression()
    print(tester.test_feature_map())
    print(tester.test_loss_and_grad_1())
    print(tester.test_loss_and_grad_reg_1())
    print(tester.test_loss_and_grad_2())
    print(tester.test_loss_and_grad_reg_2())
