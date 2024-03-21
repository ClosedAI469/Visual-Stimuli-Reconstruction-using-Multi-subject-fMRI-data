"""
Models used in the task.
"""

import time

import numpy as np
import scipy
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class fMRIEncoder(nn.Module):
    """TODO: Implement the fMRI encoder.
    We will have more than one encoder for fMRI data. The name of this place holder does not suggest we only have one.
    We STARTS with an MLP encoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class ImageEncoder(nn.Module):
    """TODO: Implement the Image encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class CLIP(nn.Module):
    """TODO: Implement the CLIP classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class MLP(nn.Module):
    def __init__(self, input_features, output_features, *hidden_dims):
        """
        A simple multi-layer perceptron.

        :param input_features: The number of input features
        :type input_features: int
        :param output_features: The number of output features
        :type output_features: int
        :param hidden_dims: The number of neurons for each hidden layer
        :type hidden_dims: int
        """
        super().__init__()
        self.input_features, self.hidden_dims, self.output_features = (
            input_features,
            hidden_dims,
            output_features,
        )

        dims = (input_features, *hidden_dims, output_features)

        activation = nn.ReLU()

        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(dims, dims[1:]):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
            linear_layers.append(activation)

        self.model = nn.Sequential(*linear_layers[:-1])

    def forward(self, x):
        return self.model(x)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional=True,
        num_layers=1,
    ):
        """
        A flexible Long Short-Term Memory (LSTM) model that can deal with packed sequences.

        For input as a tensor, do model(x); for input as a packed sequence,
        do model(x, packing) where packing is the lengths of the sequences.

        :param input_size: The number of expected features in the input x
        :type input_size: int
        :param hidden_size: The number of features in the hidden state h
        :type hidden_size: int
        :param bidirectional: If True, becomes a bidirectional LSTM
        :type bidirectional: bool
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        """
        super().__init__()

        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x, packing=None):
        if packing is not None:
            x = pack_padded_sequence(x, packing, batch_first=True, enforce_sorted=False)

        return self.model(x)


# From https://github.com/myousefnezhad/easyfmri/blob/7c106e6564085263163f9055dd7eaf7f9b9c3ac7/Hyperalignment/GPUHA.py
class GPUHA:
    def __init__(self, Dim=None, regularization=10**-4):
        self.G = None  # Shared Space
        self.EigVal = None  # Eigenvalues (Lambda) of Shared Space
        self.Xtrain = None  # Transformed trained data
        self.Etrain = None  # Training Error
        self.Xtest = None  # Transformed test data
        self.ETest = None  # Testing Error
        self.Dim = Dim  # Number of Dimension
        self.regularization = regularization

    def train(self, views, verbose=True, gpu=True):
        # Start Time
        tme = time.time()

        if gpu:
            import torch

        # Show Message or not
        self.verbose = verbose
        # Number of Subjects
        self.V = np.shape(views)[0]

        try:
            if len(self.regularization) == self.V:
                self.eps = [np.float32(e) for e in self.regularization]
            else:
                self.eps = [
                    np.float32(self.regularization) for i in range(self.V)
                ]  # Assume eps is same for each view
        except:
            self.eps = [
                np.float32(self.regularization) for i in range(self.V)
            ]  # Assume eps is same for each view

        self.F = [
            int(np.shape(views)[2]) for i in range(self.V)
        ]  # Assume eps is same for each view

        if self.Dim is None:
            self.k = np.min(
                (np.shape(views)[1], np.shape(views)[2])
            )  # Dimensionality of embedding we want to learn
        else:
            try:
                self.k = np.int32(self.Dim)
            except:
                self.k = np.shape(views)[
                    2
                ]  # Dimensionality of embedding we want to learn

        N = views[0].shape[0]

        _Stilde = np.float32(np.zeros(self.k))
        _Gprime = np.float32(np.zeros((N, self.k)))

        ProjectMats = list()

        if N > self.k:
            gpu = False

        # Take SVD of each view, to calculate A_i and T_i
        for i, (eps, view) in enumerate(zip(self.eps, views)):
            if self.verbose:
                print("TRAIN DATA -> View %d -> Run SVD ..." % (i + 1))

            if not gpu:
                A, S_thin, B = scipy.linalg.svd(view, full_matrices=False)
            else:
                A, S_thin, B = torch.svd(torch.Tensor(view).cuda(), some=False)

            if self.verbose:
                print("TRAIN DATA -> View %d -> Calculate Sigma inverse ..." % (i + 1))

            if not gpu:
                S2_inv = 1.0 / (np.multiply(S_thin, S_thin) + eps)
                T = np.diag(np.sqrt(np.multiply(np.multiply(S_thin, S2_inv), S_thin)))

            else:
                S2_inv = 1.0 / (
                    torch.mul(S_thin, S_thin) + torch.Tensor([eps]).cuda()[0]
                )
                T = torch.diag(torch.sqrt(torch.mul(torch.mul(S_thin, S2_inv), S_thin)))

            if self.verbose:
                print("TRAIN: Calculate dot product AT for View %d" % (i + 1))

            if not gpu:
                ajtj = A.dot(T)
            else:
                ajtj = torch.mm(A, T).cpu().numpy()

            if gpu:
                torch.cuda.empty_cache()

            ProjectMats.append(ajtj)
            if self.verbose:
                print(
                    "TRAIN DATA -> View %d -> Calculate Incremental PCA ..." % (i + 1)
                )
            _Gprime, _Stilde = self._batch_incremental_pca(
                ajtj, _Gprime, _Stilde, i, self.verbose, gpu
            )
            if self.verbose:
                print("TRAIN DATA -> View %d -> Decomposing data matrix ..." % (i + 1))
        self.G = _Gprime
        self.EigVal = _Stilde
        self.Xtrain = list()
        self.Etrain = list()
        if verbose:
            print("TRAIN DATA -> Mapping to shared space ...")
            # Get mapping to shared space
        for pid, project in enumerate(ProjectMats):
            xtrprokject = np.dot(np.dot(project, np.transpose(project)), self.G)
            # Save features
            self.Xtrain.append(xtrprokject)
            # Save errors
            self.Etrain.append(np.linalg.norm(xtrprokject - self.G) ** 2)
            if verbose:
                print("TRAIN DATA -> View %d is projected ..." % (pid + 1))

        if verbose:
            print("Calculating training error ...")

        return self.Xtrain, self.G, time.time() - tme, np.mean(self.Etrain), self.Etrain

    def test(self, views, G=None, verbose=True):
        tme = time.time()
        if G is not None:
            self.G = G
        else:
            if self.G is None:
                if verbose:
                    print("There is no G")
                return None

        # Show Message or not
        self.verbose = verbose
        # Number of Subjects
        self.V_test = np.shape(views)[0]

        try:
            if len(self.regularization) == self.V_test:
                self.eps_test = [np.float32(e) for e in self.regularization]
            else:
                self.eps_test = [
                    np.float32(self.regularization) for i in range(self.V_test)
                ]  # Assume eps is same for each view
        except:
            self.eps_test = [
                np.float32(self.regularization) for i in range(self.V_test)
            ]  # Assume eps is same for each view

        self.F_test = [
            int(np.shape(views)[2]) for i in range(self.V_test)
        ]  # Assume eps is same for each view

        if self.Dim is None:
            self.k = np.shape(views)[2]  # Dimensionality of embedding we want to learn
        else:
            try:
                self.k = np.int32(self.Dim)
            except:
                self.k = np.shape(views)[
                    2
                ]  # Dimensionality of embedding we want to learn

        self.Xtest = list()
        self.ETest = list()

        # Take SVD of each view, to calculate A_i and T_i
        for i, (eps, view) in enumerate(zip(self.eps_test, views)):
            if self.verbose:
                print("TEST DATA -> View %d -> Run SVD ..." % (i + 1))
            A, S_thin, B = scipy.linalg.svd(view, full_matrices=False)
            if self.verbose:
                print("TEST DATA -> View %d -> Calculate Sigma inverse ..." % (i + 1))
            S2_inv = 1.0 / (np.multiply(S_thin, S_thin) + eps)
            T = np.diag(np.sqrt(np.multiply(np.multiply(S_thin, S2_inv), S_thin)))
            if self.verbose:
                print("TEST: Calculate dot product AT for View %d" % (i + 1))
            ajtj = A.dot(T)

            xteprokject = np.dot(np.dot(ajtj, np.transpose(ajtj)), self.G)
            # Save Data
            self.Xtest.append(xteprokject)
            # Save Error
            self.ETest.append(np.linalg.norm(xteprokject - self.G) ** 2)

            if verbose:
                print("TEST DATA -> View %d is projected ..." % (i + 1))

        return self.Xtest, time.time() - tme, np.mean(self.ETest), self.ETest

    def get_G(self):
        return self.G

    def get_Xtrain(self):
        return self.Xtrain

    def get_Xtest(self):
        return self.Xtest

    @staticmethod
    def _batch_incremental_pca(x, G, S, i, verbose, gpu):
        if gpu:
            import torch
        r = G.shape[1]
        b = x.shape[0]
        xh = G.T.dot(x)
        H = x - G.dot(xh)
        if verbose:
            print("TRAIN DATA -> View %d -> IPCA -> Run QR decomposition ..." % (i + 1))
        if not gpu:
            J, W = scipy.linalg.qr(H, overwrite_a=True, mode="full", check_finite=False)
        else:
            J, W = torch.qr(torch.Tensor(H).cuda())
            J = J.cpu().numpy()
            W = W.cpu().numpy()

        if verbose:
            print("TRAIN DATA -> View %d -> IPCA -> Run bmat ..." % (i + 1))
        Q = np.bmat([[np.diag(S), xh], [np.zeros((b, r), dtype=np.float32), W]])

        if verbose:
            print(
                "TRAIN DATA -> View %d -> IPCA -> Run SVD decomposition on Q ..."
                % (i + 1)
            )
            print("TRAIN DATA -> View %d -> IPCA -> Q size: " % (i + 1), np.shape(Q))
        try:
            if not gpu:
                G_new, St_new, _ = scipy.linalg.svd(
                    Q, full_matrices=False, check_finite=False
                )
            else:
                G_new, St_new, _ = torch.svd(torch.Tensor(Q).cuda(), some=False)
                G_new = G_new.cpu().numpy()
                St_new = St_new.cpu().numpy()

        except:
            print("WARNING: SVD for View %d is not coverage!" % (i + 1))
            return G, S
        St_new = St_new[:r]
        if verbose:
            print("TRAIN DATA -> View %d -> IPCA -> Run dot product ..." % (i + 1))
        G_new = np.asarray(np.bmat([G, J]).dot(G_new[:, :r]))
        if gpu:
            torch.cuda.empty_cache()
        return G_new, St_new
