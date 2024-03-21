"""
Models used in the task.
"""

import time
from utils.TonysMLP import TonysMLP
import numpy as np
import scipy
import torch
from torch import nn, optim
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


# From https://github.com/myousefnezhad/easyfmri/blob/master/Hyperalignment/GPUHA.py
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


# From https://github.com/myousefnezhad/easyfmri/blob/master/Hyperalignment/DHA.py
class DHA:
    def __init__(
        self,
        net_shape,
        activation,
        loss_type="mse",
        optim="sgd",
        iteration=10,
        epoch=10,
        learning_rate=0.1,
        regularization=10**-4,
        best_result_enable=True,
        gpu_enable=torch.cuda.is_available(),
    ):

        assert len(net_shape) > 0, "Model must have at least one layer!"
        assert len(activation) > 0, "Please enter the activation function"
        if len(activation) != 1:
            assert len(net_shape) == len(
                activation
            ), "Length of the model and the list of activation functions are not matched!"

        assert int(iteration) > 0, "Iteration must be grater than 0"

        assert int(epoch) > 0, "Epoch must be grater than 0"

        assert float(regularization) > 0, "Regularization must be grater than 0"

        assert float(learning_rate) > 0, "Learning rate must be grater than 0"

        assert (
            str.lower(optim) == "adam" or str.lower(optim) == "sgd"
        ), "Optimization algorithm is wrong!"

        assert (
            str.lower(loss_type) == "norm"
            or str.lower(loss_type) == "mean"
            or str.lower(loss_type) == "soft"
            or str.lower(loss_type) == "mse"
        ), "Loss type is wrong!"

        self.best_result_enable = best_result_enable
        self.net_shape = net_shape
        self.activation = activation
        self.iteration = int(iteration)
        self.epoch = int(epoch)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.gpu_enable = gpu_enable
        self.loss_type = str.lower(loss_type)
        self.optim = str.lower(optim)
        self.Share = None
        self.ha_loss_vec = None
        self.ha_loss = None
        self.ha_loss_test_vec = None
        self.ha_loss_test = None
        self.TrainFeatures = None
        self.TrainRuntime = None
        self.TestFeatures = None
        self.TestRuntime = None

    def train(self, views, verbose=True):
        tic = time.time()
        assert (
            len(np.shape(views)) == 3
        ), "Data shape must be 3D, i.e. sub x time x voxel"

        self.Share = None
        self.TrainFeatures = None

        NumSub, NumTime, NumVoxel = np.shape(views)
        NumFea = self.net_shape[-1]
        if NumFea is None:
            NumFea = np.min((NumTime, NumVoxel))
            if verbose:
                print(
                    "Number of features is automatically assigned, Features: ", NumFea
                )
                self.net_shape[-1] = NumFea

        Share = np.random.randn(NumTime, NumFea)

        if self.loss_type == "mse":
            criterion = torch.nn.MSELoss()
        elif self.loss_type == "soft":
            criterion = torch.nn.MultiLabelSoftMarginLoss()
        elif self.loss_type == "mean":
            criterion = torch.mean
        elif self.loss_type == "norm":
            criterion = torch.norm
        else:
            raise Exception(
                "Loss function type is wrong! Options: 'mse', 'soft', 'mean', or 'norm'"
            )

        self.ha_loss_vec = list()

        self.ha_loss = None

        for j in range(self.iteration):

            NewViews = list()
            G = torch.Tensor(Share)

            for s in range(NumSub):
                net_shape = np.concatenate(([NumVoxel], self.net_shape))
                net = TonysMLP(
                    model=net_shape,
                    activation=self.activation,
                    gpu_enable=self.gpu_enable,
                )

                if self.optim == "adam":
                    optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
                elif self.optim == "sgd":
                    optimizer = optim.SGD(net.parameters(), lr=self.learning_rate)
                else:
                    raise Exception(
                        "Optimization algorithm is wrong! Options: 'adam' or 'sgd'"
                    )

                X = torch.Tensor(views[s])
                net.train()

                for epoch in range(self.epoch):
                    # Send data to GPU
                    if self.gpu_enable:
                        X = X.cuda()
                        G = G.cuda()

                    optimizer.zero_grad()
                    fX = net(X)

                    if self.loss_type == "mse" or self.loss_type == "soft":
                        loss = criterion(fX, G) / NumTime
                    else:
                        loss = criterion(fX - G) / NumTime

                    loss.backward()
                    optimizer.step()
                    sum_loss = loss.data.cpu().numpy()

                    if verbose:
                        print(
                            "TRAIN, UPDATE NETWORK: Iteration {:5d}, Subject {:6d}, Epoch {:6d}, loss error: {}".format(
                                j + 1, s + 1, epoch + 1, sum_loss
                            )
                        )

                NewViews.append(net(X).data.cpu().numpy())

            ha_model = GPUHA(Dim=NumFea, regularization=self.regularization)

            if NumFea >= NumTime:
                ha_model.train(views=NewViews, verbose=verbose, gpu=self.gpu_enable)
            else:
                ha_model.train(views=NewViews, verbose=verbose, gpu=False)

            Share = ha_model.G
            out_features = ha_model.Xtrain
            error = np.mean(ha_model.Etrain)

            if error == 0:
                assert (
                    self.Share is not None
                ), "All extracted features are zero, i.e. number of features is not enough for creating a shared space"
                self.TrainRuntime = time.time() - tic
                return self.TrainFeatures, self.Share

            if self.best_result_enable:
                if self.ha_loss is None:
                    self.Share = Share
                    self.TrainFeatures = out_features
                    self.ha_loss = error

                if error <= self.ha_loss:
                    self.Share = Share
                    self.TrainFeatures = out_features
                    self.ha_loss = error
            else:
                self.Share = Share
                self.TrainFeatures = out_features
                self.ha_loss = error

            if verbose:
                print("Hyperalignment error: {}".format(error))

            self.ha_loss_vec.append(error)

        self.TrainRuntime = time.time() - tic
        return self.TrainFeatures, self.Share

    def test(self, views, TrainShare=None, verbose=True):
        tic = time.time()
        assert (
            len(np.shape(views)) == 3
        ), "Data shape must be 3D, i.e. sub x time x voxel"

        self.TestFeatures = None

        NumSub, NumTime, NumVoxel = np.shape(views)
        NumFea = self.net_shape[-1]
        if NumFea is None:
            NumFea = np.min((NumTime, NumVoxel))
            if verbose:
                print(
                    "Number of features is automatically assigned, Features: ", NumFea
                )
                self.net_shape[-1] = NumFea

        if TrainShare is not None:
            Share = TrainShare
            self.Share = TrainShare
        elif self.Share is not None:
            Share = self.Share

        if self.loss_type == "mse":
            criterion = torch.nn.MSELoss()
        elif self.loss_type == "soft":
            criterion = torch.nn.MultiLabelSoftMarginLoss()
        elif self.loss_type == "mean":
            criterion = torch.mean
        elif self.loss_type == "norm":
            criterion = torch.norm
        else:
            raise Exception(
                "Loss function type is wrong! Options: 'mse', 'soft', 'mean', or 'norm'"
            )

        self.ha_loss_test_vec = list()
        self.ha_loss_test = None

        NewViews = list()
        G = torch.Tensor(Share)

        for s in range(NumSub):
            net_shape = np.concatenate(([NumVoxel], self.net_shape))
            net = TonysMLP(
                model=net_shape, activation=self.activation, gpu_enable=self.gpu_enable
            )

            if self.optim == "adam":
                optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
            elif self.optim == "sgd":
                optimizer = optim.SGD(net.parameters(), lr=self.learning_rate)
            else:
                raise Exception(
                    "Optimization algorithm is wrong! Options: 'adam' or 'sgd'"
                )

            X = torch.Tensor(views[s])
            net.train()

            for j in range(self.iteration):
                # Send data to GPU
                if self.gpu_enable:
                    X = X.cuda()
                    G = G.cuda()

                optimizer.zero_grad()
                fX = net(X)

                if self.loss_type == "mse" or self.loss_type == "soft":
                    loss = criterion(G, fX)
                else:
                    loss = criterion(G - fX)

                loss.backward()
                optimizer.step()
                sum_loss = loss.data.cpu().numpy()

                if verbose:
                    print(
                        "TEST, UPDATE NETWORK: Iteration {:6d}, Subject {:6d}, loss error: {}".format(
                            j + 1, s + 1, sum_loss
                        )
                    )

            if self.gpu_enable:
                X = X.cuda()

            NewViews.append(net(X).data.cpu().numpy())

        ha_model = GPUHA(Dim=NumFea, regularization=self.regularization)
        ha_model.test(views=NewViews, G=Share, verbose=verbose)
        self.TestFeatures = ha_model.Xtest
        self.TestRuntime = time.time() - tic
        return self.TestFeatures
