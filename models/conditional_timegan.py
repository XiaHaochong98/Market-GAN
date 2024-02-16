# -*- coding: UTF-8 -*-

import torch
from sklearn.metrics import accuracy_score

from models.ConditionalSupervisor import *
from models.DiscriminatorNetwork import *
from models.EmbeddingNetwork import *
from models.GeneratorNetwork import *
from models.RecoveryNetwork import *
from models.SupervisorNetwork import *
from models.models_utils import *


class Conditional_TimeGAN(torch.nn.Module):

    def __init__(self, args, dynamic_supervisor_args, label_supervisor_args):
        super(Conditional_TimeGAN, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.dynamic_dim = args.dynamic_dim
        self.label_dim = args.label_dim
        self.args = args

        self.embedder = EmbeddingNetwork(args)
        # add a sub recovery for each condition(D,L)
        # if there is too much conditions we can do clustering for L (check how pm do this)
        self.recovery = RecoveryNetwork(args)
        # add a sub generator for each condition(D,L)
        self.generator = GeneratorNetwork(args)
        # add a sub supervisor for each condition(D,L)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)
        self.history_embedder = HistoryEmbeddingNetwork(args)
        # update num_class,task_name in args to creat latent_dynamic_supervisor_args and latent_label_supervisor_args
        #  task_name='classification'
        # num_class=args.dynamic_dim for latent_dynamic_supervisor and num_class=args.label_dim for latent_label_supervisor
        # copy the namespace args to latent_dynamic_supervisor_args and latent_label_supervisor_args

        latent_dynamic_supervisor_args = copy.deepcopy(args)
        latent_dynamic_supervisor_args.task_name = 'latent_supervision'
        latent_dynamic_supervisor_args.num_class = args.dynamic_dim
        latent_dynamic_supervisor_args.model = args.latent_condtion_supervisor_model
        latent_dynamic_supervisor_args.enc_in = args.hidden_dim
        latent_label_supervisor_args = copy.deepcopy(args)
        latent_label_supervisor_args.task_name = 'latent_supervision'
        latent_label_supervisor_args.num_class = args.label_dim
        latent_label_supervisor_args.enc_in = args.hidden_dim
        latent_label_supervisor_args.model = args.latent_condtion_supervisor_model
        self.latent_dynamic_revovery = Condition_supervisor(latent_dynamic_supervisor_args)
        self.latent_label_recovery = Condition_supervisor(latent_label_supervisor_args)
        # load the condition supervisors for D,L if args.load_supervisors is true
        if args.load_supervisors:
            self.dynamic_supervisor = Condition_supervisor(dynamic_supervisor_args)
            self.label_supervisor = Condition_supervisor(label_supervisor_args)
            if args.is_train:
                self.dynamic_supervisor.load_state_dict(torch.load(f"{args.pretrain_model_path}/dynamic_classifier.pt"))
                self.label_supervisor.load_state_dict(torch.load(f"{args.pretrain_model_path}/label_classifier.pt"))
        else:
            self.dynamic_supervisor = Condition_supervisor(dynamic_supervisor_args)
            self.label_supervisor = Condition_supervisor(label_supervisor_args)

    def _recovery_forward(self, X, T, D=None, L=None, History=None):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features with conditions
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features without conditions
        """
        # Forward Pass
        # concatenate X,D,L
        if History is not None:
            History_embbedding = self.history_embedder(History)

        H = self.embedder(X, T, D=D, L=L, H=History_embbedding)
        X_tilde = self.recovery(H, T, D=D, L=L, H=History_embbedding)

        # condition supervision loss
        if D is not None:
            # 1. Dynamics
            D_hat = self.dynamic_supervisor(X_tilde)
            # calculate the loss for mutil-label classification
            # print('D_hat',D_hat.shape,'D',D.shape)
            D_loss = torch.nn.functional.binary_cross_entropy(D_hat, D)

            dynamic_prediction_latent = self.latent_dynamic_revovery(H, T=T, H=History_embbedding)
            # print('dynamic_prediction_latent',dynamic_prediction_latent.shape,'D',D.shape)
            Dynamics_loss_latent = torch.nn.functional.binary_cross_entropy(dynamic_prediction_latent, D)

        if L is not None:
            # 2. Label
            L_hat = self.label_supervisor(X_tilde)
            # calculate the loss for mutil-label classification
            L_loss = torch.nn.functional.binary_cross_entropy(L_hat, L)

            label_prediction_latent = self.latent_label_recovery(H, T=T, H=History_embbedding)

            Label_loss_latent = torch.nn.functional.binary_cross_entropy(label_prediction_latent, L)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T, D=D, L=L, H=History_embbedding)
        # print(H.shape,H_hat_supervise.shape)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :],
            H[:, 1:, :]
        )  # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S
        if D is not None or L is not None:
            return E_loss, E_loss0, E_loss_T0, D_loss, L_loss, G_loss_S, Dynamics_loss_latent, Label_loss_latent
        else:
            return E_loss, E_loss0, E_loss_T0, G_loss_S

    def _supervisor_forward(self, X, T, D, L, History=None):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        if History is not None:
            History_embbedding = self.history_embedder(History)
        H = self.embedder(X, T, D=D, L=L, H=History_embbedding)
        H_hat_supervise = self.supervisor(H, T, D=D, L=L, H=History_embbedding)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output

        return S_loss

    def _discriminator_forward(self, X, T, Z, D=None, L=None, History=None, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        if History is not None:
            History_embbedding = self.history_embedder(History)
        H = self.embedder(X, T, D=D, L=L, H=History_embbedding).detach()

        # Generator
        E_hat = self.generator(Z, T, D=D, L=L, H=History_embbedding).detach()
        H_hat = self.supervisor(E_hat, T, D=D, L=L, H=History_embbedding).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T, D=D, L=L, H=History_embbedding)  # Encoded original data
        Y_fake = self.discriminator(H_hat, T, D=D, L=L, H=History_embbedding)  # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T, D=D, L=L, H=History_embbedding)  # Output of generator

        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss, D_loss_real, D_loss_fake, D_loss_fake_e

    def _generator_forward(self, X, T, Z, D=None, L=None, History=None, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input and the conditonal input(dynamic and label)
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        if History is not None:
            History_embbedding = self.history_embedder(History)
        H = self.embedder(X, T, D=D, L=L, H=History_embbedding)
        H_hat_supervise = self.supervisor(H, T, D=D, L=L, H=History_embbedding)

        # Generator Forward Pass
        E_hat = self.generator(Z, T, D=D, L=L, H=History_embbedding)
        H_hat = self.supervisor(E_hat, T, D=D, L=L, H=History_embbedding)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T, D=D, L=L, H=History_embbedding)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T, D=D, L=L, H=History_embbedding)  # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T, D=D, L=L, H=History_embbedding)  # Output of generator

        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])  # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(
            torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        if self.args.conditional:
            G_loss = G_loss_U + gamma * G_loss_U_e + 10 * G_loss_S + 10 * G_loss_V
        else:
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V
        # condition supervision loss
        # 1. Dynamics
        if D is not None:
            D_hat = self.dynamic_supervisor(X_hat)
            D_loss = torch.nn.functional.binary_cross_entropy(D_hat, D)
            Dynamic_hat_s = self.latent_dynamic_revovery(E_hat, T=T, H=History_embbedding)
            D_loss_latent = torch.nn.functional.mse_loss(Dynamic_hat_s, D)
        if L is not None:
            # 2. Label
            L_hat = self.label_supervisor(X_hat)
            L_loss = torch.nn.functional.binary_cross_entropy(L_hat, L)
            Label_hat_s = self.latent_label_recovery(E_hat, T=T, H=History_embbedding)
            L_loss_latent = torch.nn.functional.mse_loss(Label_hat_s, L)
        if D is not None or L is not None:
            return G_loss, D_loss, L_loss, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, D_loss_latent, L_loss_latent
        else:
            return G_loss, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V

    def _pretrain_dynamics_supervisor_foward(self, X, D):
        # X shape: (batch_size, seq_len, feature_dim)
        train_p = self.dynamic_supervisor(X)
        Dynamics_loss = torch.nn.functional.binary_cross_entropy(train_p, D)
        result = tensor_threshold(train_p, 1)
        acc = accuracy_score(D.detach().cpu().numpy(), result)

        return Dynamics_loss, acc

    def _pretrain_label_supervisor_foward(self, X, L):

        train_p = self.label_supervisor(X)
        # multi-label one-hot prediction loss
        Label_loss = torch.nn.functional.binary_cross_entropy(train_p, L)
        # using the max value as the one-hot label
        result = tensor_threshold(train_p, 1)
        acc = accuracy_score(L.detach().cpu().numpy(), result)

        return Label_loss, acc

    def _inference(self, Z, T, D=None, L=None, History=None):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        if History is not None:
            History_embbedding = self.history_embedder(History)
        E_hat = self.generator(Z, T, D=D, L=L, H=History_embbedding)

        H_hat = self.supervisor(E_hat, T, D=D, L=L, H=History_embbedding)
        X_hat = self.recovery(H_hat, T, D=D, L=L, H=History_embbedding)
        return X_hat

    def forward(self, X, obj, T=None, Z=None, D=None, L=None, H=None, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
            - D: the dynamic information (B, H, 1)
            - L: the one-hot label information (B, H, L)
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")
            # print('tensors X to device')
            X = torch.FloatTensor(X)
            X = X.to(self.device)
        if D is not None:
            # print('tensors D to device')
            D = torch.FloatTensor(D)
            D = D.to(self.device)
        if L is not None:
            # print('tensors L to device')
            L = torch.FloatTensor(L)
            L = L.to(self.device)
        if H is not None:
            # print('tensors H to device')
            H = torch.FloatTensor(H)
            H = H.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T, D, L, H)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T, D, L, H)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z, D=D, L=L, History=H)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, T, Z, D, L, H)

            return loss

        elif obj == "inference":

            X_hat = self._inference(Z, T, D, L, H)
            X_hat = X_hat.cpu().detach()

            return X_hat

        elif obj == "pretrain_dynamic_supervisor":
            if D is None:
                raise ValueError("`D` is not given")
            dynamics_loss, acc = self._pretrain_dynamics_supervisor_foward(X, D)
            return dynamics_loss, acc

        elif obj == "pretrain_label_supervisor":
            if L is None:
                raise ValueError("`L` is not given")
            label_loss, acc = self._pretrain_label_supervisor_foward(X, L)
            return label_loss, acc

        else:
            raise ValueError(
                "`obj` should be either `autoencoder`, `supervisor`, `generator`,`discriminator`,`inference`, `pretrain_dynamic_supervisor`, `pretrain_label_supervisor`")

        return loss
