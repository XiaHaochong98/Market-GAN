# -*- coding: UTF-8 -*-
# Local modules
import os
import sys

sys.path.append(os.path.abspath('../'))
from typing import Dict, Union
from utils.util import *
# 3rd party modules
import torch
# Self-written modules
from dataset.dataset import ConditionalTimeGANDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


def embedding_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        history_e_opt: torch.optim.Optimizer,
        latent_dynamic_supervisor_opt: torch.optim.Optimizer,
        latent_label_supervisor_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None
) -> None:
    """The training loop for the embedding and recovery functions
    """

    # first train the embedder and recovery and history embedding
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for loader in dataloader:

            # mask the conditions if not using conditional model
            if args.conditional:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
            else:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
                D_mb = None
                L_mb = None
                H_mb = None
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            if args.conditional:
                _, E_loss0, E_loss_T0, D_loss, L_loss, G_loss_S, Dynamics_loss_latent, Label_loss_latent = model(X=X_mb,
                                                                                                                 T=T_mb,
                                                                                                                 D=D_mb,
                                                                                                                 L=L_mb,
                                                                                                                 H=H_mb,
                                                                                                                 obj="autoencoder")
            else:
                _, E_loss0, E_loss_T0, G_loss_S = model(X=X_mb, T=T_mb, D=D_mb, L=L_mb, H=H_mb, obj="autoencoder")
            # loss = np.sqrt(E_loss_T0.item())
            loss = E_loss_T0.item()

            # Backward Pass
            if args.conditional:
                condtional_loss = D_loss * args.dynamic_weight * 0.2 + L_loss * args.label_weight * 0.2
                Total_loss = E_loss0 + condtional_loss
                # Total_loss = E_loss0
            else:
                Total_loss = E_loss0
            Total_loss.backward()

            # Update model parameters for embedder, recovery and history embedding
            if Total_loss > 0.15 or condtional_loss * 5 > 0.6:
                e_opt.step()
                r_opt.step()
                history_e_opt.step()

            # Log loss
            if args.conditional:
                logger.set_description(
                    f"Epoch: {epoch}, Loss: {loss:.4f}, D_loss: {D_loss.item():.4f}, L_loss: {L_loss.item():.4f}")
                if writer:
                    writer.add_scalar(
                        "Embedding/Loss:",
                        loss,
                        epoch
                    )
                    writer.add_scalar(
                        "Embedding/D_loss:",
                        D_loss.item(),
                        epoch
                    )
                    writer.add_scalar(
                        "Embedding/L_loss:",
                        L_loss.item(),
                        epoch
                    )
                    writer.flush()
            else:
                logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
                if writer:
                    writer.add_scalar(
                        "Embedding/Loss:",
                        loss,
                        epoch
                    )
                    writer.flush()
    # print loss
    if args.conditional:
        print("Loss: {:.4f}, D_loss: {:.4f}, L_loss: {:.4f}".format(loss, D_loss.item(), L_loss.item()))

    # train embber and the latent conditional supervisor
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        for loader in dataloader:

            # mask the conditions if not using conditional model
            if args.conditional:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
            else:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
                D_mb = None
                L_mb = None
                H_mb = None
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            if args.conditional:
                _, E_loss0, E_loss_T0, D_loss, L_loss, G_loss_S, Dynamics_loss_latent, Label_loss_latent = model(X=X_mb,
                                                                                                                 T=T_mb,
                                                                                                                 D=D_mb,
                                                                                                                 L=L_mb,
                                                                                                                 H=H_mb,
                                                                                                                 obj="autoencoder")
            else:
                _, E_loss0, E_loss_T0, G_loss_S = model(X=X_mb, T=T_mb, D=D_mb, L=L_mb, H=H_mb, obj="autoencoder")
            # loss = np.sqrt(E_loss_T0.item())
            loss = E_loss_T0.item()

            # Backward Pass
            if args.conditional:
                # Total_loss = E_loss0+ Dynamics_loss_latent*args.dynamic_weight + Label_loss_latent*args.label_weight
                # Total_loss=E_loss0*10+Dynamics_loss_latent + Label_loss_latent
                Total_loss = Dynamics_loss_latent + Label_loss_latent
            else:
                Total_loss = E_loss0
            Total_loss.backward()

            # Update model parameters for embedder, recovery and history embedding
            # e_opt.step()
            # r_opt.step()
            # history_e_opt.step()
            latent_dynamic_supervisor_opt.step()
            # latent_dynamic_supervisor_opt.step()
            latent_label_supervisor_opt.step()
            # latent_label_supervisor_opt.step()

            # Log loss for final batch of each epoch (29 iters)
            if args.conditional:
                logger.set_description(
                    f"Epoch: {epoch}, Loss: {loss:.4f}, Dynamics_loss_latent: {Dynamics_loss_latent.item():.4f}, Label_loss_latent: {Label_loss_latent.item():.4f}")
                if writer:
                    writer.add_scalar(
                        "Embedding/Loss:",
                        loss,
                        epoch
                    )
                    writer.add_scalar(
                        "Embedding/Dynamics_loss_latent:",
                        Dynamics_loss_latent.item(),
                        epoch
                    )
                    writer.add_scalar(
                        "Embedding/Label_loss_latent:",
                        Label_loss_latent.item(),
                        epoch
                    )
                    writer.flush()
            else:
                logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
                if writer:
                    writer.add_scalar(
                        "Embedding/Loss:",
                        loss,
                        epoch
                    )
                    writer.flush()
    # print loss
    if args.conditional:
        print("Loss: {:.4f}, D_loss: {:.4f}, L_loss: {:.4f}".format(loss, D_loss.item(), L_loss.item()))


def supervisor_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        s_opt: torch.optim.Optimizer,
        g_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None
) -> None:
    """The training loop for the supervisor function
    """
    # supervisor training
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, S_loss: 0")
    for epoch in logger:
        for loader in dataloader:
            if args.conditional:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
            else:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
                D_mb = None
                L_mb = None
                H_mb = None
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            if args.conditional:
                S_loss = model(X=X_mb, T=T_mb, Z=None, D=D_mb, L=L_mb, H=H_mb, obj="supervisor")
            else:
                S_loss = model(X=X_mb, T=T_mb, Z=None, D=D_mb, L=L_mb, H=H_mb, obj="supervisor")

            # Backward Pass
            # S_loss.backward()
            S_loss.backward()
            # loss = np.sqrt(S_loss.item())
            S_loss = S_loss.item()

            # Update model parameters
            s_opt.step()
            # Log loss
            logger.set_description(f"Epoch: {epoch}, S_loss: {S_loss:.4f}")
            if writer:
                writer.add_scalar(
                    "Supervisor/S_loss:",
                    S_loss,
                    epoch
                )
                writer.flush()


def joint_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        s_opt: torch.optim.Optimizer,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        latent_dynamic_supervisor_opt: torch.optim.Optimizer,
        latent_label_supervisor_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
) -> None:
    """The training loop for training the model altogether
    """

    if args.conditional:
        print('pretrain generator with latent condition supervision')
        # generator pretrain with latent condition supervision
        logger = trange(
            args.gan_epochs,
            desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
        )
        for epoch in logger:
            for loader in tqdm(dataloader, position=1, desc="dataloader", leave=False, colour='red',
                               ncols=80):
                if args.conditional:
                    X_mb, T_mb, D_mb, L_mb, H_mb = loader
                ## Generator Training
                # print("generator training time:",2+addtional_generator_training)
                # Random Generator
                # Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
                # Z_mb = torch.rand((D_mb.shape[0], args.max_seq_len, args.Z_dim))
                Z_mb = torch.rand((X_mb.shape[0], args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss, dynmaic_loss_g, label_loss_g, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, D_loss_latent, L_loss_latent = model(
                    X=X_mb, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb, H=H_mb, obj="generator")
                Total_G_loss = D_loss_latent * args.dynamic_weight + L_loss_latent * args.label_weight + 10 * G_loss_V
                D_loss_latent = D_loss_latent.item()
                L_loss_latent = L_loss_latent.item()
                G_loss_V = G_loss_V.item()
                Total_G_loss.backward()

                # Update model parameters
                g_opt.step()
                logger.set_description(
                    f"Epoch: {epoch}, D_loss_latent: {D_loss_latent:.4f}, L_loss_latent: {L_loss_latent:.4f}, G_loss_V: {G_loss_V:.4f}")
                if writer:
                    writer.add_scalar(
                        'Joint/D_loss_latent:',
                        D_loss_latent,
                        epoch)
                    writer.add_scalar(
                        'Joint/L_loss_latent:',
                        L_loss_latent,
                        epoch)
                    writer.add_scalar(
                        'Joint/G_loss_V:',
                        G_loss_V,
                        epoch)

                    writer.flush()
    print(f'Generator pretrain with latent condition supervision finished')
    print(f'D_loss_latent: {D_loss_latent:.4f}, L_loss_latent: {L_loss_latent:.4f}, G_loss_V: {G_loss_V:.4f}')

    # train generator with discriminator

    logger = trange(
        args.gan_epochs,
        desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
    )
    print("generator_strengthen_factor: ", args.generator_strengthen_factor)
    for epoch in logger:
        for loader in tqdm(dataloader, position=1, desc="dataloader", leave=False, colour='red',
                           ncols=80):
            if args.conditional:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
            else:
                X_mb, T_mb, D_mb, L_mb, H_mb = loader
                D_mb = None
                L_mb = None
                H_mb = None
            if epoch % args.generator_strengthen_factor == 0:
                addtional_generator_training = 1
            else:
                addtional_generator_training = 0
            ## Generator Training
            for _ in range(2 + addtional_generator_training):
                # print("generator training time:",2+addtional_generator_training)
                # Random Generator
                # Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
                # Z_mb = torch.rand((D_mb.shape[0], args.max_seq_len, args.Z_dim))
                Z_mb = torch.rand((X_mb.shape[0], args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                if args.conditional:
                    G_loss, dynmaic_loss_g, label_loss_g, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, D_loss_latent, L_loss_latent = model(
                        X=X_mb, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb, H=H_mb, obj="generator")
                    Dynmaic_loss_g = dynmaic_loss_g.item()
                    Label_loss_g = label_loss_g.item()
                    D_loss_latent = D_loss_latent.item()
                    L_loss_latent = L_loss_latent.item()
                else:
                    G_loss, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V = model(X=X_mb, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb,
                                                                             H=H_mb, obj="generator")
                if args.conditional:
                    Total_G_loss = G_loss + dynmaic_loss_g * args.dynamic_weight + label_loss_g * args.label_weight
                else:
                    Total_G_loss = G_loss
                Total_G_loss.backward()
                G_loss_S_1 = G_loss_S.item()
                G_loss_U = G_loss_U.item()
                G_loss_U_e = G_loss_U_e.item()
                G_loss_V = G_loss_V.item()
                # G_loss = np.sqrt(G_loss.item())
                G_loss = Total_G_loss.item()

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                if args.conditional:
                    E_loss, _, E_loss_T0, dynmaic_loss_e, label_loss_e, G_loss_S, Dynamics_loss_latent, Label_loss_latent = model(
                        X=X_mb, T=T_mb, D=D_mb, L=L_mb, H=H_mb, obj="autoencoder")
                    # dynmaic_loss_e.backward()
                    # label_loss_e.backward()
                    Dynamic_loss_e = dynmaic_loss_e.item()
                    Label_loss_e = label_loss_e.item()
                else:
                    E_loss, _, E_loss_T0, G_loss_S = model(X=X_mb, T=T_mb, D=D_mb, L=L_mb, H=H_mb, obj="autoencoder")
                if args.conditional:
                    Total_E_loss = E_loss + dynmaic_loss_e * args.dynamic_weight + label_loss_e * args.label_weight + Dynamics_loss_latent * args.dynamic_weight + Label_loss_latent * args.label_weight
                else:
                    Total_E_loss = E_loss
                Total_E_loss.backward()
                G_loss_S_2 = G_loss_S.item()
                E_loss_T0 = E_loss_T0.item()
                # E_loss = np.sqrt(E_loss.item())
                E_loss = Total_E_loss.item()
                # Update model parameters
                e_opt.step()
                r_opt.step()
                if args.conditional:
                    latent_dynamic_supervisor_opt.step()
                    latent_label_supervisor_opt.step()

            # Random Generator
            # Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))
            # Z_mb = torch.rand((D_mb.shape[0], args.max_seq_len, args.Z_dim))
            Z_mb = torch.rand((X_mb.shape[0], args.max_seq_len, args.Z_dim))
            ## Discriminator Training
            model.zero_grad()
            # Forward Pass
            if args.conditional:
                D_loss, D_loss_real, D_loss_fake, D_loss_fake_e = model(X=X_mb, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb, H=H_mb,
                                                                        obj="discriminator")
            else:
                D_loss, D_loss_real, D_loss_fake, D_loss_fake_e = model(X=X_mb, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb, H=H_mb,
                                                                        obj="discriminator")
            D_loss_real = D_loss_real.item()
            D_loss_fake = D_loss_fake.item()
            D_loss_fake_e = D_loss_fake_e.item()
            # Check Discriminator loss
            if D_loss > args.dis_thresh:
                # Backward Pass
                D_loss.backward()
                # Update model parameters
                d_opt.step()
            D_loss = D_loss.item()
            if args.conditional:
                logger.set_description(
                    f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}, Dyn_e: {Dynamic_loss_e:.4f}, Lab_e: {Label_loss_e:.4f},D_loss_s: {D_loss_latent:.4f}, L_loss_s: {L_loss_latent:.4f}, Dyn_g: {Dynmaic_loss_g:.4f}, Lab_g: {Label_loss_g:.4f}, E_T0: {E_loss_T0:.4f}, G_S_1: {G_loss_S_1:.4f}, G_S_2: {G_loss_S_2:.4f}, G_U: {G_loss_U:.4f}, G_U_e: {G_loss_U_e:.4f}, G_V: {G_loss_V:.4f}, D_real: {D_loss_real:.4f}, D_fake: {D_loss_fake:.4f}, D_fake_e: {D_loss_fake_e:.4f}"
                )
            else:
                logger.set_description(
                    f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}, E_T0: {E_loss_T0:.4f}, G_S_1: {G_loss_S_1:.4f}, G_S_2: {G_loss_S_2:.4f}, G_U: {G_loss_U:.4f}, G_U_e: {G_loss_U_e:.4f}, G_V: {G_loss_V:.4f}, D_real: {D_loss_real:.4f}, D_fake: {D_loss_fake:.4f}, D_fake_e: {D_loss_fake_e:.4f}"
                )
            if writer:
                writer.add_scalar(
                    'Joint/Embedding_Loss:',
                    E_loss,
                    epoch
                )
                writer.add_scalar(
                    'Joint/Generator_Loss:',
                    G_loss,
                    epoch
                )
                writer.add_scalar(
                    'Joint/Discriminator_Loss:',
                    D_loss,
                    epoch
                )
                if args.conditional:
                    writer.add_scalar(
                        'Joint/Dynamic_loss_e:',
                        Dynamic_loss_e,
                        epoch
                    )
                    writer.add_scalar(
                        'Joint/Label_loss_e:',
                        Label_loss_e,
                        epoch
                    )
                    writer.add_scalar(
                        'Joint/Dynamic_loss_g:',
                        Dynmaic_loss_g,
                        epoch
                    )
                    writer.add_scalar(
                        'Joint/Label_loss_g:',
                        Label_loss_g,
                        epoch
                    )
                writer.flush()
    # print the final loss
    if args.conditional:
        print(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}, Dyn_e: {Dynamic_loss_e:.4f}, Lab_e: {Label_loss_e:.4f},D_loss_s: {D_loss_latent:.4f}, L_loss_s: {L_loss_latent:.4f}, Dyn_g: {Dynmaic_loss_g:.4f}, Lab_g: {Label_loss_g:.4f}, E_T0: {E_loss_T0:.4f}, G_S_1: {G_loss_S_1:.4f}, G_S_2: {G_loss_S_2:.4f}, G_U: {G_loss_U:.4f}, G_U_e: {G_loss_U_e:.4f}, G_V: {G_loss_V:.4f}, D_real: {D_loss_real:.4f}, D_fake: {D_loss_fake:.4f}, D_fake_e: {D_loss_fake_e:.4f}"
            )


def supervisors_pretain_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        dynamic_supervisor_opt: torch.optim.Optimizer,
        label_supervisor_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None
):
    """The training loop for the supervisor function
    """
    logger = trange(args.pretrain_epochs, desc=f"Epoch: 0, Dynamics_loss: 0,Dynamics_acc:0, Label_loss: 0,Label_acc:0")
    for epoch in logger:
        for X_mb, T_mb, D_mb, L_mb, H_mb in tqdm(dataloader, position=1, desc="dataloader", leave=False, colour='red',
                                                 ncols=80):
            # Reset gradients
            model.zero_grad()
            # print('forward')
            # Forward Pass
            dynamic_loss, d_acc = model(X=X_mb, D=D_mb, obj="pretrain_dynamic_supervisor")
            label_loss, l_acc = model(X=X_mb, L=L_mb, obj="pretrain_label_supervisor")
            # print('backward')
            # Backward Pass
            dynamic_loss.backward()
            label_loss.backward()
            d_loss = dynamic_loss.item()
            l_loss = label_loss.item()
            # print('step')
            # Update model parameters
            dynamic_supervisor_opt.step()
            label_supervisor_opt.step()

            # Log loss
            logger.set_description(
                f"Epoch: {epoch}, Dynamics_loss: {d_loss:.4f},Dynamics_acc:{d_acc:.4f}, Label_loss: {l_loss:.4f},Label_acc:{l_acc:.4f}")
            if writer:
                writer.add_scalar(
                    "Pretrain_dynamic_supervisor/Loss:",
                    d_loss,
                    epoch
                )
                writer.add_scalar(
                    "Pretrain_label_supervisor/Loss:",
                    l_loss,
                    epoch
                )
                writer.flush()


def conditional_timegan_trainer(model, data, time, args, dynamics=None, labels=None, history=None):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
        - dynamics (numpy.ndarray): The dynamic for the model to be conditioned on
        - labels (numpy.ndarray): The label for the model to be conditioned on
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    # Initialize TimeGAN dataset and dataloader
    conditional_dataset = ConditionalTimeGANDataset(data=data, time=time, dynamic=dynamics, label=labels,
                                                    history=history)
    # print all attributes of the dataset
    # print('H',conditional_dataset.H)
    conditional_dataloader = torch.utils.data.DataLoader(
        dataset=conditional_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    dataloader = conditional_dataloader
    # if args.conditional:
    #     dataset=conditional_dataset
    #     dataloader=conditional_dataloader
    # else:
    #     dataset = TimeGANDataset(data, time)
    #     dataloader = torch.downstream_utils.data.DataLoader(
    #         dataset=dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False
    #     )

    model.to(args.device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)
    history_e_opt = torch.optim.Adam(model.history_embedder.parameters(), lr=args.learning_rate)
    latent_dynamic_supervisor_opt = torch.optim.Adam(model.latent_dynamic_revovery.parameters(), lr=args.learning_rate)
    latent_label_supervisor_opt = torch.optim.Adam(model.latent_label_recovery.parameters(), lr=args.learning_rate)
    dynamic_supervisor_opt = torch.optim.Adam(model.dynamic_supervisor.parameters(), lr=args.learning_rate)
    label_supervisor_opt = torch.optim.Adam(model.label_supervisor.parameters(), lr=args.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))
    if args.conditional:
        if args.load_supervisors:
            print(f"\n Use Loading Supervisors from {args.pretrain_model_path}")
        else:
            print("\nStart supervisor pretraining")
            supervisors_pretain_trainer(
                model=model,
                dataloader=conditional_dataloader,
                dynamic_supervisor_opt=dynamic_supervisor_opt,
                label_supervisor_opt=label_supervisor_opt,
                args=args,
                writer=writer
            )

            torch.save(model.dynamic_supervisor.state_dict(), f"{args.model_path}/dynamic_classifier.pt")
            print(f"\nDynamic_classifier saved at path: {args.model_path}/dynamic_classifier.pt")
            torch.save(model.label_supervisor.state_dict(), f"{args.model_path}/label_classifier.pt")
            print(f"\nLabel_classifier saved at path: {args.model_path}/label_classifier.pt")

    print("\nStart Embedding Network Training")
    embedding_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        history_e_opt=history_e_opt,
        latent_dynamic_supervisor_opt=latent_dynamic_supervisor_opt,
        latent_label_supervisor_opt=latent_label_supervisor_opt,
        args=args,
        writer=writer
    )

    print("\nStart Training with Supervised Loss Only")
    supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        args=args,
        writer=writer
    )

    print("\nStart Joint Training")
    joint_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        latent_dynamic_supervisor_opt=latent_dynamic_supervisor_opt,
        latent_label_supervisor_opt=latent_label_supervisor_opt,
        args=args,
        writer=writer,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.model_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/model.pt")
    print(f"\nSaved at path: {args.model_path}")


def conditional_timegan_condition_prediction(model, X_mb, H_mb, D_mb, L_mb, args):
    """The evaluation procedure for the conditions"""
    # Load model for evaluation
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    # with open(f"{args.model_path}/args.pickle", "rb") as fb:
    #     args = torch.load(fb)

    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        args_old = torch.load(fb)
    args_old.device = args.device
    args = args_old

    # model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))

    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    print("\n Evaluate conditions...")
    conditional_dataset = ConditionalTimeGANDataset(data=X_mb, time=D_mb, dynamic=D_mb, label=L_mb, history=H_mb)
    conditional_dataloader = torch.utils.data.DataLoader(
        dataset=conditional_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    with torch.no_grad():
        d_acc_list = []
        l_acc_list = []
        d_loss_list = []
        l_loss_list = []
        for X_mb, T_mb, D_mb, L_mb, H_mb in tqdm(conditional_dataloader, position=1, desc="dataloader", leave=False,
                                                 colour='red',
                                                 ncols=80):
            dynamic_loss, d_acc = model(X=X_mb, D=D_mb, obj="pretrain_dynamic_supervisor")
            label_loss, l_acc = model(X=X_mb, L=L_mb, obj="pretrain_label_supervisor")
            d_acc_list.append(d_acc)
            l_acc_list.append(l_acc)
            d_loss_list.append(dynamic_loss)
            l_loss_list.append(label_loss)
        # print(f"Dynamic Accuracy: {d_acc}", f"Label Accuracy: {l_acc}")

    return sum(d_acc_list) / len(d_acc_list), sum(l_acc_list) / len(l_acc_list), sum(d_loss_list) / len(
        d_loss_list), sum(l_loss_list) / len(l_loss_list)


def conditional_timegan_generator(model, T, args, dynamics=None, labels=None, history=None):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
        - dynamics (List[int]): The dynamics to be generated on
        - labels (List[int]): The labels to be generated on
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model directory not found...")
    batch_size = args.batch_size
    # Load arguments and model
    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        args_old = torch.load(fb)
    args_old.device = args.device
    args = args_old

    # model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
    #
    # print("\nGenerating Data...")
    # # Initialize model to evaluation mode and run without gradients
    # model.to(args.device)
    # print('model on device',args.device)

    # divide the data into batches
    # Initialize TimeGAN dataset and dataloader
    Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))
    conditional_dataset = ConditionalTimeGANDataset(data=Z, time=T, dynamic=dynamics, label=labels, history=history)
    # print all attributes of the dataset
    # print('H',conditional_dataset.H)
    print('generation batch_size is', batch_size)
    conditional_dataloader = torch.utils.data.DataLoader(
        dataset=conditional_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    generated_data_list = []
    model.eval()
    with torch.no_grad():
        # Generate fake data
        for Z_mb, T_mb, D_mb, L_mb, H_mb in tqdm(conditional_dataloader, position=1, desc="dataloader", leave=False,
                                                 colour='red',
                                                 ncols=80):
            generated_data = model(X=None, T=T_mb, Z=Z_mb, D=D_mb, L=L_mb, H=H_mb, obj="inference")
            generated_data_list.append(generated_data)
        # merge the generated data
        generated_data = torch.cat(generated_data_list, dim=0)

    return generated_data.numpy()
