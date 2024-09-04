from __future__ import annotations
from argparse import Namespace
from glob import glob
from functools import partial
from typing import Optional, Tuple, Union
import jax
import jax.numpy as np
from jax import random
from flax.training.train_state import TrainState
from jax.scipy.linalg import block_diag
from flax.training import checkpoints
from flax import linen as nn
from orbax import checkpoint
from lob.encoding import Vocab
from lob.lob_seq_model import BatchFullLobPredModel, BatchLobPredModel, BatchPaddedLobPredModel, FullLobPredModel#, ParFullLobPredModel

#from lob.lob_seq_model import BatchLobPredModel
from lob.train_helpers import create_train_state, eval_step, init_Lambda_V_Vinv, prep_batch, cross_entropy_loss, compute_accuracy
from mamba.ssm import init_S5SSM
from mamba.ssm_init import make_DPLR_HiPPO
# from s5.ssm import init_S5SSM
# from s5.ssm_init import make_DPLR_HiPPO
from lob.lobster_dataloader import LOBSTER_Dataset, LOBSTER

import lob.validation_helpers as valh
from constants import TrainArgs



def load_args_from_checkpoint(
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> Namespace:

    """Load arguments from checkpoint"""
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    raw_restored = checkpoints.restore_checkpoint(
        checkpoint_path,
        None,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    args = Namespace(**raw_restored['config'])
    return args


def load_checkpoint(
        state: TrainState,
        path: str,
        config_dict: dict,
        step: Optional[int] = None,
    ) -> TrainState:
    ckpt = {
        'model': state,
        'config': config_dict,
        'metrics': {
            'loss_train': np.nan,
            'loss_val': np.nan,
            'loss_test': np.nan,
            'acc_val': np.nan,
            'acc_test': np.nan,
        }
    }
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    restored = checkpoints.restore_checkpoint(
        path,
        ckpt,
        step=step,
        orbax_checkpointer=orbax_checkpointer
    )
    return restored


def init_train_state(
        args: TrainArgs,
        n_classes: int,
        seq_len: int,
        book_dim: int,
        book_seq_len,
        print_shapes=False
    ) -> Tuple[TrainState, Union[partial[BatchLobPredModel], partial[FullLobPredModel]]]:

    in_dim = n_classes

    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)
    
    ssm_lr = args.ssm_lr_base

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr
    Lambda, V, Vinv = init_Lambda_V_Vinv(args)

    if print_shapes:
        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))
        print("book_seq_len", book_seq_len)
        print("book_dim", book_dim)

    padded = False
    retrieval = False
    speech = False


    ssm_init_fn = init_S5SSM(
        H=args.d_model,
        L=seq_len,
        P=args.ssm_size_base,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        expand_factor=args.expand_factor,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional
    )
    
    if args.use_book_data:
        # if args.num_devices > 1:
        #     model_cls = ParFullLobPredModel
        # else:
        #     model_cls = BatchFullLobPredModel
        
        model_cls = partial(
            # projecting sequence lengths down has appeared better than padding
            BatchFullLobPredModel,
            #BatchPaddedLobPredModel,
            #model_cls,
            args=args,
            ssm=ssm_init_fn,
            book_seq_len=book_seq_len,
            d_output=n_classes,
            d_model=args.d_model,
            d_book=book_dim,
            n_message_layers=args.n_message_layers,  # 2
            n_fused_layers=args.n_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    else:
        if args.num_devices > 1:
            raise NotImplementedError("Message only model not implemented for multi-device training")
        
        model_cls = partial(
            BatchLobPredModel,
            ssm=ssm_init_fn,
            book_seq_len=book_seq_len,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # initialize training state
    state = create_train_state(
        model_cls,
        init_rng,
        padded,
        retrieval,
        use_book_data=args.use_book_data,
        in_dim=in_dim,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
        num_devices=args.num_devices,
    )

    return state, model_cls
