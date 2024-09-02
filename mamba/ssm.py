from functools import partial
import jax
import jax.numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal
from scipy.signal import cont2discrete


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones((Lambda.shape[0], Lambda.shape[1]))
    Delta_ = Delta[..., None] * np.ones((*Delta.shape, Lambda.shape[-1]))
    Lambda_bar = jax.vmap(lambda d: np.exp(Lambda * d))(Delta_)
    B_tilde_ = jnp.expand_dims(B_tilde, axis=1) * np.ones(Lambda_bar.shape)
    B_bar = (1/Lambda * (Lambda_bar-Identity)) * B_tilde_
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(nn.Module):
    Lambda_re_init: jnp.ndarray
    Lambda_im_init: jnp.ndarray
    V: jnp.ndarray
    Vinv: jnp.ndarray
    H: int
    L: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    kernel_size: int = 4
    expand_factor: int = 2
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            L           (int32):     sequence length
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """
        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        self.expanded_P = self.expand_factor * local_P
        # self.expanded_P = self.expand_factor * self.P
        self.input_projB = nn.Dense(self.expanded_P * self.H)
        self.input_projLambda = nn.Dense(self.expanded_P * self.H)
        self.gated_proj = nn.Dense(self.H)
        
        # Convolution layer
        self.conv = nn.Conv(
            features=self.expanded_P * self.H,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        ) 


        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im
        # self.Lambda = jnp.tile(self.Lambda, self.L).reshape((self.L, self.expanded_P))
        

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (self.L, self.expanded_P)  # (sequence length, feature dim, expanded state size)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        self.B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        C_shape = (self.L, self.expanded_P, 2)
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            # TODO: Implement bidirectional implementation
            # if self.bidirectional:
            #     C = self.param("C", C_init, (self.H, 2 * self.P, 2))
            #     self.C_tilde = C[..., 0] + 1j * C[..., 1]

            # else:
            self.C = self.param("C", C_init, C_shape)
            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        else:
            # if self.bidirectional:
            #     self.C1 = self.param("C1",
            #                          lambda rng, shape: init_CV(C_init, rng, shape, self.V),
            #                          C_shape)
            #     self.C2 = self.param("C2",
            #                          lambda rng, shape: init_CV(C_init, rng, shape, self.V),
            #                          C_shape)

            #     C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
            #     C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
            #     self.C_tilde = np.concatenate((C1, C2), axis=-1)

            # else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                C_shape)
            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.H, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        step = jnp.tile(step, (self.L, 1))

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, self.B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, self.B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        # self.B_proj = nn.Dense(local_P * local_P)
        # self.C_proj = nn.Dense(local_P)
        # self.Lambda_proj = nn.Dense(local_P)
        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.L, self.H,))
        self.D_proj = nn.Dense(1)

        self.output_proj = nn.Dense(self.H)


    def __call__(self, input_sequence: jax.typing.ArrayLike):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        # Step 1: Input projection and reshaping
        B_proj = self.input_projB(input_sequence)  # shape: (B, L, expand * d_inner)
        # B_proj = B_proj.reshape((self.L, self.H, self.expanded_P))

        Lambda_proj = self.input_projLambda(input_sequence)  # shape: (B, L, expand * d_inner)
        Lambda_proj = Lambda_proj.reshape((self.L, self.H, self.expanded_P))
        Lambda_act = nn.silu(Lambda_proj)


        # Step 2: Convolution and activation
        B_conv = self.conv(B_proj)  # shape: (B, expand, seq_len, d_inner)
        B_act = nn.silu(B_conv)  # Apply activation function
        B_act = B_act.reshape((self.L, self.H, self.expanded_P))


        # Compute input-dependent B_bar, C_bar, and Lambda_bar
        # B_bar = jnp.einsum('tij,tj->ti', self.B_bar, B_act) # nn.sigmoid(B_conv)
        B_bar = self.B_bar * B_act
        Lambda_bar = nn.softplus(self.Lambda_bar + Lambda_act)

        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_bar, B_bar))

        ys =  jax.vmap(lambda x, c: (c @ x).real)(xs.reshape((self.L, self.expanded_P, self.H)), self.C_tilde)
        # ys = jnp.einsum('lij, li -> lj', xs, self.C_tilde)
        
        # ys =  (self.C_tilde @ xs).real

        # Gating mechanism
        gt = self.gated_proj(input_sequence)
        gt = nn.silu(gt)
        # gt = gt.reshape((self.L, self.H))


        # ys = apply_ssm(Lambda_bar,
        #                B_bar,
        #                self.C_tilde,
        #                input_sequence,
        #                self.conj_sym,
        #                self.bidirectional)

        # Add feedthrough matrix output Du;
        D = self.D_proj(input_sequence)
        # D = jnp.broadcast_to(D, (self.L, self.H, self.expanded_P))
        D = nn.softplus(D)
        # Du = jax.vmap(lambda u: D * u)(input_sequence)
        Du = D * input_sequence

        ht_pre_gate = ys + Du
        # gated multiplication: ℎ𝑡 = (1 − 𝑔𝑡)ℎ𝑡−1 + 𝑔𝑡𝑥t
        ht = (1 - gt) * ht_pre_gate + gt * input_sequence
        return self.output_proj(ht)


def init_S5SSM(H,
               L,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               expand_factor,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   L=L,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   expand_factor=expand_factor,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)
