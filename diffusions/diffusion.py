from functools import partial
from typing import Type, Tuple
from networks.model import Model
from networks.types import Params, InfoDict, PRNGKey, Batch
import flax.linen as nn
import jax.numpy as jnp
import jax


def sample_trunc_normal(rng: PRNGKey, shape: Tuple, threshold: float = 0.2):
    """
    sample truncated normal so that both tails are rejected.
    """
    x_max = jnp.abs(jax.scipy.stats.norm.ppf(threshold / 2))
    x_min = - x_max
    n_samples = jnp.prod(jnp.array(shape)).astype(int)
    samples = jnp.zeros((0,))  # empty for now
    while samples.shape[0] < n_samples:
        rng, key = jax.random.split(rng)
        s = jax.random.normal(key, (n_samples,))
        accepted = s[(s >= x_min) & (s <= x_max)]
        samples = jnp.concatenate((samples, accepted), axis=0)
    samples = samples[:n_samples]
    return samples.reshape(shape)


def rescale_noise_cfg(cfg_noise: jnp.array, cond_noise: jnp.array, guidance_rescale: float = 0.7):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    # skip the batch dimensions, and calculate the noise of all place
    std_cond = cond_noise.std(axis=list(range(1, cond_noise.ndim)), keepdims=True)
    std_cfg = cfg_noise.std(axis=list(range(1, cfg_noise.ndim)), keepdims=True)
    # rescale the results from guidance (fixes overexposure)
    noise_rescaled = cfg_noise * (std_cond / std_cfg + 1e-6)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    cfg_noise = guidance_rescale * noise_rescaled + (1 - guidance_rescale) * cfg_noise
    return cfg_noise


class DDPM(nn.Module):
    noise_predictor: Type[nn.Module]
    time_embedding: Type[nn.Module]
    time_processor: Type[nn.Module]
    cond_embedding: Type[nn.Module] = None

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,  # s = [obs, cond] if it's conditional
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):
        t_ff = self.time_embedding()(time)
        time_suffix = self.time_processor()(t_ff, training=training)

        if self.cond_embedding is not None:
            # last dim gives the class token
            embed_feature = self.cond_embedding()(s[:, -1])  # gives the shape of (B,) array
            s = jnp.concatenate([s[:, :-1], embed_feature], axis=-1)

        reverse_input = jnp.concatenate([s, a, time_suffix], axis=-1)
        return self.noise_predictor()(reverse_input, training=training)


# noise_pred_apply_fn is static since the network structure is immutable
@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler'))
def ddpm_sampler(rng, noise_pred_apply_fn, params, T, observations, alphas, alpha_hats, sample_temperature,
                 repeat_last_step, clip_sampler, prior: jnp.array):
    batch_size = observations.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        input_time = input_time_proto * t
        # noise_model(s, a, time, training=training) in DDPM

        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # re-parameterization of distribution (4) in DDPM paper
        x0_hat = 1 / jnp.sqrt(alpha_hats[t]) * (current_x - jnp.sqrt(1 - alpha_hats[t]) * eps_pred)

        if clip_sampler:
            x0_hat = jnp.clip(x0_hat, -1, 1)

        # equation (7) in DDPM paper, equivalent to (7), here using x0_hat just for clipping
        current_x = 1 / (1 - alpha_hats[t]) * (jnp.sqrt(alpha_hats[t - 1]) * (1 - alphas[t]) * x0_hat +
                                               jnp.sqrt(alphas[t]) * (1 - alpha_hats[t - 1]) * current_x)

        # alpha_1 = 1 / jnp.sqrt(alphas[t])
        # alpha_2 = ((1 - alphas[t]) / (jnp.sqrt(1 - alpha_hats[t])))
        # current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # sigmas_t = jnp.sqrt((1 - alphas[t]) * (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]))
        sigmas_t = jnp.sqrt((1 - alphas[t]))  # both have similar results
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (sigmas_t * z_scaled)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=T)  # unroll = 5

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler'))
def legacy_ddpm_sampler(rng, noise_pred_apply_fn, params, T, observations, alphas, alpha_hats, sample_temperature,
                        repeat_last_step, clip_sampler, prior: jnp.array):
    batch_size = observations.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        input_time = input_time_proto * t
        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        alpha_1 = 1 / jnp.sqrt(alphas[t])
        alpha_2 = ((1 - alphas[t]) / (jnp.sqrt(1 - alpha_hats[t])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (jnp.sqrt((1 - alphas[t])) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=T)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'critic_tar_apply_fn', 'T',
                                   'repeat_last_step', 'clip_sampler'))
def ddpm_sampler_with_q_guidance(rng, noise_pred_apply_fn, params, critic_tar_apply_fn, q_params, guidance_scale,
                                 T, observations, alphas, alpha_hats, sample_temperature,
                                 repeat_last_step, clip_sampler, prior: jnp.array):
    batch_size = observations.shape[0]
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))
    q_grad_fn = jax.vmap(jax.grad(lambda x0, s: critic_tar_apply_fn(q_params, s, x0).mean(axis=0)))

    def fn(input_tuple, t):
        current_x, rng_ = input_tuple
        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)
        input_time = input_time_proto * t
        # noise_model(s, a, time, training=training) in DDPM

        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # q_grad = q_grad_fn(current_x, observations)
        rng_, key_ = jax.random.split(rng_)
        q_grad = q_grad_fn(current_x, observations)

        # q_norm = jnp.abs(critic_tar_apply_fn(q_params, observations, current_x)).mean()
        # q_grad /= q_norm
        # q_grad /= (1e-3 + jnp.abs(q_grad).mean())
        # q_grad /= jnp.linalg.norm(q_grad, axis=-1, keepdims=True)

        eps_pred -= guidance_scale * jnp.sqrt(1 - alpha_hats[t]) * q_grad

        x0_hat = 1 / jnp.sqrt(alpha_hats[t]) * (current_x - jnp.sqrt(1 - alpha_hats[t]) * eps_pred)

        if clip_sampler:
            x0_hat = jnp.clip(x0_hat, -1, 1)

        # equation (7) in DDPM paper, equivalent to (7), here using x0_hat just for clipping
        current_x = 1 / (1 - alpha_hats[t]) * (jnp.sqrt(alpha_hats[t - 1]) * (1 - alphas[t]) * x0_hat +
                                               jnp.sqrt(alphas[t]) * (1 - alpha_hats[t - 1]) * current_x)

        rng_, key_ = jax.random.split(rng_)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # sigmas_t = jnp.sqrt((1 - alphas[t]) * (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]))
        sigmas_t = jnp.sqrt((1 - alphas[t]))  # both have similar results
        # remove the noise of t = 0
        current_x = current_x + (t > 1) * (sigmas_t * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn,
                                    (prior, denoise_key),
                                    jnp.arange(T, 0, -1),  # since alphas <- cat[0, alphas]; betas <- cat[1, betas]
                                    unroll=T)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler',
                                   'ddim_step', 'ddim_eta'))
def ddim_sampler(rng, noise_pred_apply_fn, params, T, observations, alphas, alpha_hats,
                 sample_temperature, repeat_last_step, clip_sampler, prior: jnp.array, ddim_step, ddim_eta=0):
    """
    dim(obs_with_prompt) = dim(obs) + 1, the prompt is one scalar value
    """
    batch_size = observations.shape[0]
    c = T // ddim_step  # jump step
    ddim_time_seq = jnp.concatenate([jnp.arange(T, 0, -c), jnp.array([0])])
    input_time_proto = jnp.ones((*prior.shape[:-1], 1))

    def fn(input_tuple, i):
        # work on the last dim
        current_x, rng_ = input_tuple

        t, prev_t = ddim_time_seq[i], ddim_time_seq[i + 1]

        input_time = input_time_proto * t

        # input_time = jnp.expand_dims(jnp.array([t]).repeat(current_x.shape[0]), axis=1)

        # if guidance_scale > 0:
        #     # use classifier-free guidance when guidance_scale > 1
        #     # treat the last dimension as the class token
        #     # observations is of dim [B*repeats, d_obs]
        #     unc_obs = jnp.concatenate([obs_with_prompt[:, :-1], jnp.zeros((batch_size, 1))], axis=-1)
        #
        #     eps_c = noise_pred_apply_fn(params, obs_with_prompt, current_x, input_time, training=False)
        #     eps_unc = noise_pred_apply_fn(params, unc_obs, current_x, input_time, training=False)
        #     eps_pred = eps_unc + guidance_scale * (eps_c - eps_unc)
        #
        #     eps_pred = rescale_noise_cfg(eps_pred, eps_c, guidance_rescale=guidance_rescale)
        #
        # else:
        eps_pred = noise_pred_apply_fn(params, observations, current_x, input_time, training=False)

        # sigmas_t = ddim_eta * jnp.sqrt((1 - alpha_hats[prev_t]) / (1 - alpha_hats[t]) * (1 - alphas[t]))
        sigmas_t = ddim_eta * jnp.sqrt((1 - alphas[t]))  # both have similar results

        alpha_1 = 1 / jnp.sqrt(alphas[t])
        alpha_2 = jnp.sqrt(1 - alpha_hats[t])
        alpha_3 = jnp.sqrt(1 - alpha_hats[prev_t] - sigmas_t ** 2)

        current_x = alpha_1 * (current_x - alpha_2 * eps_pred) + alpha_3 * eps_pred

        rng_, key_ = jax.random.split(rng_, 2)
        z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
        z_scaled = sample_temperature * z
        current_x = current_x + sigmas_t * z_scaled

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng_), ()

    rng, denoise_key = jax.random.split(rng, 2)
    output_tuple, () = jax.lax.scan(fn, (prior, denoise_key), jnp.arange(len(ddim_time_seq) - 1),
                                    unroll=T)

    for _ in range(repeat_last_step):
        output_tuple, () = fn(output_tuple, 0)

    action_0, rng = output_tuple
    # action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng


@jax.jit
def jit_update_diffusion_model(actor: Model,
                               batch: Batch,
                               rng: PRNGKey,
                               T,
                               alpha_hats) -> Tuple[PRNGKey, Tuple[Model, InfoDict]]:
    rng, t_key, noise_key, tr_key = jax.random.split(rng, 4)
    # learnable t is ranged from 1, 2,...,T corresponding to the indices of alphas
    # assert len(alpha_hat) == T + 1
    t = jax.random.randint(t_key, (batch.actions.shape[0],), 1, T + 1)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch.actions.shape)

    # noisy_actions = jnp.sqrt(alpha_hat[t]) * batch.actions + (1 - jnp.sqrt((alpha_hat[t]))) * eps_sample
    noisy_actions = jnp.sqrt(alpha_hats[t]) * batch.actions + jnp.sqrt(1 - alpha_hats[t]) * eps_sample

    def actor_loss_fn(paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = actor.apply(paras,
                               batch.observations,
                               noisy_actions,
                               t,  # noice that t is ranged from 0, 1, ..., T-1, pay attention to sampling method
                               rngs={'dropout': tr_key},
                               training=True)

        actor_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()

        return actor_loss, {'actor_loss': actor_loss}

    return rng, actor.apply_gradient(actor_loss_fn)
