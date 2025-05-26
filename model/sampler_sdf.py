from typing import Optional, List, Union
import numpy as np
import torch
from labml import monit
from .latent_diffusion import LatentDiffusion

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class DiffusionSampler:
    """
    ## Base class for sampling algorithms
    """
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model
        # Get number of steps the model was trained with $T$
        self.n_steps = model.n_steps


class SDFSampler(DiffusionSampler):
    """
    ## DDPM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDPM samples images by repeatedly removing noise by sampling step by step from
    $p_\theta(x_{t-1} | x_t)$,

    \begin{align}

    p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big) \\

    \mu_t(x_t, t) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\

    \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t \\

    x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta \\

    \end{align}
    """

    model: LatentDiffusion

    def __init__(
        self,
        model: LatentDiffusion,
        max_l,
        h,
        is_autocast=False,
        is_show_image=False,
        device=None,
        debug_mode=False
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # selected time steps ($\tau$) $1, 2, \dots, T$
        # self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)
        self.tau = torch.tensor([13, 53, 116, 193, 310, 443, 587, 730, 845, 999], device=self.device) # torch.tensor([999, 845, 730, 587, 443, 310, 193, 116, 53, 13])
        # self.tau = torch.tensor(np.asarray(list(range(self.n_steps)), dtype=np.int32), device=self.device)
        self.used_n_steps = len(self.tau)

        self.is_show_image = is_show_image

        self.autocast = torch.cuda.amp.autocast(enabled=is_autocast)

        self.out_channel = self.model.eps_model.out_channels
        self.max_l = max_l
        self.h = h
        self.debug_mode = debug_mode
        self.guidance_scale = 7.5
        self.guidance_rescale = 0.7

        # now, we set the coefficients
        with torch.no_grad():
            # $\bar\alpha_t$
            self.alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            self.alpha_bar_prev = torch.cat([self.alpha_bar.new_tensor([1.]), self.alpha_bar[:-1]])
            # $\sigma_t$ in DDIM
            self.sigma_ddim = torch.sqrt((1-self.alpha_bar_prev)/(1-self.alpha_bar)*(1-self.alpha_bar/self.alpha_bar_prev)) # DDPM noise schedule

            # $\frac{1}{\sqrt{\bar\alpha}}$
            self.one_over_sqrt_alpha_bar = 1 / (self.alpha_bar ** 0.5)
            # $\frac{\sqrt{1-\bar\alpha}}{\sqrt{\bar\alpha}}$
            self.sqrt_1m_alpha_bar_over_sqrt_alpha_bar = (1 - self.alpha_bar)**0.5 / self.alpha_bar**0.5

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = self.alpha_bar ** 0.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1 - self.alpha_bar) ** 0.5

    def get_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        background_cond: Optional[torch.Tensor],
    ):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param background_cond: background condition
        """

        if background_cond is not None:
            x = torch.cat([x, background_cond], 1) if background_cond is not None else x

        e_t = self.model(x,t)
        return e_t

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        background_cond: Optional[torch.Tensor],
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.,
        same_noise_all_measure: bool = False,
        X0EditFunc = None,
        use_classifier_free_guidance = False,
        use_melody = False,
        device = "cpu",
    ):
        print("p_sample")
        """
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        """
        # Get current tau_i and tau_{i-1}
        tau_i = self.tau[t]
        step_tau_i = self.tau[step]

        # Get $\epsilon_\theta$
        with self.autocast:
            if use_classifier_free_guidance:
                if use_melody:
                    assert background_cond.shape[1] == 6 # chd_onset, chd_sustain, null_chd_onset, null_chd_sustain, melody_onset, melody_sustain
                    null_melody = -torch.ones_like(background_cond[:,4:,:,:])
                    null_background_cond = torch.cat([background_cond[:,2:4,:,:], null_melody], axis=1)
                    real_background_cond = torch.cat([background_cond[:,:2,:,:], background_cond[:,4:,:,:]], axis=1)

                    e_tau_i_null = self.get_eps(x, tau_i, null_background_cond)
                    e_tau_i_real = self.get_eps(x, tau_i, real_background_cond)
                    e_tau_i = e_tau_i_null + self.guidance_scale * (e_tau_i_real-e_tau_i_null)
                    if self.guidance_rescale > 0:
                        e_tau_i = rescale_noise_cfg(e_tau_i, e_tau_i_real, guidance_rescale=self.guidance_rescale)
                else:
                    assert background_cond.shape[1] == 4 # chd_onset, chd_sustain, null_chd_onset, null_chd_sustain
                    null_background_cond = background_cond[:,2:,:,:]
                    real_background_cond = background_cond[:,:2,:,:]
                    e_tau_i_null = self.get_eps(x, tau_i, null_background_cond)
                    e_tau_i_real = self.get_eps(x, tau_i, real_background_cond)
                    e_tau_i = e_tau_i_null + self.guidance_scale * (e_tau_i_real-e_tau_i_null)
                    if self.guidance_rescale > 0:
                        e_tau_i = rescale_noise_cfg(e_tau_i, e_tau_i_real, guidance_rescale=self.guidance_rescale)
            else:
                if use_melody:
                    assert background_cond.shape[1] == 4 # chd_onset, chd_sustain, melody_onset, melody_sustain
                    e_tau_i = self.get_eps(x, tau_i, background_cond)
                else:
                    print(background_cond.shape)
                    assert background_cond.shape[1] == 2 # chd_onset, chd_sustain
                    e_tau_i = self.get_eps(x, tau_i, background_cond)

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha}}$
        one_over_sqrt_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.one_over_sqrt_alpha_bar[step_tau_i]
        )
        # $\frac{\sqrt{1-\bar\alpha}}{\sqrt{\bar\alpha}}$
        sqrt_1m_alpha_bar_over_sqrt_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_1m_alpha_bar_over_sqrt_alpha_bar[step_tau_i]
        )

        # $\sigma_t$ in DDIM
        sigma_ddim = x.new_full(
            (bs, 1, 1, 1), self.sigma_ddim[step_tau_i]
        )


        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # predicted x_0 in DDIM
        predicted_x0 = one_over_sqrt_alpha_bar * x[:, 0: e_tau_i.size(1)] - sqrt_1m_alpha_bar_over_sqrt_alpha_bar * e_tau_i

        # edit predicted x_0
        if X0EditFunc is not None:
            predicted_x0 = X0EditFunc(x0=predicted_x0, background_condition=background_cond, sampler_device=device)
            e_tau_i = (one_over_sqrt_alpha_bar * x[:, 0: e_tau_i.size(1)] - predicted_x0) / sqrt_1m_alpha_bar_over_sqrt_alpha_bar

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            if same_noise_all_measure:
                noise = torch.randn((1, predicted_x0.shape[1], 16, predicted_x0.shape[3]), device=self.device).repeat(1,1,int(predicted_x0.shape[2]/16),1)
            else:
                noise = torch.randn((1, *predicted_x0.shape[1:]), device=self.device)
        # Different noise for each sample
        else:
            if same_noise_all_measure:
                noise = torch.randn(predicted_x0.shape[0], predicted_x0.shape[1], 16, predicted_x0.shape[3], device=self.device).repeat(1,1,int(predicted_x0.shape[2]/16),1)
            else:
                noise = torch.randn(predicted_x0.shape, device=self.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        if step > 0:
            step_tau_i_m_1 = self.tau[step-1]
            # $\sqrt{\bar\alpha_{\tau_i-1}}$
            sqrt_alpha_bar_prev = x.new_full(
                (bs, 1, 1, 1), self.sqrt_alpha_bar[step_tau_i_m_1]
            )
            # $\sqrt{1-\bar\alpha_{\tau_i-1}-\sigma_\tau^2}$
            sqrt_1m_alpha_bar_prev_m_sigma2 = x.new_full(
                (bs, 1, 1, 1), (1 - self.alpha_bar[step_tau_i_m_1] - self.sigma_ddim[step_tau_i] ** 2) ** 0.5
            )
            direction_to_xt = sqrt_1m_alpha_bar_prev_m_sigma2 * e_tau_i
            x_prev = sqrt_alpha_bar_prev * predicted_x0 + direction_to_xt + sigma_ddim * noise
        else:
            x_prev = predicted_x0 + sigma_ddim * noise

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        
        #
        return x_prev, predicted_x0, e_tau_i

    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        """
        ### Sample from $q(x_t|x_0)$

        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        background_cond: Optional[torch.Tensor] = None,
        #autoreg_cond: Optional[torch.Tensor] = None,
        #external_cond: Optional[torch.Tensor] = None,
        repeat_noise: bool = False,
        temperature: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        t_start: int = 0,
        same_noise_all_measure: bool = False,
        X0EditFunc = None,
        use_classifier_free_guidance = False,
        use_melody = False,
        device = "cpu",
    ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param t_start: t_start
        """

        # Get device and batch size
        bs = shape[0]

        ######
        print(shape)
        ######


        # Get $x_T$
        if same_noise_all_measure:
            x = x_last if x_last is not None else torch.randn(shape[0],shape[1],16,shape[3], device=self.device).repeat(1,1,int(shape[2]/16),1)
        else:
            x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(np.asarray(list(range(self.used_n_steps)), dtype=np.int32))[t_start:]

        # Sampling loop
        for step in monit.iterate('Sample', time_steps):
            # Time step $t$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            x, pred_x0, e_t = self.p_sample(
                x,
                background_cond,
                #autoreg_cond,
                #external_cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                same_noise_all_measure=same_noise_all_measure,
                X0EditFunc = X0EditFunc,
                use_classifier_free_guidance = use_classifier_free_guidance,
                use_melody=use_melody,
                device=device,
            )

            s1 = step + 1

            # if self.is_show_image:
            #     if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
            #         show_image(x, f"exp/img/x{s1}.png")

        # Return $x_0$
        # if self.is_show_image:
        #     show_image(x, f"exp/img/x0.png")

        return x

    @torch.no_grad()
    def paint(
        self,
        x: Optional[torch.Tensor] = None,
        background_cond: Optional[torch.Tensor] = None,
        #autoreg_cond: Optional[torch.Tensor] = None,
        #external_cond: Optional[torch.Tensor] = None,
        t_start: int = 0,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        same_noise_all_measure: bool = False,
        X0EditFunc = None,
        use_classifier_free_guidance = False,
        use_melody = False,
    ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param background_cond: background condition
        :param autoreg_cond: autoregressive condition
        :param external_cond: external condition
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        """
        # Get  batch size
        bs = orig.size(0)

        if x is None:
            x = torch.randn(orig.shape, device=self.device)

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        # time_steps = np.flip(self.time_steps[: t_start])
        time_steps = np.flip(np.asarray(list(range(self.used_n_steps)), dtype=np.int32))[t_start:]

        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            # index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(
                x,
                background_cond,
                #autoreg_cond,
                #external_cond,
                t=ts,
                step=step,
                same_noise_all_measure=same_noise_all_measure,
                X0EditFunc = X0EditFunc,
                use_classifier_free_guidance = use_classifier_free_guidance,
                use_melody=use_melody,
            )

            # Replace the masked area with original image
            if orig is not None:
                assert mask is not None
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, self.tau[step], noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

            s1 = step + 1

        return x

    def generate(self, background_cond=None, batch_size=1,
                 same_noise_all_measure=False, X0EditFunc=None, use_classifier_free_guidance=False, use_melody=False, device="cpu"):

        shape = [batch_size, self.out_channel, self.max_l, self.h]

        if self.debug_mode:
            return torch.randn(shape, dtype=torch.float)

        return self.sample(shape, background_cond, same_noise_all_measure=same_noise_all_measure, 
                           X0EditFunc=X0EditFunc, use_classifier_free_guidance=use_classifier_free_guidance, use_melody=use_melody, device=device)
        
