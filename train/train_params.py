class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


# for chord condition only
params_combined_cond = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=1, # with pitch augmentation, we empirically found that 1 epoch is enough for training
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=4,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=2,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)

# for melody condition and chord condition
params_separate_cond = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=1, # with pitch augmentation, we empirically found that 1 epoch is enough for training
    learning_rate=5e-5,
    max_grad_norm=10,
    fp16=True,

    # unet
    in_channels=6,
    out_channels=2,
    channels=64,
    attention_levels=[2, 3],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 4, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=2,

    # ldm
    linear_start=0.00085,
    linear_end=0.0120,
    n_steps=1000,
    latent_scaling_factor=0.18215
)