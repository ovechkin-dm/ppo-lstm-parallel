### PPO with LSTM and Parallel processing

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/xEIYogwkByg/0.jpg)](http://www.youtube.com/watch?v=xEIYogwkByg)

#### Requirements
* Tensorflow 1.05
* Numpy
* Openai gym

#### Features
* Discrete policy 
* Continous policy
* Different losses
* LSTM support
* Parallelism
* Penalty for going out of action bounds  

#### Training

To start training process:
```
python train_parallel.py %ENV_NAME%
```

To start play on trained model:
```
python play.py %ENV_NAME%
``` 

To start tensorboard while training:
```
tensorboard --logdir=logs/%ENV_NAME%
```

All trained models located at `models/%ENV_NAME%/`

If you want to override hyperparameters, you should create/modify `props/%ENV_NAME%.properties` file

#### Model parameters

* `clip_eps` clipping bound (if using clipped surrogate objective) (default `0.2`)
* `grad_step` learning rate for Adam (default `0.0001`)
* `discount_factor` Discount factor (default `0.99`)
* `gae_factor` lambda for Generalized advantage estimation (default `0.98`)
* `batch_size` Batch size for training. Each gathering worker will be collecting `batch_size / gather_per_worker` episode steps (default `4096`)
* `rollout_size` Full size of rollout. Worker will be training on `rollout_size / batch_size` batches (default `8192`)
* `epochs` Num epochs for training on `rollout_size` timesteps (default `10`)
* `entropy_coef` Entropy penalty (default `0`)
* `hidden_layer_size` Common hidden layer size (LSTM unit size) (default `128`)
* `timestep_size` (LSTM only) number of timesteps per LSTM batch. Input data will be of shape `(batch_size, timestep_size, state_dim)`
* `kl_target` (only for `kl_loss`) (default `0.01`)
* `use_kl_loss` Use kl_loss or clipped objective (default `False`)
* `init_beta` (only for `kl_loss`) multiplying parameter for kl_loss (default `1.0`)
* `eta` multiplying parameter for hinge loss
* `recurrent` Use recurrent NN (default `False`)
* `worker_num` Number of workers. After optimization step each worker sends gradients to master process.
* `gather_per_worker` Number of "experience gathering" workers per optimizing worker.
* `nn_std` To use or not variance estimation through NerualNet or just use set of trainable variables.
* `reward_transform` reward transforming function (allowable_values: `scale`, `positive`, `identity`)

Many of implementation details are taken from [here](https://github.com/pat-coady/trpo)


 