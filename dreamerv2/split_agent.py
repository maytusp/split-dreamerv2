import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd
import elements
import common
import expl
from augmentation import Augmentator
import numpy as np

class Agent(common.Module):

  def __init__(self, config, logger, actspce, step, dataset):
    self.config = config
    self._logger = logger
    self._action_space = actspce
    self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
    self._should_expl = elements.Until(int(
        config.expl_until / config.action_repeat))
    self._counter = step
    with tf.device('cpu:0'):
      self.step = tf.Variable(int(self._counter), tf.int64)
    self._dataset = dataset
    self.wm = WorldModel(self.step, config)
    self._task_behavior = ActorCritic(config, self.step, self._num_act)
    reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(actspce),
        plan2explore=lambda: expl.Plan2Explore(
            config, self.wm, self._num_act, self.step, reward),
        model_loss=lambda: expl.ModelLoss(
            config, self.wm, self._num_act, self.step, reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    self.train(next(self._dataset))

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    tf.py_function(lambda: self.step.assign(
        int(self._counter), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._num_act))
      state = latent, action
    elif obs['reset'].any():
      state = tf.nest.map_structure(lambda x: x * common.pad_dims(
          1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)
    latent, action = state
#     if self.config.encode_whole:
#       obs_hat = {}
#       obs_hat['image'] = tf.expand_dims(obs['image'], axis=0)
#       embed_hat = self.wm.encoder_hat(self.wm.preprocess_hat(obs_hat))  
#       post_hat = self.wm.aux_vae.observe(embed_hat)
#     else:
#       post_hat = None
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, sample)
    feat = self.wm.rssm.get_feat(latent)
    
    
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self.step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
    action = common.action_noise(action, noise[mode], self._action_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    start_hat = outputs['post_hat']
    if self.config.pred_discount:  # Last step could be terminal.
      start = tf.nest.map_structure(lambda x: x[:, :-1], start)
    reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    metrics.update(self._task_behavior.train(self.wm, start, reward, start_hat))
    if self.config.expl_behavior != 'greedy':
      if self.config.pred_discount:
        data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data):
    return {'openl': self.wm.video_pred(data)}
  def report_vary_z_global(self, data):
    return {'vary_z_global' : self.wm.vary_z_global(data)}
  def report_vary_z_local(self, data):
    return {'vary_z_local' : self.wm.vary_z_local(data)}
  def report_vary_x_hat(self, data):
    return {'vary_x_hat' : self.wm.vary_x_hat(data)}


class WorldModel(common.Module):

  def __init__(self, step, config):
    self.step = step
    self.config = config
    self.rssm = common.split_RSSM(stoch_hat = config.aux_vae.stoch, discrete_hat = config.aux_vae.discrete, **config.split_rssm, encode_whole = config.encode_whole)
    self.aux_vae = common.aux_vae(**config.aux_vae) # add auxiliary vae
    self.heads = {}
    shape = config.image_size + (1 if config.grayscale else 3,)
    self.encoder = common.ConvEncoder(**config.encoder)
    self.encoder_hat = common.split_ConvEncoder(**config.encoder_hat) # add auxiliary encoder
    self.decoder_hat = common.split_ConvDecoder(shape, **config.decoder_hat) # add auxiliary decoder
    self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    self.augmentor = Augmentator(type=config.augmentation, size=config.patch_size)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.encoder_hat, self.decoder_hat, self.rssm, self.aux_vae, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))        
    return state, outputs, metrics

  def loss(self, data_input, state=None):
    # Prepare data
    data = self.preprocess(data_input)
    data_hat = self.preprocess_hat(data_input)
    if self.config.encode_whole:
      data_hat['image'] = data_hat['image'][:, 0:1, :, :, :]
    
    # Auxiliary VAE
    embed_hat = self.encoder_hat(data_hat)
    post_hat = self.aux_vae.observe(embed_hat)

    
    # Main VAE
    embed = self.encoder(data)
    post, prior = self.rssm.observe(embed, data['action'], state)
    
    # Increasing beta
    if self.config.step_beta:
      b_max = self.config.beta_max
      b_min = self.config.beta_min        
      m = (b_max - b_min) / self.config.final_beta_step
      current_step = tf.cast(self.step, tf.float32)
      beta = (m * current_step) + b_min
    else:
      beta = 1    
    
#     Compute KL losses

    kl_loss, kl_value = self.rssm.kl_loss(post, prior, post_hat, discrete = self.config.split_rssm.discrete, merge_kl = self.config.merge_kl, **self.config.kl)
    assert len(kl_loss.shape) == 0
    
    if self.config.merge_kl:
      losses = {'kl': beta*kl_loss}  
    else:
      kl_loss_hat, kl_value_hat = self.aux_vae.kl_loss(post_hat, prior, stoch_hat = self.config.aux_vae.stoch, discrete_hat = self.config.aux_vae.discrete, use_main_prior = self.config.use_main_prior, **self.config.kl)
      assert len(kl_loss_hat.shape) == 0
      losses = {'kl': beta*kl_loss, 'kl_hat': beta*kl_loss_hat}  
      
    likes = {}
    # get z
    feat = self.rssm.get_feat(post)
    
    # get z_hat
    feat_hat = self.aux_vae.get_feat(post_hat)
    
    # Combine z and z_hat
    feat_merge = self.rssm.get_feat_merge(post, post_hat, self.config.encode_whole)
    
#     # Add reconstruction loss for auxilirary VAE
    like_hat = tf.cast(self.decoder_hat(feat_hat).log_prob(data_hat['image']), tf.float32)
    likes['image_hat'] = like_hat # Scramble image loss
    losses['image_hat'] = -like_hat.mean()
    
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      if name=='image':
        inp = feat_merge
      else:
        inp = feat if grad_head else tf.stop_gradient(feat)
        
      like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
      likes[name] = like
      losses[name] = -like.mean()

    
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post, post_hat=post_hat,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    if not(self.config.merge_kl):
      metrics['model_kl_hat'] = kl_value_hat.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    # Metrics update for aux vae
    metrics['post_ent_hat'] = self.aux_vae.get_dist(post).entropy().mean() 
    return model_loss, post, outs, metrics


  def imagine(self, policy, start, horizon):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = self.rssm.get_feat(state)
      action = policy(tf.stop_gradient(feat)).sample()
      succ = self.rssm.img_step(state, action)
      return succ, feat, action
    feat = 0 * self.rssm.get_feat(start)
    action = policy(feat).mode()
    succs, feats, actions = common.static_scan(step, tf.range(horizon), (start, feat, action))
       
    
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in succs.items()}
    if 'discount' in self.heads:
      discount = self.heads['discount'](feats).mean()
    else:
      discount = self.config.discount * tf.ones_like(feats[..., 0])
    return feats, states, actions, discount
  


  @tf.function
  def preprocess(self, obs_input):
    dtype = prec.global_policy().compute_dtype
    obs = obs_input.copy()
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def preprocess_hat(self, obs_input):
    dtype = prec.global_policy().compute_dtype
    obs = obs_input.copy()
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    obs_bs, obs_length, n_row, n_col, n_channel = obs['image'].shape
    obs['image'] = tf.reshape(obs['image'], (obs_bs*obs_length, n_row, n_col, n_channel))
  # obs['image'] = tf.stop_gradient(tf.map_fn(fn=lambda t: self.augmentor.augment(t), elems = obs['image']))
    obs['image'] = tf.vectorized_map(lambda t: self.augmentor.augment(t), obs['image'])
    obs['image'] = tf.reshape(obs['image'], (obs_bs, obs_length, n_row, n_col, n_channel))
#     else:
#       obs_length, n_row, n_col, n_channel = obs['image'].shape
#       obs['image'] = tf.reshape(obs['image'], (obs_length, n_row, n_col, n_channel))
#   #     obs['image'] = tf.stop_gradient(tf.map_fn(fn=lambda t: self.augmentor.augment(t), elems = obs['image']))
#       obs['image'] = tf.vectorized_map(lambda t: self.augmentor.augment(t), obs['image'])
#       obs['image'] = tf.reshape(obs['image'], (obs_length, n_row, n_col, n_channel))        
#     obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
#     if 'discount' in obs:
#       obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data_input):
    
    data_hat = self.preprocess_hat(data_input)
    embed_hat = self.encoder_hat(data_hat)
    aux_idx = 1 if self.config.encode_whole else 5
    states_hat = self.aux_vae.observe(embed_hat[:6, :aux_idx])   
    
    data = self.preprocess(data_input)
    truth = data['image'][:6] + 0.5    
    embed = self.encoder(data)
    states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](
        self.rssm.get_feat_merge(states, states_hat, self.config.encode_whole)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    
    if self.config.use_main_prior:
      prior_hat = prior
    else:
      prior_hat_stat = {}
      if not(self.config.encode_whole):
        prior_hat_stat['logit'] = (1/self.config.aux_vae.discrete)*np.ones((6, truth.shape[1]-5, self.config.aux_vae.stoch,          self.config.aux_vae.discrete))
      else:
        prior_hat_stat['logit'] = (1/self.config.aux_vae.discrete)*np.ones((6, 1, self.config.aux_vae.stoch,                        self.config.aux_vae.discrete))
      dist_hat = self.aux_vae.get_dist(prior_hat_stat)
      prior_hat = {}
      prior_hat['stoch'] = dist_hat.sample()
    
    openl = self.heads['image'](self.rssm.get_feat_merge(prior, prior_hat, self.config.encode_whole)).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

  def vary_z_global(self, data_input):
    data_hat = self.preprocess_hat(data_input)
    embed_hat = self.encoder_hat(data_hat)
    states_hat = self.aux_vae.observe(embed_hat[:6, 5:6])
    
    data = self.preprocess(data_input)
    embed = self.encoder(data)
    states, _ = self.rssm.observe(embed[:6, 5:6], data['action'][:6, 5:6]) # Pick sample 6, time position 5
    for i in range(6):
      if i == 0:
        recon = self.heads['image'](self.rssm.get_feat_merge(states, states_hat, self.config.encode_whole)).mode()
      else:
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:6], init)                
        recon = tf.concat([recon, self.heads['image'](self.rssm.get_feat_merge(prior, states_hat, self.config.encode_whole)).mode()], 3)
    recon = recon + 0.5
    B, T, H, W, C = recon.shape
    return recon.transpose((1, 0, 2, 3, 4)).reshape((T, B*H, W, C))    

  def vary_z_local(self, data_input):
    data_hat = self.preprocess_hat(data_input)
    embed_hat = self.encoder_hat(data_hat)
    states_hat = self.aux_vae.observe(embed_hat[:6, 5:6]) 
    
    data = self.preprocess(data_input)
    embed = self.encoder(data)
    states, _ = self.rssm.observe(embed[:6, 5:6], data['action'][:6, 5:6]) # Pick sample 6, time position 5
       
    prior_hat_stat = {}
    prior_hat_stat['logit'] = (1/self.config.aux_vae.discrete)*np.ones((6, 1, self.config.aux_vae.stoch,           self.config.aux_vae.discrete))
    dist_hat = self.aux_vae.get_dist(prior_hat_stat)
    for i in range(6):
      if i == 0:
        recon = self.heads['image'](self.rssm.get_feat_merge(states, states_hat, self.config.encode_whole)).mode()
      else:
        if self.config.use_main_prior:
          init = {k: v[:, -1] for k, v in states.items()}
          prior = self.rssm.imagine(data['action'][:6, 5:6], init)
          prior_hat = prior
        else:
          prior_hat = {}
          prior_hat['stoch'] = dist_hat.sample()
        recon = tf.concat([recon, self.heads['image'](self.rssm.get_feat_merge(states, prior_hat, self.config.encode_whole)).mode()], 3)
    recon = recon + 0.5
    B, T, H, W, C = recon.shape
    return recon.transpose((1, 0, 2, 3, 4)).reshape((T, B*H, W, C))

  def vary_x_hat(self, data_input):
    data_hat = self.preprocess_hat(data_input)
    embed_hat = self.encoder_hat(data_hat)
    states_hat = self.aux_vae.observe(embed_hat[:6, 5:6])    
    
    data = self.preprocess(data_input)
    embed = self.encoder(data)
    states, _ = self.rssm.observe(embed[:6, 5:6], data['action'][:6, 5:6]) # Pick sample 6, time position 5
    
    prior_hat_stat = {}
    prior_hat_stat['logit'] = (1/self.config.aux_vae.discrete)*np.ones((6, 1, self.config.aux_vae.stoch,           self.config.aux_vae.discrete))
    dist_hat = self.aux_vae.get_dist(prior_hat_stat)
    for i in range(6):
      if i == 0:
        recon = self.decoder_hat(self.aux_vae.get_feat(states_hat)).mode()
      else:
        if self.config.use_main_prior:
          init = {k: v[:, -1] for k, v in states.items()}
          prior = self.rssm.imagine(data['action'][:6, 5:6], init)
          prior_hat = prior
        else:
          prior_hat = {}
          prior_hat['stoch'] = dist_hat.sample()
        recon = tf.concat([recon, self.decoder_hat(self.aux_vae.get_feat(prior_hat)).mode()], 3)
    recon = recon + 0.5
    B, T, H, W, C = recon.shape
    return recon.transpose((1, 0, 2, 3, 4)).reshape((T, B*H, W, C))

class ActorCritic(common.Module):

  def __init__(self, config, step, num_actions):
    self.config = config
    self.step = step
    self.num_actions = num_actions
    self.actor = common.MLP(num_actions, **config.actor)
    self.critic = common.MLP([], **config.critic)
    if config.slow_target:
      self._target_critic = common.MLP([], **config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **config.critic_opt)

  def train(self, world_model, start, reward_fn, start_hat):
    metrics = {}
    hor = self.config.imag_horizon
    with tf.GradientTape() as actor_tape:
      feat, state, action, disc = world_model.imagine(self.actor, start, hor)
      reward = reward_fn(feat, state, action)
      target, weight, mets1 = self.target(feat, action, reward, disc)
      actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets3 = self.critic_loss(feat, action, target, weight)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, feat, action, target, weight):
    metrics = {}
    policy = self.actor(tf.stop_gradient(feat))
    if self.config.actor_grad == 'dynamics':
      objective = target
    elif self.config.actor_grad == 'reinforce':
      baseline = self.critic(feat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
    elif self.config.actor_grad == 'both':
      baseline = self.critic(feat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.step)
      objective = mix * target + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.step)
    objective += ent_scale * ent[:-1]
    actor_loss = -(weight[:-1] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, feat, action, target, weight):
    dist = self.critic(feat)[:-1]
    target = tf.stop_gradient(target)
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, feat, action, reward, disc):
    reward = tf.cast(reward, tf.float32)
    disc = tf.cast(disc, tf.float32)
    value = self._target_critic(feat).mode()
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0)
    weight = tf.stop_gradient(tf.math.cumprod(tf.concat(
        [tf.ones_like(disc[:1]), disc[:-1]], 0), 0))
    metrics = {}
    metrics['reward_mean'] = reward.mean()
    metrics['reward_std'] = reward.std()
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, weight, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)