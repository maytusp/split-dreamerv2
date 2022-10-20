
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import common


class RSSM(common.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

class split_RSSM(common.Module):

  def __init__(
      self, stoch_hat = 30, discrete_hat = 32, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1, use_half_static_prior = False, h_from_zl = False, encode_whole = False):
    super().__init__()
    self._stoch = stoch
    self._stoch_hat = stoch_hat
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._discrete_hat = discrete_hat
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self.use_half_static_prior = use_half_static_prior
    self.h_from_zl = h_from_zl
    self.encode_whole = encode_whole
    self._cell = GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    #if self.encode_whole:
    post, prior = common.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
#     else:
#       post, prior = common.static_scan(
#         lambda prev, inputs: self.obs_step(prev[0], *inputs),
#         (action, embed, state_hat), (state, state))      
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

#   @tf.function
#   def imagine(self, action, state=None, state_hat=None):
#     swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
#     if state is None:
#       state = self.initial(tf.shape(action)[0])
#     assert isinstance(state, dict), state
#     action = swap(action)
#     if self.encode_whole:
#       prior = common.static_scan(lambda prev_state, prev_act: self.img_step(prev_state, prev_act, state_hat), action,             state)
#     else:
#       prior = common.static_scan(lambda prev_state, inputs: self.img_step(prev_state, *inputs), (action, state_hat), state)
#     else:
#       prior = common.static_scan(self.img_step, action, state)
#     prior = {k: swap(v) for k, v in prior.items()}
#     return prior

  # Original
#   def get_feat_merge(self, state, state_hat):
#     stoch = self._cast(state['stoch'])
#     stoch_hat = self._cast(state_hat['stoch'])
#     if self._discrete:
#       batch_size_and_time_length = stoch.shape[:-2]        
#       shape = batch_size_and_time_length + [self._stoch * self._discrete]
#       stoch = tf.reshape(stoch, shape)
#       shape_hat = batch_size_and_time_length + [self._stoch_hat * self._discrete_hat]
#       stoch_hat = tf.reshape(stoch_hat, shape_hat)
#     return tf.concat([stoch, stoch_hat, state['deter']], -1)    

  # Whole trajectory
  def get_feat_merge(self, state, state_hat, encode_whole):
    stoch = self._cast(state['stoch'])
    stoch_hat = self._cast(state_hat['stoch'])
    if self._discrete:
      batch_size, time_length = stoch.shape[:-2]      
      shape = [batch_size, time_length] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)        
      if not(encode_whole):
        shape_hat = [batch_size, time_length] + [self._stoch_hat * self._discrete_hat]
        stoch_hat = tf.reshape(stoch_hat, shape_hat)
      else: 
        shape_hat = [batch_size, time_length] + [self._stoch_hat * self._discrete_hat]
        # stoch_hat = tf.expand_dims(stoch_hat, axis=1)
        stoch_hat = tf.repeat(stoch_hat, repeats=time_length, axis=1)
        stoch_hat = tf.reshape(stoch_hat, shape_hat)
        
    return tf.concat([stoch, stoch_hat, state['deter']], -1)    

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist


  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action, sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
#     state_hat_stoch = self._cast(state_hat['stoch'])
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
#       if self.h_from_zl:
#         if self.encode_whole:
#           shape_hat = [state_hat_stoch.shape[0] * state_hat_stoch.shape[1]] + [self._stoch * self._discrete]
#         else:
#           shape_hat = state_hat_stoch.shape[:-2] + [self._stoch * self._discrete]
#         state_hat_stoch = tf.reshape(state_hat_stoch, shape_hat)
#     if self.h_from_zl: 
#       x = tf.concat([prev_stoch, state_hat_stoch, prev_action], -1)
#     else:
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
    deter = prev_state['deter']
    x, deter = self._cell(x, [deter])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('img_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    # Half static prior
    if self.use_half_static_prior:
      uni_prior = {}
      uni_prior_shape = stoch.shape.as_list()
      uni_prior_shape[1] = uni_prior_shape[1]//2
      uni_prior['logit'] = (1/self._discrete)*np.ones(uni_prior_shape)
      dist_uni = self.get_dist(uni_prior)
      stoch_uni = dist_uni.sample() if sample else dist_uni.mode()
      stoch_total = tf.concat([stoch[:,:uni_prior_shape[1],:], stoch_uni], 1)
      stats_total = {}
      stats_total['logit'] = tf.concat([stats['logit'][:,:uni_prior_shape[1],:], uni_prior['logit']], 1)
    else:
      stoch_total = stoch
      stats_total = stats
    
    prior = {'stoch': stoch_total, 'deter': deter, **stats_total}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, post_hat, forward, balance, free, free_avg, discrete, merge_kl):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    
    # z_global
    lhs_g, rhs_g = (prior, post) if forward else (post, prior)      
    mix = balance if forward else (1 - balance)
    
    if merge_kl:
      # z_local
      prior_hat = {}
      prior_hat['logit'] = (1/discrete)*np.ones(post_hat['logit'].shape)
      lhs_l, rhs_l = (prior_hat, post_hat) if forward else (post_hat, prior_hat)    
    
      # Merge z_global and z_local
      lhs, rhs = {}, {}
      lhs['logit'], rhs['logit'] = tf.concat([lhs_g['logit'], lhs_l['logit']], 0), tf.concat([rhs_g['logit'], rhs_l['logit']], 0)
    else:
      lhs, rhs = lhs_g, rhs_g
    
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class aux_vae(common.Module):

  def __init__(
      self, stoch=30, hidden=200, discrete=False, act=tf.nn.elu,
      std_act='softplus', min_std=0.1):
    super().__init__()
    self._stoch = stoch
    self._discrete = discrete
    self._hidden = hidden
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._std_act = std_act
    self._min_std = min_std
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype))
    return state

  @tf.function
  def observe(self, embed):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    embed = swap(embed)
    post = self.obs_step(embed)
    post = {k: swap(v) for k, v in post.items()}
    return post

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return stoch

  def get_dist(self, state):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, embed, sample=True):
    # prior = self.img_step(prev_state, prev_action, sample)
    # x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(embed) #(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, **stats}
    return post

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, main_prior, stoch_hat, discrete_hat, forward, balance, free, free_avg, use_main_prior):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    if use_main_prior:
      prior = main_prior
    else:
      prior = {}
      prior['logit'] = (1/discrete_hat)*np.ones(post['logit'].shape)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

class ConvEncoder(common.Module):

  def __init__(
      self, depth=32, act=tf.nn.elu, kernels=(4, 4, 4, 4), keys=['image']):
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._depth = depth
    self._kernels = kernels
    self._keys = keys

  @tf.function
  def __call__(self, obs):
    if tuple(self._keys) == ('image',):
      x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
      for i, kernel in enumerate(self._kernels):
        depth = 2 ** i * self._depth
        x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
      x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
      shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
      return tf.reshape(x, shape)

    else:
      dtype = prec.global_policy().compute_dtype
      features = []
      for key in self._keys:
        value = tf.convert_to_tensor(obs[key])
        if value.dtype.is_integer:
          value = tf.cast(value, dtype)
          semilog = tf.sign(value) * tf.math.log(1 + tf.abs(value))
          features.append(semilog[..., None])
        elif len(obs[key].shape) >= 4:
          x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
          for i, kernel in enumerate(self._kernels):
            depth = 2 ** i * self._depth
            x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
          x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
          shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
          features.append(tf.reshape(x, shape))
        else:
          raise NotImplementedError((key, value.dtype, value.shape))
      return tf.concat(features, -1)


class ConvDecoder(common.Module):

  def __init__(
      self, shape=(64, 64, 3), depth=32, act=tf.nn.elu, kernels=(5, 5, 6, 6)):
    self._shape = shape
    self._depth = depth
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._kernels = kernels

  def __call__(self, features):
    ConvT = tfkl.Conv2DTranspose
    x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class split_ConvEncoder(common.Module):

  def __init__(
      self, depth=32, act=tf.nn.elu, kernels=(4, 4, 4, 4), keys=['image']):
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._depth = depth
    self._kernels = kernels
    self._keys = keys

  @tf.function
  def __call__(self, obs):
    if tuple(self._keys) == ('image',):
      x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
      for i, kernel in enumerate(self._kernels):
        depth = 2 ** i * self._depth
        x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
      x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
      shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
      return tf.reshape(x, shape)

    else:
      dtype = prec.global_policy().compute_dtype
      features = []
      for key in self._keys:
        value = tf.convert_to_tensor(obs[key])
        if value.dtype.is_integer:
          value = tf.cast(value, dtype)
          semilog = tf.sign(value) * tf.math.log(1 + tf.abs(value))
          features.append(semilog[..., None])
        elif len(obs[key].shape) >= 4:
          x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
          for i, kernel in enumerate(self._kernels):
            depth = 2 ** i * self._depth
            x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
          x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
          shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
          features.append(tf.reshape(x, shape))
        else:
          raise NotImplementedError((key, value.dtype, value.shape))
      return tf.concat(features, -1)


class split_ConvDecoder(common.Module):

  def __init__(
      self, shape=(64, 64, 3), depth=32, act=tf.nn.elu, kernels=(5, 5, 6, 6)):
    self._shape = shape
    self._depth = depth
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._kernels = kernels

  def __call__(self, features):
    ConvT = tfkl.Conv2DTranspose
    x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class MLP(common.Module):

  def __init__(self, shape, layers, units, act=tf.nn.elu, **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._out = out

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    return self.get('out', DistLayer, self._shape, **self._out)(x)


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = getattr(tf.nn, act) if isinstance(act, str) else act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]


class DistLayer(common.Module):

  def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std

  def __call__(self, inputs):
    out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
      std = self.get('std', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    NotImplementedError(self._dist)
