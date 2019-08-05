mutable struct VAEMetric <: mx.AbstractEvalMetric
  loss_sum :: Float32
  kl_sum   :: Float32
  n_sample :: Int

  VAEMetric() = new(0.0, 0.0, 0)
end

function mx.update!(metric::VAEMetric, labels::mx.VecOfNDArray, preds::mx.VecOfNDArray)
  truth = labels[1]
  decode = preds[1] 
  kl = preds[2]

  metric.n_sample += size(decode)[end]
  metric.loss_sum = copy(sum(decode))[1]
  metric.kl_sum = copy(sum(kl))[1]
end

function mx.get(metric :: VAEMetric)
  return [(:Loss, metric.loss_sum / metric.n_sample),
          (:D,  metric.kl_sum / metric.n_sample)]
end

function mx.reset!(metric :: VAEMetric)
  metric.loss_sum  = 0.0
  metric.kl_sum   = 0.0
  metric.n_sample = 0
end

# encoder network
function vae_encoder(input::mx.SymbolicNode, n_z::Int)
  enc = @mx.chain input => 
  mx.FullyConnected(name=:encode_fc1, num_hidden=500) =>
  mx.Activation(name=:encode_act1, act_type=:relu) =>
  mx.FullyConnected(name=:encode_fc2, num_hidden=250) =>
  mx.Activation(name=:encode_act2, act_type=:relu)

  mu  = @mx.chain enc =>
  mx.FullyConnected(name=:encode_fc3_mu, num_hidden=n_z) =>
  mx.Activation(name=:encode_act3_mu, act_type=:tanh)

  s   = @mx.chain enc =>
  mx.FullyConnected(name=:encode_fc3_s, num_hidden=n_z) =>
  mx.Activation(name=:encode_act3_s, act_type=:tanh)

  return mu, s
end

# reparameterization trick
function vae_sampler(mu::mx.SymbolicNode, s::mx.SymbolicNode, epsilon::mx.SymbolicNode)
  v  = s .* s
  z  = (mu .+ (v .* epsilon))
  kl = 0.5(v + (mu .* mu) - 1.0 - log(v))
  return z, kl
end

# decoder network
function vae_decoder(z::mx.SymbolicNode)
  return @mx.chain z =>
  mx.FullyConnected(name=:decode_fc1, num_hidden=250) =>
  mx.Activation(name=:decode_act1, act_type=:relu) =>
  mx.FullyConnected(name=:decode_fc2, num_hidden=500) =>
  mx.Activation(name=:decode_act2, act_type=:relu) =>
  mx.FullyConnected(name=:decode_fc3, num_hidden=28*28) =>
  mx.Activation(name=:decode_act3, act_type=:sigmoid) =>
  reshape((28,28,-1)) 
end

function VAE(n_z::Int, data_in_name::Symbol=:data_in, data_out_name::Symbol=:data_out)
  # training data
  data_in  = mx.Variable(data_in_name)
  data_out = mx.Variable(data_out_name)

  # samples from N(0,I)
  epsilon = mx.normal(mx.SymbolicNode, shape=(n_z, -1))

  # define vae network
  mu, s = vae_encoder(data_in, n_z)
  z, kl = vae_sampler(mu, s, epsilon)
  dec   = vae_decoder(z)

  # define loss function
  loss  = mx.sum_axis(mx.square(mx.Flatten(data_out-dec)), axis=1)
  D     = mx.sum_axis(kl, axis=1)
  net   = mx.MakeLoss(loss + D)
  d     = mx.BlockGrad(kl)

  return mx.Group(net, d)
end

