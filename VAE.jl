type VAEMetric <: mx.AbstractEvalMetric
    mse_sum  :: Float64
    kl_sum   :: Float64
    n_sample :: Int

    VAEMetric() = new(0.0, 0.0, 0)
end

function mx.update!(metric :: VAEMetric, labels :: Vector{mx.NDArray}, preds :: Vector{mx.NDArray})
    truth = labels[1]
    decode = preds[1] 
    kl = preds[2]

    mse = mx.MSE()
    mx._update_single_output(mse, truth, decode)
    metric.mse_sum = mse.mse_sum
    metric.n_sample = mse.n_sample
    metric.kl_sum = copy(sum(kl))[1]
end

function mx.get(metric :: VAEMetric)
    return [(:Loss, metric.mse_sum / metric.n_sample),
            (:D,  metric.kl_sum/metric.n_sample)]
end

function mx.reset!(metric :: VAEMetric)
    metric.mse_sum  = 0.0
    metric.kl_sum   = 0.0
    metric.n_sample = 0
end

# encoder network
function vae_encoder(input::mx.SymbolicNode, n_z::Int)
    enc = @mx.chain mx.FullyConnected(data=input, num_hidden=1000) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden=500) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden=250) =>
                    mx.Activation(act_type=:relu)

    mu  = @mx.chain mx.FullyConnected(data=enc, num_hidden=n_z) =>
                    mx.Activation(act_type=:tanh)

    s   = @mx.chain mx.FullyConnected(data=enc, num_hidden=n_z) =>
                    mx.Activation(act_type=:tanh)

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
    return @mx.chain mx.FullyConnected(name=:decode1, data=z, num_hidden=250) =>
                     mx.Activation(act_type=:relu) =>
                     mx.FullyConnected(name=:decode2, num_hidden=500) =>
                     mx.Activation(act_type=:relu) =>
                     mx.FullyConnected(name=:decode3, num_hidden=1000) =>
                     mx.Activation(act_type=:relu) =>
                     mx.FullyConnected(name=:decode4, num_hidden=28*28) =>
                     mx.Activation(act_type=:sigmoid)
end

function VAE(n_z::Int, batch_size::Int;
             data_name::Symbol=:data, label_name::Symbol=:data_label)
    # training data
    data    = mx.Variable(data_name)
    label   = mx.Variable(label_name)

    # samples from N(0,I)
    epsilon = mx.normal(shape=(n_z, batch_size))

    # define vae network
    mu, s = vae_encoder(data, n_z)
    z, kl = vae_sampler(mu, s, epsilon)
    dec   = vae_decoder(z)

    # define loss function
    loss  = mx.sum_axis(mx.square(mx.Flatten(label)-dec), axis=1)
    D     = mx.sum_axis(kl, axis=1)
    net   = mx.MakeLoss(loss + D)
    d     = mx.BlockGrad(data=kl)

    return mx.Group(net, d)
end

