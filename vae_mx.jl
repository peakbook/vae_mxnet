#!/usr/bin/env julia
progname=basename(@__FILE__)
doc = """ Variational Auto-Encoder Test

Usage:
  $(progname) train <model> [--epoch= --batch= --ctx= --dir=]
  $(progname) eval <model> <img> [--min= --max= --ctx=]

Args:
  train     train VAE
  eval      generate image from trained VAE
  <model>   name for exporting and importing model files
  <img>     geberated image filename

Options:
  --epoch=VALUE   # of epoch for training [default: 6000]
  --batch=VALUE   batch size  [default: 100]
  --min=VALUE     minimum value of latent variable [default: -1.0]
  --max=VALUE     maximum value of latent variable [default: 1.0]
  --ctx=VALUE     context (cpu/gpu) [default: cpu]
  --dir=PATH      directory path for callback [default: epoch]

"""

using DocOpt
args = docopt(doc)

using MXNet
using Images
using Printf

include("VAE.jl")

const IMG_SHAPE = (28,28)
const N_TRAIN = 60000
const OFFSET_TRAIN = 16
const OFFSET_LABEL = 8

function get_MNIST_rawdata()
  fnames = mx.get_mnist_ubyte()
  X = open(fnames[:train_data], "r") do f
    seek(f, OFFSET_TRAIN)
    read(f)
  end
  Y = open(fnames[:train_label], "r") do f
    seek(f, OFFSET_LABEL)
    read(f)
  end

  return reshape(X/typemax(eltype(X)), (IMG_SHAPE...,N_TRAIN)),
                 reshape(Y, (N_TRAIN,))
end

function ndgrid(v1::AbstractVector{T}, v2::AbstractVector{T}) where T
  m, n = length(v1), length(v2)
  v1 = reshape(v1, m, 1)
  v2 = reshape(v2, 1, n)
  (repeat(v1, 1, n), repeat(v2, m, 1))
end

function save_image(decoder::mx.FeedForward, prefix::AbstractString; frequency::Int=10)
  mkpath(prefix)
  mx.every_n_epoch(frequency) do model, state, metric
    for i in keys(model.arg_params)
      if match(r"decode", string(i)) !== nothing
        decoder.arg_params[i] = model.arg_params[i]
      end
    end
    fname = joinpath(prefix, @sprintf("%06d.png", state.curr_epoch))
    data = vae_eval(decoder, -1.0, 1.0)
    save_as_tile_images(data, fname)
  end
end

function save_params(model::mx.FeedForward, prefix::AbstractString; frequency::Int=10)
  params = Dict{Symbol, mx.NDArray}()
  mx.every_n_epoch(frequency) do model, state, metric
    for i in keys(model.arg_params)
      if match(r"decode", string(i)) !== nothing
        params[i] = model.arg_params[i]
      end
    end
    mx.save(joinpath(prefix, @sprintf("%06d.params",state.curr_epoch)), params)
  end
end

function save_model(name::AbstractString, net::mx.FeedForward)
  mx.save(name*".json", net.arch)
  mx.save(name*".params", net.arg_params)
end

function load_model(name::AbstractString; ctx=mx.cpu())
  arch = mx.load(name*".json", mx.SymbolicNode)
  arg = mx.load(name*".params", mx.NDArray)

  model = mx.FeedForward(arch, context=ctx)
  model.aux_params = Dict{Symbol, mx.NDArray}()
  model.arg_params = arg

  return model
end

function vae_train(epoch::Int, batch_size::Int, dir::AbstractString; ctx=mx.cpu())
  n_z = 2

  filenames = mx.get_mnist_ubyte()
  X,Y = get_MNIST_rawdata()
  train_provider = mx.ArrayDataProvider(:data_in=>X, :data_out=>X,
                                        batch_size=batch_size, shuffle=true)

  vae = VAE(n_z, :data_in, :data_out)
  model = mx.FeedForward(vae, context=ctx)
  optimizer = mx.ADAM()

  input = mx.Variable(:z)
  dec = vae_decoder(input)
  decoder = mx.FeedForward(dec, context=ctx)
  decoder.aux_params = Dict{Symbol, mx.NDArray}()
  decoder.arg_params = Dict{Symbol, mx.NDArray}()

  mx.fit(model, optimizer, train_provider,
         n_epoch=epoch, 
         callbacks=[save_image(decoder, dir, frequency=10),
                    save_params(model, dir, frequency=100)],
         eval_metric=VAEMetric())

  # extract decoder params
  for i in keys(model.arg_params)
    if match(r"decode", string(i)) !== nothing
      decoder.arg_params[i] = model.arg_params[i]
    end
  end 

  return decoder
end

function vae_eval(model::mx.FeedForward, vmin::Float64, vmax::Float64; w::Integer=20, ctx=mx.cpu())
  batch_size = w*w

  l = range(vmin,stop=vmax,length=w)
  dd = ndgrid(l,l)
  Xin = collect(hcat(vec(dd[1]),vec(dd[2]))')

  eval_provider = mx.ArrayDataProvider(:z=>Xin, batch_size=batch_size)
  probs = mx.predict(model, eval_provider)

  return reshape(probs, (IMG_SHAPE..., w, w))
end

function save_as_tile_images(data::Array, fname::AbstractString; w::Integer=20)
  data = reshape(data, (IMG_SHAPE..., w, w))
  imgdata = repeat(zeros(IMG_SHAPE...), w, w)
  for j=1:w, i=1:w
    f = scaleminmax(minimum(data[:,:,i,j]), maximum(data[:,:,i,j]))
    data[:,:,i,j] = f.(data[:,:,i,j])
    imgdata[(i-1)*IMG_SHAPE[1]+1:i*IMG_SHAPE[1],
            (j-1)*IMG_SHAPE[2]+1:j*IMG_SHAPE[2]] = f.(data[:,:,i,j])
  end
  Images.save(fname, colorview(Gray, imgdata'))
end

function main(args)
  ctx = args["--ctx"] == "cpu" ? mx.cpu() : mx.gpu()
  if args["train"]
    epoch = parse(Int, args["--epoch"])
    batch = parse(Int, args["--batch"])
    decoder = vae_train(epoch, batch, args["--dir"], ctx=ctx)
    save_model(args["<model>"], decoder)
  elseif args["eval"]
    vmin = parse(Float64, args["--min"])
    vmax = parse(Float64, args["--max"])
    model = load_model(args["<model>"], ctx=ctx)
    data = vae_eval(model, vmin, vmax, ctx=ctx)
    save_as_tile_images(data, args["<img>"])
  end
end

main(args)

