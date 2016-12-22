#!/usr/bin/env julia
progname=basename(@__FILE__)
doc = """

Usage:
    $(progname) train [--N= --ctx=]
    $(progname) eval [--min= --max= --ctx=]
    $(progname) view [--min= --max= --ctx=]

Args:
    train          Train VAE
    test           Test VAE

Options:
    --N=VALUE      # of Epoch for training [default: 1000]
    --min=VALUE    minimum value of latent variable [default: -1.0]
    --max=VALUE    maximum value of latent variable [default: 1.0]
    --ctx=VALUE    Context (cpu/gpu) [default: cpu]

"""

using DocOpt
using MXNet
using Gtk, Gtk.ShortNames
using Images

include("VAE.jl")
 
const SYM_NAME = "vae.sym"
const ARG_NAME = "vae.arg"

function get_MNIST_rawdata()
    fnames = mx.get_mnist_ubyte()
    X = open(fnames[:train_data], "r") do f
        read(f)
    end
    Y = open(fnames[:train_label], "r") do f
        read(f)
    end
    return reshape(X[17:end]/0xff, (28,28,60000)), reshape(Y[9:end], (60000,))
end

function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end

function rescale(X::Array)
    vmax = maximum(X)
    vmin = minimum(X)
    return (X-vmin)/(vmax-vmin)
end

function vae_train(N::Int; ctx=mx.cpu())
    batch_size = 100
    n_z = 2

    filenames = mx.get_mnist_ubyte()
    X,Y = get_MNIST_rawdata()
    train_provider = mx.ArrayDataProvider(:data=>X, :data_label=>X,
                                          batch_size=batch_size, shuffle=true)
    
    vae = VAE(n_z, batch_size)
    model = mx.FeedForward(vae, context=ctx)
    optimizer = mx.ADAM()

    mx.fit(model, optimizer, train_provider,
           n_epoch=N, callbacks=[mx.speedometer()],
           eval_metric=VAEMetric())

    # delete encoder params
    for i in keys(model.arg_params)
        if !ismatch(r"decode", string(i))
            delete!(model.arg_params, i)
        end
    end

    # gen decoder
    input = mx.Variable(:z)
    dec = vae_decoder(input)

    # save decoder params
    mx.save(SYM_NAME, dec)
    mx.save(ARG_NAME, model.arg_params)
end

function load_model(fname_node::String, fname_arg::String;
                    ctx=mx.cpu())
    dec = mx.load(fname_node, mx.SymbolicNode)
    arg = mx.load(fname_arg, mx.NDArray)

    model = mx.FeedForward(dec, context=ctx)
    model.aux_params = Dict{Symbol, mx.NDArray}()
    model.arg_params = arg

    return model
end

function vae_eval(vmin::Float64, vmax::Float64; ctx=mx.cpu())
    w = 20
    batch_size = w*w
    model = load_model(SYM_NAME, ARG_NAME)

    l = linspace(-vmin,vmax,w)
    dd = ndgrid(l,l)
    Xin = hcat(vec(dd[1]),vec(dd[2]))'

    eval_provider = mx.ArrayDataProvider(:z=>Xin, batch_size=batch_size)
    probs = mx.predict(model, eval_provider)

    data = reshape(probs, (28,28,w,w))
    imgdata = repmat(zeros(28,28), w, w)
    for j=1:w, i=1:w
        imgdata[(i-1)*28+1:i*28, (j-1)*28+1:j*28] = rescale(data[:,:,i,j])
    end
    Images.save("vae.png", grayim(imgdata))
end

function val_changed(ptr, user_data)
    sc1, sc2, data, img, pixbuf, model = user_data

    val1 = getproperty(@Adjustment(sc1), :value, Float64)
    val2 = getproperty(@Adjustment(sc2), :value, Float64)

    dp   = mx.ArrayDataProvider(:z=>reshape([val1, val2], (2,1)))
    prob = reshape(mx.predict(model, dp), (28,28))

    prob = rescale(prob)
    for i = 1:28, j=1:28
        pval = round(UInt8, prob[i,j]*255)
        data[(i-1)*10+1:i*10,(j-1)*10+1:j*10] = Gtk.RGB(pval, pval, pval)
    end
    G_.from_pixbuf(img, pixbuf)

    return nothing
end

function vae_win(vmin::Float64, vmax::Float64; ctx=mx.cpu())
    model = load_model(SYM_NAME, ARG_NAME, ctx=ctx)
    # pre run
    dp    = mx.ArrayDataProvider(:z=>reshape([0,0], (2,1)))
    prob  = mx.predict(model, dp)

    # gen window
    range  = vmin:(vmax-vmin)/100:vmax
    sc1    = @Scale(false, range)
    sc2    = @Scale(false, range)
    vb     = @Box(:v)
    win    = @Window(vb,"VAE Demo")
    data   = Gtk.RGB[Gtk.RGB(0,0,0) for i=1:280, j=1:280]
    pixbuf = @Pixbuf(data=data, has_alpha=false)
    img    = @Image(pixbuf)

    setproperty!(@Adjustment(sc1), :value, 0.0)
    setproperty!(@Adjustment(sc2), :value, 0.0)
    setproperty!(img, :width_request, 280)
    setproperty!(img, :height_request, 280)

    push!(vb, sc1)
    push!(vb, sc2)
    push!(vb, img)

    signal_connect(val_changed, sc1, :value_changed, Void,
                   (), false, (sc1,sc2,data,img,pixbuf,model))
    signal_connect(val_changed, sc2, :value_changed, Void,
                   (), false, (sc1,sc2,data,img,pixbuf,model))

    showall(win)
    signal_connect(win, :destroy) do w
        Gtk.gtk_quit()
    end
    Gtk.gtk_main()
end


function main()
    args = docopt(doc)
    ctx = args["--ctx"] == "cpu" ? mx.cpu() : mx.gpu()
    if args["train"]
        N = parse(Int, args["--N"])
        vae_train(N, ctx=ctx)
    elseif args["eval"]
        vmin = parse(Float64, args["--min"])
        vmax = parse(Float64, args["--max"])
        vae_eval(vmin, vmax, ctx=ctx)
    elseif args["view"]
        vmin = parse(Float64, args["--min"])
        vmax = parse(Float64, args["--max"])
        vae_win(vmin, vmax, ctx=ctx)
    end
end

main()

