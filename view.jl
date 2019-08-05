#!/usr/bin/env julia
progname=basename(@__FILE__)
doc = """ Variational Auto-Encoder Viewer

Usage:
  $(progname) <arch> <param> [--ctx= --min= --max=]

Args:
  <arch>    arch (*.json)
  <param>   param (*.params)

Options:
  --min=VALUE     minimum value of latent variable [default: -1.0]
  --max=VALUE     maximum value of latent variable [default: 1.0]
  --ctx=VALUE     context (cpu/gpu) [default: cpu]

"""
using DocOpt
args = docopt(doc)

using MXNet
using Gtk
import Images: scaleminmax

const IMG_SHAPE = (28,28)
const SCALING_FACTOR = 15

function GdkScaleSimple(buf::GdkPixbuf, w::Int, h::Int, interp::Integer)
  return convert(GdkPixbuf, ccall((:gdk_pixbuf_scale_simple, Gtk.libgdk), Ptr{GdkPixbuf}, (Ptr{GObject}, Cint, Cint, Cint), buf, w, h, interp))
end

function val_changed(ptr, user_data)
  sc1, sc2, data, img, model = user_data

  val1 = GAccessor.value(GtkAdjustment(sc1))
  val2 = GAccessor.value(GtkAdjustment(sc2))

  dp   = mx.ArrayDataProvider(:z=>reshape([val1, val2], (2,1)))
  prob = reshape(mx.predict(model, dp), IMG_SHAPE)
  
  f = scaleminmax(minimum(prob), maximum(prob))
  prob = f.(prob)
  for i = 1:IMG_SHAPE[1], j=1:IMG_SHAPE[2]
    pval = round(UInt8, prob[i,j]*typemax(UInt8))
    data[i,j] = Gtk.RGB(pval, pval, pval)
  end
  pixbuf = GdkPixbuf(data=data, has_alpha=false)
  p = GdkScaleSimple(pixbuf, IMG_SHAPE[1]*SCALING_FACTOR, IMG_SHAPE[2]*SCALING_FACTOR, Gtk.GConstants.GdkInterpType.NEAREST)
  set_gtk_property!(img, :pixbuf, p)

  return nothing
end

function vae_win(model, vmin::Float64, vmax::Float64)
  # pre run
  dp    = mx.ArrayDataProvider(:z=>mx.NDArray(reshape([0,0], (2,1))))
  prob  = mx.predict(model, dp)

  W = IMG_SHAPE[1]
  H = IMG_SHAPE[2]

  # gen window
  range  = vmin:(vmax-vmin)/100:vmax
  sc1    = GtkScale(false, range)
  sc2    = GtkScale(false, range)
  vb     = GtkBox(:v)
  win    = GtkWindow(vb,"VAE Demo")
  data   = Gtk.RGB[Gtk.RGB(0,0,0) for i=1:W, j=1:H]
  pixbuf = GdkPixbuf(data=data, has_alpha=false)
  img    = GtkImage(pixbuf)

  set_gtk_property!(GtkAdjustment(sc1), :value, 0.0)
  set_gtk_property!(GtkAdjustment(sc2), :value, 0.0)
  set_gtk_property!(img, :width_request, W*SCALING_FACTOR)
  set_gtk_property!(img, :height_request, H*SCALING_FACTOR)

  push!(vb, sc1)
  push!(vb, sc2)
  push!(vb, img)

  signal_connect(val_changed, sc1, :value_changed, Nothing,
                 (), false, (sc1,sc2,data,img,model))
  signal_connect(val_changed, sc2, :value_changed, Nothing,
                 (), false, (sc1,sc2,data,img,model))

  showall(win)
  signal_connect(win, :destroy) do w
    Gtk.gtk_quit()
  end
  Gtk.gtk_main()
end

function load_model(arch_name::AbstractString, param_name::AbstractString; ctx=mx.cpu())
  arch = mx.load(arch_name, mx.SymbolicNode)
  arg = mx.load(param_name, mx.NDArray)

  model = mx.FeedForward(arch, context=ctx)
  model.aux_params = Dict{Symbol, mx.NDArray}()
  model.arg_params = arg

  return model
end

function main(args)
  ctx = args["--ctx"] == "cpu" ? mx.cpu() : mx.gpu()
  vmin = parse(Float64, args["--min"])
  vmax = parse(Float64, args["--max"])
  model = load_model(args["<arch>"], args["<param>"], ctx=ctx)
  vae_win(model, vmin, vmax)
end

main(args)

