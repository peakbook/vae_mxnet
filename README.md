# Variational Autoencoder

## Requirements
- Julia v1.0
- [MXNet](https://github.com/dmlc/MXNet.jl)
- [Images](https://github.com/timholy/Images.jl)
- [DocOpt](https://github.com/docopt/DocOpt.jl)
- [Gtk](https://github.com/JuliaGraphics/Gtk.jl) (Optional)

## Usage
```bash
$ ./vae_mx.jl train model --batch=600 --epoch=100 --ctx=gpu 
$ ./vae_mx.jl eval model result.png
```
For more information, see `./vae_mx.jl --help`.
You can also check the output image with a simple GUI.
``` bash
$ ./view.jl model.json model.params 
```

![VAE_example](http://peakbook.github.io/images/VAE_example.png)
