# lenia-gpu

Lenia universe written in rust.
Based on Arrayfire, so it can run on gpu via CUDA (best option) and OpenCl, or on cpu via OpenCl.

It's just implementation from this doc: https://arxiv.org/pdf/2005.03742.pdf

Controls:
 - p - pause simulation
 - up/down - select parameter
 - left/right/enter - change parameter
 - s - save configurations to file
 - q - exit

Layer and channel data are saved to .toml, matrix values itself to .bin.
For now is best to create core lenia preset in file manager, and then tweak it's settings via UI.

There are two --bin targets, recommended method is to launch ui target, it will create compute target as a child and control it with tcp commands.

Features ideas are greatly appreciated.

![example](https://github.com/HVisMyLife/lenia_gpu/blob/master/output.gif)

There are 4 functions, that can be used as kernel or growth map, each can be centered (moved halfway down) or/and have sigmoid cutoff (default is hard).

Default map size is 2048x2048, and kernel radius of 92 (185x185)
