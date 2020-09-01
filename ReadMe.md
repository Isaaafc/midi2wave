# README

PyTorch implementation of midi2wave. Currently at the stage of implementing Gated PixelCNN and WaveNet. Check for updates on my [blog] (https://isaaafc.github.io)

## Applications

My end goal is to build an application that acts like an audio "transformer". We can build many useful cases such as live jamming with AI and "Harmonizer".

## Plan of implementation

Midi2Wave builds on many previous structures, it is neccessary to understand the previous ones before diving into implementing this.

In order, the structures to implement are:

- [PixelCNN](https://arxiv.org/pdf/1601.06759.pdf) [Reference Blog](http://sergeiturukin.com/2017/02/22/pixelcnn.html)
- [Gated PixelCNN](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf)
- [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)
- Midi2Wave
