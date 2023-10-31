# MAD Creator

An experiment of using Cross-Modal Retrieval to create [MADmovies](https://zh.wikipedia.org/wiki/MAD%E7%89%87).

The idea is to use a multimodal pre-trained model, i.e. [Chinese-Clip](https://github.com/OFA-Sys/Chinese-CLIP), to create vector presentations of images. Given some text, the most related images can then be retrieved using nearest neighbor search as the basis for MADmovie creation.
