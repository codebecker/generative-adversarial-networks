# Text to Image Synthesis

### About
Implementation of [Generative Adversarial Text-to-Image Synthesis](https://arxiv.org/abs/1605.05396) by Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele and Honglak Lee


#### How to generate samples
Adjust the configuration in ```main.py```as follows
- ```ds_name```: valid inputs are 'fmnist', 'mnist', 'coco10', 'coco' 
- ```embedding_name```: valid inputs are 'transformers' (uses [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)) and [fastText](https://fasttext.cc/docs/en/crawl-vectors.html)
- ```sample_size```: defines, how many images are generated
- ```nz```: defines the latent feature size of the noice vector
- ```text_to_image```: use embedding-free vanilla GANS or embadding-based GANS
- ```extended```: use CIFAR10 with predefined sentence-embeddings
- as well as epoch, learning_rated, ...
