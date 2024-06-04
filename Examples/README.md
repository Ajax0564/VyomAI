# Examples
**Each example is in one single notebook for readability and understanding**

**Each example implemented from scratch using Pytorch 2.0 **
| Task | dataset link | Pyotch 2.0 | description
|---|---|:---:|:---:|
|[**`text classification`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Classification.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model for text classification|
|[**`masked language modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/masked_language_modeling.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅| encoder model pretraining with mlm style| 
|[**`electra language modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/electra-pretraining.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model pretraining with electra style| 
|[**`casual language-modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Decoder.ipynb) | [mark-twain-books](https://www.kaggle.com/datasets/msinger007/mark-twain-books) |✅|decoder model pretraining with gpt style and kv-cache for fast inference | 
|[**`knowledge distilation`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Knowledge_distilation.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|initilization of a model from pretrained model|
|[**`seq2seq modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Seq2seq-absolute.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|seq2seq model training for image caption with kv-cache|
|[**`adapters`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/adapters.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|Lora and Dora for parameter efficient tunning|
|[**`vit`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/vit.ipynb) | [Scene-classification](https://www.kaggle.com/datasets/nitishabharathi/scene-classification) |✅|visual image transformer for image classification|
|[**`detr`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/clip.ipynb) | [Global-Wheat-Detection](https://www.kaggle.com/competitions/global-wheat-detection) |✅|implementation of detr DEtection TRansformer encoder decoder model for object detection|
|[**`clip`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/detr.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|implementation of contrastive language-image pre-training|
|[**`vision language multimodel`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Multimodel_basic.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|A minimmal vision-language model implementation with image-text fusion  to generate image caption with RoPE and kv-cache|
****More to come**