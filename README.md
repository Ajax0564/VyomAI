# VyomAI
**Transfomer models implementation from scratch using pytorch to make it more accessible for research purposes.** 

# Examples
**Each example is in one single notebook for readability and understanding**

**Each example implemented from scratch using Pytorch 2.0**
| Task | dataset link | Pyotch 2.0 | description
|---|---|:---:|:---:|
|[**`text classification`**](https://) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model for text classification|
|[**`masked language modeling`**](https://) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅| encoder model pretraining with mlm style| 
|[**`electra language modeling`**](https://) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model pretraining with electra style| 
|[**`casual language-modeling`**](https://) | [mark-twain-books](https://www.kaggle.com/datasets/msinger007/mark-twain-books) |✅|decoder model pretraining with gpt style and kv-cache for fast inference | 
|[**`knowledge distilation`**](https://) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|initilization of a model from pretrained model|
|[**`seq2seq modeling`**](https://) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|seq2seq model training for image caption with kv-cache|
|[**`adapters`**](https://) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|Lora and Dora for parameter efficient tunning|
|[**`vit`**](https://) | [Scene-classification](https://www.kaggle.com/datasets/nitishabharathi/scene-classification) |✅|visual image transformer for image classification|
|[**`clip`**](https://) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|implementation of contrastive language-image pre-training|
|[**`vision language multimodel`**](https://) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|vision model output fusion with decoder input to generate image caption with RoPE and kv-cache| |

****Many more to come**