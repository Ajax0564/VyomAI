# VyomAI
**Transfomer models implementation from scratch using pytorch to make it more accessible for research purposes.** 

# Examples
**Each example is in one single notebook for readability and understanding**

**Each example implemented from scratch using Pytorch 2.0 **
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
|[**`vision language multimodel`**](https://) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|A minimmal vision-language model implementation with image-text fusion  to generate image caption with RoPE and kv-cache|
****More to come**


# Usage
```python
from VyomAI import EncoderModel, EncoderForMaskedLM
from VyomAI import EncoderConfig
config = EncoderConfig()
encoder = EncoderModel(config,pos_embedding_type='rope')
#pos_embedding_type supported: Absolute, sinusoidal, RoPE
#attention_type supported: gqa, Vanila

```
## More About VyomAI

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

At a granular level, it support the following components:
| Component | Description |
| ---- | --- |
| **Encoder** | Text encoder model with Bert like architecture that support absolute, sin,Rope embedding and GQA , Vanila attention|
| **Decoder** | Text decoder model with GPT like architecture that support absolute, sin,RoPE embedding and GQA , Vanila attention and KV-Cache for fast inference |
| **Seq2Seq** |Model with Bart like architecture that support absolute, sin,RoPE embedding and GQA, Vanila attention and KV-Cache for fast inference encoder can be text or image type |
| **VisionEncoder** | Model with Vit like architecture for image encoding ****more to come** |
|**Multimodel** | A Minimal vision-language model ****more to come** |



## Improvement and Contribution
We appreciate all contributions.
If you want to contribute new features, utility functions, or tutorials please  open an issue and discuss the feature with us.



## Resources
**Some helpfull learning resources**
- [1] https://www.youtube.com/@stanfordonline
- [2] https://d2l.ai/
- [3] https://pytorch.org/tutorials/

## References
- [1] https://github.com/huggingface/transformers
- [2] https://github.com/facebookresearch/detr