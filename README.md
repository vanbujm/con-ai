# Constitutional AI
This repo is an attempt to reproduce the results of Anthropic's paper on Constitutional AI. The paper can be found 
[here](https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input). In 
particular, I am using the Hugging Face method described [here](https://huggingface.co/blog/constitutional_ai).

In short I will attempt the following:
- Create a dataset using Mistral-7B-Instruct-v0.1 from some of Anthropics 
[Red teaming prompts](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- Fine-tune the model on this dataset
- Evaluate the model on its ability to generate text that is aligned with the constitution

I'm going to attempt to do as much in possible in Typescript, as I think it is a wholly superior language to Python. 😜
