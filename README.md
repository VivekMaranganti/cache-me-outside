Cache-Me-Outside is an application that recommends the cheapest model capable of answering a given prompt within acceptable latency range. 

Problem: 
  People tend to overpay for inference and computation (using top models for all tasks). In addition to this, the cheapest provider is not always  
  the most cost-effective once factors such as quality, tokenization differences, and failure rates are considered. 

Solution: 
  Cache-Me-Outside evaluates prompts in real time and routes any request to the optimal model. 

Instead of optimizing solely on price, Cache-Me-Outside considers... 
  1) Accuracy
  2) Latency
  3) Failure Rate
  4) Tokenization Efficiency

Stack
-----
- Python
- FastAPI
- Redis
- Docker
- ChatGPT
- Claude
- Together AI

Future Work/Plans: 
