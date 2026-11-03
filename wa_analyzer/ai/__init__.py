"""Local AI analyses (everything runs on your own computer):

  sentiment   transformer sentiment + emotion per turn (multilingual XLM-R + Turkish BERTurk)
  topics      multilingual sentence embeddings + clustering -> topics over time
  read        a local LLM (Ollama) reads conversations: warmth, engagement, interest, tone, summaries
  label       a local LLM labels a sample of messages (training data)
  review      you check the LLM labels in the terminal (accuracy check)
  train       train a fast classifier on the labels (sentence embeddings + logistic regression)

Works on CPU and on NVIDIA GPUs (--device auto|cpu|cuda). Run: python -m wa_analyzer.ai --help
"""
