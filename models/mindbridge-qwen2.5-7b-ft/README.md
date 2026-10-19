# mindbridge-qwen2.5-7b-ft:latest

这是 MindBridge Python 版默认接入的本地 Ollama 微调模型目录。

`Modelfile` 会加载同目录下的 GGUF 权重文件：

```text
mindbridge-qwen2.5-7b-ft-q4_k_m.gguf
```

模型权重较大，不建议直接放进应用发布包。交付时可以把 GGUF 或 zip 单独发送给客户，让客户解压到本目录后执行：

```bash
./scripts/create-finetuned-model.sh
```

如果本机其他位置已经有 GGUF 模型文件，也可以直接建立软链接：

```bash
ln -sf /path/to/mindbridge-qwen2.5-7b-ft-q4_k_m.gguf \
  models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4_k_m.gguf
```
