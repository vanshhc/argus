# Supported models

A model is supported only after it passes the Mac accuracy test and runs on a real iPhone.

| Model | Family | Mac accuracy | iPhone | iOS | Export settings | Prep time | Memory | Status |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B Instruct | llama | float32 PyTorch: pass (max 1.7e-4, top-1 100%, 3,055 tokens) | — | — | — | — | — | In progress: export next |

## Reference (Apple's own support, not this project)

| Model | iPhone | iOS | Result |
|---|---|---|---|
| Qwen3 0.6B | iPhone 15 | 27.0.1 | Runs; engine preparation 520 s on first load. See the study workspace. |
