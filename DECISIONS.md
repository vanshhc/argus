# Decisions

## User-selected (2026-10-04)

- A separate repository from Argus. Argus is the product; this is infrastructure.
- Goal: support open-source models beyond Apple's iOS list, each one tested.
- First model: Llama 3.2 1B Instruct.

## Provisional choices

These are the assistant's choices. The user can change them.

| Choice | Reason | Trade-off |
|---|---|---|
| Working name `coreai-model-ports` | Describes the work; no Apple trademark as the first word | Can change before publication |
| Use Apple's `coreai-models` as a pinned dependency, not a fork | Less code to maintain; Apple keeps compression and compilation | Apple's internal API can change between revisions |
| Register new families at run time | No edits to Apple's files | Uses Apple's internal registry, not a public plugin API |
| Start the Llama port from Apple's iOS Mistral port | Closest existing architecture | Must check every difference, not only RoPE |
| Mac accuracy test before any phone test | Finds wrong output early and cheaply | Needs the full model in float32 on the Mac |
| No license yet | User has not chosen one | Do not publish until chosen |

## Pinned dependency

`apple/coreai-models` revision `52c84ba874b2c57adcede08a671ce96ed1b3f433`. This is the revision tested on the iPhone 15 in the study workspace.

## 2026-10-04: Llama port design and test limits

Problem: Meta has not approved weight access yet. The port code and its accuracy test do not need the real weights.

Choice:

- `LlamaForCausalLMForiOS` subclasses Apple's iOS Mistral port. Llama and Mistral share layer names and block structure. Only the RoPE module changes.
- The new RoPE module gets its frequencies from Hugging Face's own `ROPE_INIT_FUNCTIONS`. This uses the same formula as the reference. It does not copy it.
- Supported `rope_type` values: `default` and `llama3`. Any other value, attention bias, MLP bias, or a non-SiLU activation stops with a clear error.
- Test with a small random-weight Llama config. Its RoPE frequencies cover all three Llama 3 bands (high, medium, low).

Trade-off: the subclass depends on Apple's internal `MistralExtend.rope` attribute. The pinned revision protects us; a test fails if the attribute moves.

Pass limits, set before the first run (float32, CPU, `USE_HF_IMPL=true`):

| Check | Pass |
|---|---|
| Port vs Hugging Face, max absolute logit difference, all positions | < 1e-3 |
| Top-1 token agreement, all positions | 100% |
| Apple's prompt + extend helper (KV cache path) | Passes |
| Control: Llama config through the unmodified Mistral port | Max difference > 1e-2, so the test can detect missing scaling |
