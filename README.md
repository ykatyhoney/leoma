# Leoma Subnet

Leoma is an **AI video subnet** on [Bittensor](https://docs.learnbittensor.org/). Miners run **Text-Image to Video (TI2V)** models; validators evaluate miner outputs and set **winner-take-all** weights on-chain. The best miner earns subnet alpha each round.

**Supported model type (current):** **Text-Image to Video (TI2V)** only.

**Roadmap:** Support for **Text-to-Video (T2V)** and **Image-to-Video (I2V)** is planned.

---

## Contents

- [What is Leoma?](#what-is-leoma)
- [Workflow](#workflow)
- [Validator setup](#validator-setup)
- [Miner setup](#miner-setup)
- [Storage and API](#storage-and-api)
- [Security and production](#security-and-production)
- [Documentation](#documentation)
- [License](#license)

---

## What is Leoma?

Leoma is a **Bittensor subnet** for **AI-generated video**:

- **Current support:** **Text-Image to Video (TI2V)** — validators send a first frame (image) and a text prompt; miners return a short video. Validators score generated videos with a strict multi-aspect benchmark prompt (first-frame fidelity, prompt adherence, temporal quality, visual artifacts) and record pass/fail wins. Ranking is winner-take-all; the top miner gets full weight each round.
- **Roadmap:** **Text-to-Video (T2V)** and **Image-to-Video (I2V)** support are planned.

## Workflow

1. **Subnet owner** runs the **owner-sampler**: creates tasks (first frame + prompt from one-shot 5s clips in object storage), calls miners via Chutes, uploads task artifacts to the samples bucket, and sets the latest task id on the API.
2. **Validators** run the **evaluator** and **weight-setter**: poll the API for the latest task, download task data from S3, run GPT-4o evaluation, POST results to the Leoma API; each epoch, call **GET /weights** and set on-chain weights (winner-take-all).
3. The **API** (subnet owner) computes rank (dominance rule) and exposes **GET /weights**. Validators use this to set weights on-chain.
4. **Miners** register a Hugging Face model (naming: `leoma` prefix, hotkey suffix) and Chute endpoint via on-chain commit. They receive challenges; the best performer earns subnet alpha.

| Role | In Leoma |
|------|----------|
| **Miner** | Upload a TI2V model to Hugging Face (name: `leoma...` + your hotkey), deploy to Chutes, commit on-chain. Earn subnet alpha when your outputs win. |
| **Validator** | Run evaluator + weight-setter (e.g. `leoma serve`). Requires API URL, S3-compatible read access to the **samples** bucket (Hippius or Cloudflare R2 via `OBJECT_STORAGE_BACKEND`), OpenAI API key, and Bittensor wallet. |

---

## Validator setup

Validators run the **evaluator** and the **weight-setter**. Task creation and miner calls are done by the subnet owner (owner-sampler), not validators.

### Prerequisites

- **Bittensor wallet** (coldkey + hotkey) registered as a validator on the Leoma subnet.
- **Leoma API** URL (the deployed owner API).
- **Object storage (S3-compatible):** **read-only** access to the **samples** bucket (evaluator downloads task data; evaluation results go to the Leoma API with hotkey signature). Default is **Cloudflare R2** (`R2_ENDPOINT` and `R2_SAMPLES_*` keys). Set `OBJECT_STORAGE_BACKEND=hippius` to use Hippius keys instead — see `env.example`.
- **OpenAI API key** (for GPT-4o evaluation in the evaluator).
- **Validator registration in API DB:** An admin must add your validator hotkey (and UID, stake) so the API includes you in stake-weighted scoring (e.g. `leoma db add-validator --uid <uid> --hotkey <ss58>`).

### Environment variables

Set these in `.env` (copy from `env.example`):

| Variable | Description |
|----------|-------------|
| `API_URL` | Leoma API base URL (e.g. `https://api.leoma.ai`) |
| `NETUID` | Subnet ID (e.g. `99`) |
| `NETWORK` | Bittensor network (`finney` for mainnet) |
| `WALLET_NAME` | Bittensor wallet name (e.g. `default`) |
| `HOTKEY_NAME` | Bittensor hotkey name (e.g. `default`) |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o (evaluator) |
| `EPOCH_LEN` | Blocks per epoch (e.g. `180`); optional |
| `OBJECT_STORAGE_BACKEND` | `r2` (default) or `hippius` |
| `R2_ENDPOINT`, `R2_REGION`, `R2_SAMPLES_BUCKET`, `R2_SAMPLES_READ_*` | Default backend: R2 S3 API URL (e.g. `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`), region (often `auto`), bucket, and read keys for the evaluator |
| `HIPPIUS_*` | When `OBJECT_STORAGE_BACKEND=hippius`: endpoint, region, bucket names, and keys (see `env.example`) |

### Quick start (Docker, recommended)

This repo’s `docker-compose.yml` runs the **validator** (evaluator + weight-setter in one container) and optionally **Watchtower** for auto-updates.

```bash
# 1. Clone the repo
git clone https://github.com/RendixNetwork/leoma.git
cd leoma

# 2. Create .env from example
cp env.example .env
# Edit .env: API_URL, OPENAI_API_KEY, HIPPIUS_SAMPLES_READ_*, WALLET_NAME, HOTKEY_NAME, NETUID, NETWORK

# 3. Run
docker compose up -d
```

This starts **leoma-validator** (`leoma serve`: evaluator + weight-setter) and **leoma-watchtower**. Mount your Bittensor wallets so the container can sign weight-setting transactions; the compose file uses `~/.bittensor/wallets:/root/.bittensor/wallets:ro`.

**Auto-update with Watchtower:** Build and push the image on subnet code updates; Watchtower will pull and restart the validator container. See `env.example` for `WATCHTOWER_POLL_INTERVAL` and related options.

### Manual installation

```bash
# Requires Python 3.12+
pip install -e .   # or: uv pip install -e .

# Run validator (evaluator + weight-setter in one process)
leoma serve
```

### Split processes (advanced)

You can run evaluator and weight-setter as separate processes:

```bash
leoma servers evaluator   # Polls GET /tasks/latest, downloads from S3, GPT-4o, POSTs to API
leoma servers validator   # Every epoch: GET /weights, set on-chain
```

### API authentication

Endpoints that require validator identity (e.g. `POST /samples/batch`) use **signature auth**. Send headers: `X-Validator-Hotkey`, `X-Signature`, `X-Timestamp`. Message to sign: `SHA256(request_body):timestamp` (UTF-8), with your validator keypair. See the [API reference](https://docs.leoma.ai/api) in the docs.

---

## Miner setup

To run a **miner** on the Leoma subnet: upload your **Text-Image to Video (TI2V)** model to Hugging Face, deploy to Chutes, and commit on-chain.

### 1. Upload your model to Hugging Face

- Fine-tune or adapt a **TI2V** model, then upload it to [Hugging Face](https://huggingface.co/) as a model repository.
- **Model naming (required):** The repository name must **start with `leoma`** and **end with your miner hotkey** (SS58 address).  
  Example: `your_username/leoma-5F3sa2TJAWMqDhxG6jhV4N8ko9SxwGy8TpaNS1repo5DvT9`.
- **Revision (required):** Use a **specific revision** — the full Git **commit SHA** of the model version you deploy. Do not use branch names like `main`.

### 2. Deploy to Chutes and commit on-chain

1. **Deploy to Chutes** (so validators can call your model):
   ```bash
   leoma miner push --model-name <your-hf-repo> --model-revision <full-commit-sha> --chutes-api-key <api-key> --chute-user <chutes-username>
   ```
   Use the **full commit SHA** as `--model-revision`. Note the **Chute ID** from the output.

2. **Commit on-chain** (register model + Chute for validators):
   ```bash
   leoma miner commit --model-name <your-hf-repo> --model-revision <full-commit-sha> --chute-id <chute-id> --coldkey <wallet-name> --hotkey <ss58-address>
   ```
   Your wallet (coldkey/hotkey) must be registered on the subnet.

### 3. Monitor your miner

- **Network page (app):** View leaderboard, valid miners, and recent evaluations. Confirm your hotkey appears and is **valid**.
- **CLI:** Fetch the current rank list (same data as the dashboard):
  ```bash
  leoma get-rank
  ```
- **API:** `GET /miners/list`, `GET /miners/{hotkey}`, `GET /scores/rank` — check `is_valid`, `invalid_reason`, and `eligible` (completeness ≥ `SCORER_COMPLETENESS_THRESHOLD`, default **80%**, over the consecutive `SCORER_TASK_WINDOW` task_ids ending at max `task_id`).

---

## Storage and API

- **Storage:** Source videos live in the **source bucket**; task artifacts and evaluation results live in the **samples bucket**. The owner chooses **Hippius** or **Cloudflare R2** with `OBJECT_STORAGE_BACKEND` and the matching env vars (`HIPPIUS_*` or `R2_*`). Validators need **read-only** access to the samples bucket. See `env.example` and the [Storage](https://docs.leoma.ai/storage) doc.
- **API:** The Leoma API provides health, miners, samples, scores, tasks, weights, and blacklist endpoints. Validators use **GET /tasks/latest**, **POST /samples/batch**, and **GET /weights**. See the [API reference](https://docs.leoma.ai/api).

---

## Security and production

- **Dependencies:** Critical packages are pinned in `pyproject.toml`. Before production, run `pip audit` (or use Dependabot/Snyk) for known vulnerabilities.
- **Production env:** Set `LEOMA_ENV=production` (or `ENVIRONMENT=production`) so the API enforces non-default DB credentials and exception logs omit full tracebacks.
- **CORS:** Set `CORS_ORIGINS` to a comma-separated list of allowed frontend origins. Leave unset for development (allows `*`).
- **API auth:** Validator requests use hotkey signature auth; admin-only actions require hotkeys listed in `ADMIN_HOTKEYS`. See `env.example` for `SIGNATURE_EXPIRY_SECONDS` and related options.

---

## Documentation

Full documentation (getting started, miner setup, validator setup, storage, API reference):

- **Docs site:** [https://docs.leoma.ai](https://docs.leoma.ai)

Resources:

- **App / dashboard:** Leoma frontend (Overview, Product, Network, Docs, Help)
- **Whitepaper:** Protocol details and incentives
- **Community:** Discord, Twitter, GitHub (see the app Help page)

---

## License

MIT
