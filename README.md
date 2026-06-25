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

Task generation and score aggregation are **fully decentralized** — every permissioned validator runs its own sampler and aggregates scores locally. The owner-api is only a thin coordinator (rotation clock + permissioned allowlist + dashboard).

1. **Rotation:** the owner-api `GET /rotation` is the authoritative clock. Sampling rotates across the permissioned-validator allowlist every `SAMPLING_ROTATION_INTERVAL` blocks (default 100 ≈ 20 min); only one validator samples per window, so miners never get concurrent requests. `task_id = current_block // interval`.
2. **Validation (every validator):** each validator independently reads the chain (commitments + metagraph) and validates each miner — commit rules (model named `leoma…<hotkey>`), HuggingFace model hash, Chute hot, duplicate-model detection — then reports the result to the owner-api (`POST /miners/report`). The owner-api tallies a **majority consensus** into `valid_miners` for the dashboard; it no longer validates miners itself.
3. **Sampler + self-evaluator (each validator on its turn):** samples its **locally-validated** miners — picks a one-shot 5s clip + first frame from the source bucket, describes it with Gemini, calls each miner's Chute, uploads the task artifacts to its **own** R2 bucket, then **evaluates its own task** (no cross-validation) and publishes `evaluation_results/<hotkey>.json` to its bucket + dual-reports to the API for the dashboard, and announces it (`POST /tasks/announce`).
4. **Weight-setter (every validator):** each epoch reads all peers' verdicts from their buckets (read keys shared peer-to-peer via `PEER_VALIDATORS`) and aggregates them with **per-validator-average equal weight** — each validator's own pass-rate per miner, then the mean across validators (no validator counts more for sampling more) — ranks miners (dominance rule), and sets winner-take-all weights on-chain.
5. **Miners** register a Hugging Face model (naming: `leoma` prefix, hotkey suffix) and Chute endpoint via on-chain commit. They receive challenges; the best performer earns subnet alpha.

| Role | In Leoma |
|------|----------|
| **Miner** | Upload a TI2V model to Hugging Face (name: `leoma...` + your hotkey), deploy to Chutes, commit on-chain. Earn subnet alpha when your outputs win. |
| **Validator** | Run miner-validation + sampler (self-evaluating) + weight-setter (e.g. `leoma serve`). Requires API URL, an **own R2 bucket** (write) + **peer bucket** read keys, source-bucket read keys, Chutes + Gemini API keys (+ optional `HF_TOKEN` for gated models), and a Bittensor wallet. Must be on the owner's permissioned allowlist. |

---

## Validator setup

Validators run the **sampler** (which self-evaluates its own tasks) and the **weight-setter** in one process (`leoma serve`). Task generation and score aggregation are decentralized — each validator samples on its turn and aggregates peers' results locally.

### Prerequisites

- **Bittensor wallet** (coldkey + hotkey) registered as a validator on the Leoma subnet.
- **Leoma API** URL (the deployed owner-api coordinator).
- **Permissioned allowlist:** the subnet owner must add your hotkey to `PERMISSIONED_VALIDATORS` so you can read `GET /rotation` and announce tasks.
- **Own R2 bucket (write):** `R2_OWN_BUCKET` + `R2_OWN_WRITE_*`. The sampler publishes task artifacts and its own verdicts here.
- **Peer bucket read keys:** `PEER_VALIDATORS` — a JSON list of every permissioned validator (including you) with read-only creds, shared peer-to-peer. Used to aggregate scores.
- **Source bucket read keys:** `R2_VIDEOS_READ_*` + `R2_SOURCE_BUCKET` — the owner shares one read-only key pair so the sampler can fetch source clips.
- **Chutes API key** (`CHUTES_API_KEY`) for the sampler to call miners and for miner validation (Chute-hot check); **Gemini API key** (`GEMINI_API_KEY`) for clip description + evaluation; optional **`HF_TOKEN`** for validating gated/private HuggingFace models.
- **Validator registration in API DB:** validators with stake ≥ `MIN_VALIDATOR_STAKE` are synced automatically from the metagraph; an admin can also add one manually (e.g. `leoma db add-validator --uid <uid> --hotkey <ss58>`).

### Environment variables

Set these in `.env` (copy from `env.example`):

| Variable | Description |
|----------|-------------|
| `API_URL` | Leoma API base URL (e.g. `https://api.leoma.ai`) |
| `NETUID` | Subnet ID (e.g. `99`) |
| `NETWORK` | Bittensor network (`finney` for mainnet) |
| `WALLET_NAME` | Bittensor wallet name (e.g. `default`) |
| `HOTKEY_NAME` | Bittensor hotkey name (e.g. `default`) |
| `GEMINI_API_KEY` | Gemini key (clip description in the sampler + video evaluation) |
| `CHUTES_API_KEY` | Chutes key (sampler calls each miner's I2V Chute) |
| `EPOCH_LEN` | Blocks per epoch (e.g. `180`); optional |
| `R2_ENDPOINT`, `R2_REGION` | R2 S3 API URL (e.g. `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`) and region (often `auto`) |
| `R2_SOURCE_BUCKET`, `R2_VIDEOS_READ_*` | Source bucket + read keys (sampler downloads source clips) |
| `R2_OWN_BUCKET`, `R2_OWN_WRITE_*` | This validator's own result bucket + write keys (sampler publishes tasks + verdicts here) |
| `PEER_VALIDATORS` | JSON list of all permissioned validators with bucket + read keys (peer aggregation) |

### Quick start (Docker, recommended)

This repo’s `docker-compose.yml` runs the **validator** (sampler + weight-setter in one container) and optionally **Watchtower** for auto-updates.

```bash
# 1. Clone the repo
git clone https://github.com/RendixNetwork/leoma.git
cd leoma

# 2. Create .env from example
cp env.example .env
# Edit .env: API_URL, GEMINI_API_KEY, CHUTES_API_KEY, R2_OWN_BUCKET, R2_OWN_WRITE_*,
#            PEER_VALIDATORS, R2_VIDEOS_READ_*, WALLET_NAME, HOTKEY_NAME, NETUID, NETWORK

# 3. Run
docker compose up -d
```

This starts **leoma-validator** (`leoma serve`: sampler (self-evaluating) + weight-setter) and **leoma-watchtower**. Mount your Bittensor wallets so the container can sign weight-setting transactions; the compose file uses `~/.bittensor/wallets:/root/.bittensor/wallets:ro`.

**Auto-update with Watchtower:** Build and push the image on subnet code updates; Watchtower will pull and restart the validator container. See `env.example` for `WATCHTOWER_POLL_INTERVAL` and related options.

### Manual installation

```bash
# Requires Python 3.12+
pip install -e .   # or: uv pip install -e .

# Run validator (sampler with self-eval + weight-setter in one process)
leoma serve
```

### Split processes (advanced)

You can run the loops as separate processes:

```bash
leoma servers sampler     # On your turn: sample miners, self-evaluate, publish to own bucket, announce
leoma servers validator   # Every epoch: aggregate peers' verdicts (per-validator average), set on-chain
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

- **Storage:** Source videos live in one shared **source bucket** (validators get read-only keys via `R2_VIDEOS_READ_*`). Each validator owns its own **result bucket** (`R2_OWN_BUCKET`) where its sampled tasks and evaluation results live; validators share read-only keys to each other's buckets via `PEER_VALIDATORS` to aggregate scores. Decentralized buckets use **Cloudflare R2**. See `env.example` and the [Storage](https://docs.leoma.ai/storage) doc.
- **API (coordinator):** The owner-api provides health, miners, samples (dashboard), scores, tasks, rotation, and blacklist endpoints. Validators use **GET /rotation** (whose turn), **POST /tasks/announce**, **GET /tasks/latest**, **POST /samples/batch** (dashboard dual-report), and **POST /miners/report** (miner-validation results). Miner validity and on-chain weights are computed by the validators — the owner-api only tallies the **majority consensus** of miner reports for the dashboard and does not validate miners or fetch weights. See the [API reference](https://docs.leoma.ai/api).
- **Dashboard endpoints (decentralized):** **GET /overview** (network snapshot), **GET /miners/active** (valid + chute-hot miners with per-validator-average score, rank, eligibility, activity), **GET /validators** and **GET /validators/{hotkey}** (each validator's liveness + rotation participation; stake is informational, not used for weighting), and **GET /scores** (per-validator-average leaderboard). Task media is presigned from the sampler's bucket (the API needs `PEER_VALIDATORS` read keys for previews).

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
