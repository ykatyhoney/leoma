// PM2 config for the Leoma king-of-the-hill VALIDATOR (runs on the validator host).
//
// Scans on-chain miner reveals, dispatches duels to the eval server (reached at
// EVAL_SERVER_URL — typically an SSH tunnel to the GPU box), crowns winners, and
// sets weights across the king chain. Fill wallet + bucket + Hippius Hub creds
// from your secrets manager; do not commit real secrets.
module.exports = {
  apps: [
    {
      name: "leoma-validator",
      script: "leoma",
      args: "serve",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      env: {
        NETWORK: "finney",
        NETUID: "99",
        WALLET_NAME: "default",
        HOTKEY_NAME: "default",

        // Where duels are dispatched (SSH-tunnel to the GPU eval box).
        EVAL_SERVER_URL: "http://localhost:9000",

        // Duel parameters live in chain.toml (the consensus surface), not here.
        LEOMA_KING_CHAIN_SIZE: "5",
        LEOMA_WEIGHT_INTERVAL: "300",
        LEOMA_BURN_UID: "0",

        // This validator's own state bucket (durable king state).
        R2_OWN_BUCKET: "",
        R2_OWN_WRITE_ACCESS_KEY: "",
        R2_OWN_WRITE_SECRET_KEY: "",
      },
    },
  ],
};
