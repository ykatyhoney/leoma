// PM2 config for the Leoma EVAL SERVER (runs on the GPU box).
//
// Downloads king + challenger weights from Hippius Hub by digest, loads the
// pinned diffusers I2V pipeline, generates on the deterministic held-out clips,
// and scores each generation against the real continuation. One duel at a time.
// The validator reaches this over EVAL_SERVER_URL (SSH tunnel to :9000).
module.exports = {
  apps: [
    {
      name: "leoma-eval-server",
      script: "leoma",
      args: "servers eval-server",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      env: {
        EVAL_SERVER_HOST: "0.0.0.0",
        EVAL_SERVER_PORT: "9000",

        // Hippius Hub (OCI model registry) auth — token OR username/password.
        HIPPIUS_HUB_TOKEN: "",
        HIPPIUS_HUB_USERNAME: "",
        HIPPIUS_HUB_PASSWORD: "",
        LEOMA_MODEL_CACHE_DIR: "/tmp/leoma/hippius_models",

        // Source-video corpus read creds (ground-truth continuations).
        R2_VIDEOS_READ_ACCESS_KEY: "",
        R2_VIDEOS_READ_SECRET_KEY: "",
      },
    },
  ],
};
