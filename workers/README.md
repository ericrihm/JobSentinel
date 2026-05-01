# JobSentinel API — Cloudflare Worker

This directory contains the Cloudflare Worker that powers the `api.jobsentinel.org` backend. It handles job scan requests, rate limiting via KV, and persists scan history in D1.

---

## Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/): `npm install -g wrangler`
- A Cloudflare account with the `jobsentinel.org` zone active

---

## One-Time Setup

### 1. Authenticate with Cloudflare

```bash
wrangler login
```

Verify your account:

```bash
wrangler whoami
```

Copy your **Account ID** and paste it into `wrangler.toml` (uncomment the `account_id` line).

---

### 2. Create the KV Namespace (rate limiting)

```bash
wrangler kv:namespace create "RATE_LIMIT"
```

The command outputs something like:

```
{ binding = "RATE_LIMIT", id = "abc123...", preview_id = "def456..." }
```

Open `wrangler.toml` and replace the placeholder values:

```toml
[[kv_namespaces]]
binding = "RATE_LIMIT"
id = "abc123..."          # from the command above
preview_id = "def456..."  # from the command above
```

---

### 3. Create the D1 Database

```bash
wrangler d1 create jobsentinel-db
```

The command outputs:

```
{ database_name = "jobsentinel-db", database_id = "ghi789..." }
```

Open `wrangler.toml` and replace the placeholder:

```toml
[[d1_databases]]
binding = "DB"
database_name = "jobsentinel-db"
database_id = "ghi789..."  # from the command above
```

---

### 4. Initialize the Remote Schema

```bash
npm run db:init:remote
```

This runs `wrangler d1 execute` against the live D1 instance using `src/schema.sql`.

---

## Deploy

```bash
npm run deploy
```

This runs `wrangler deploy`. On success, Wrangler prints the worker URL (e.g., `https://jobsentinel-api.<your-subdomain>.workers.dev`).

---

## Custom Domain Setup (api.jobsentinel.org)

1. In the [Cloudflare dashboard](https://dash.cloudflare.com), go to **Workers & Pages**.
2. Select **jobsentinel-api**.
3. Open the **Triggers** tab and click **Add Custom Domain**.
4. Enter `api.jobsentinel.org` and confirm.

Cloudflare will automatically provision a TLS certificate and create the DNS record. The `[[routes]]` block in `wrangler.toml` is also set so that `wrangler deploy` routes traffic from `api.jobsentinel.org/*` to the worker.

---

## Environment Variables

Sensitive values (API keys, secrets) must be stored as Worker Secrets — never in `wrangler.toml`.

```bash
# Set a secret (prompts for the value interactively)
wrangler secret put SECRET_NAME

# List configured secrets
wrangler secret list

# Delete a secret
wrangler secret delete SECRET_NAME
```

| Variable / Secret | Description |
|---|---|
| `ENVIRONMENT` | Set to `production` in `wrangler.toml` [vars] |
| `API_VERSION` | Current API version string |

Add any additional secrets your worker reads via `env.SECRET_NAME` using the commands above.

---

## Monitoring & Logs

Stream live logs from the deployed worker:

```bash
npm run logs
# equivalent to: wrangler tail
```

Filter by status or sampling rate:

```bash
wrangler tail --status error
wrangler tail --sampling-rate 0.1
```

View metrics (requests, errors, CPU time) in the Cloudflare dashboard under **Workers & Pages → jobsentinel-api → Metrics**.

---

## Schema Migrations

When `src/schema.sql` changes, apply it to the remote D1 database:

```bash
npm run db:init:remote
```

For incremental migrations (ALTER TABLE, etc.), run them directly:

```bash
wrangler d1 execute jobsentinel-db --remote --command "ALTER TABLE scans ADD COLUMN foo TEXT"
```

---

## Local Development

```bash
npm run dev
# equivalent to: wrangler dev
```

The worker runs locally at `http://localhost:8787` with a local KV and D1 instance. No remote resources are touched.

---

## CI/CD

Pushes to `main` that change files under `workers/` automatically deploy via the GitHub Actions workflow at `.github/workflows/deploy-worker.yml`. The workflow also runs the schema migration when `src/schema.sql` changes.

Required repository secret: `CLOUDFLARE_API_TOKEN`

Generate a token in the Cloudflare dashboard: **My Profile → API Tokens → Create Token → Edit Cloudflare Workers** template.
