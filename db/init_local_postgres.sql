CREATE TABLE IF NOT EXISTS crawled_data (
    id BIGSERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    title VARCHAR(255) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    source_type VARCHAR(32) NOT NULL,
    language VARCHAR(16) NOT NULL DEFAULT 'ja',
    quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    license_tag TEXT NOT NULL DEFAULT 'unknown',
    allowed_for_training BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crawled_data_source_type ON crawled_data(source_type);
CREATE INDEX IF NOT EXISTS idx_crawled_data_created_at ON crawled_data(created_at);
CREATE INDEX IF NOT EXISTS idx_crawled_data_allowed_for_training ON crawled_data(allowed_for_training);

CREATE TABLE IF NOT EXISTS source_policies (
    id BIGSERIAL PRIMARY KEY,
    domain_pattern TEXT UNIQUE NOT NULL,
    source_type VARCHAR(32) NOT NULL DEFAULT 'web',
    license_tag TEXT NOT NULL DEFAULT 'unknown',
    allow_training BOOLEAN NOT NULL DEFAULT TRUE,
    base_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_source_policies_source_type ON source_policies(source_type);

CREATE TABLE IF NOT EXISTS system_nodes (
    node_id TEXT PRIMARY KEY,
    role VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'paused',
    cpu_usage DOUBLE PRECISION NOT NULL DEFAULT 0,
    ram_usage DOUBLE PRECISION NOT NULL DEFAULT 0,
    target_status VARCHAR(32) NOT NULL DEFAULT 'unspecified',
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id BIGSERIAL PRIMARY KEY,
    model_tag TEXT NOT NULL DEFAULT 'default',
    avg_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    result_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    model_tag TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    source_checkpoint TEXT NOT NULL DEFAULT '',
    avg_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    promoted BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at DESC);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id BIGSERIAL PRIMARY KEY,
    dataset_tag TEXT NOT NULL,
    train_tokens BIGINT NOT NULL DEFAULT 0,
    val_tokens BIGINT NOT NULL DEFAULT 0,
    total_docs BIGINT NOT NULL DEFAULT 0,
    blocked_docs BIGINT NOT NULL DEFAULT 0,
    filtered_docs BIGINT NOT NULL DEFAULT 0,
    duplicate_docs BIGINT NOT NULL DEFAULT 0,
    source_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_created_at ON dataset_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crawled_data_fts
    ON crawled_data USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')));
