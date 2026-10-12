PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS scope_manifest (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    user_id TEXT NOT NULL,
    pipeline_id TEXT NOT NULL,
    embedding_space_id TEXT NOT NULL,
    dimension INTEGER CHECK (dimension IS NULL OR dimension > 0),
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'retired')),
    active_epoch INTEGER NOT NULL DEFAULT 0 CHECK (active_epoch >= 0),
    active_message_count INTEGER NOT NULL DEFAULT 0 CHECK (active_message_count >= 0),
    full_render_tokens_est INTEGER NOT NULL DEFAULT 0 CHECK (full_render_tokens_est >= 0),
    first_seen_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload_hash BLOB NOT NULL CHECK (length(payload_hash) = 32),
    state TEXT NOT NULL CHECK (state IN ('building', 'retryable_failed', 'ready')),
    build_fence INTEGER NOT NULL DEFAULT 0 CHECK (build_fence >= 0),
    build_owner TEXT,
    build_lease_until_ms INTEGER,
    received_ms INTEGER NOT NULL,
    published_epoch INTEGER,
    receipt_json TEXT,
    error_class TEXT,
    CHECK (
        (state = 'ready' AND published_epoch IS NOT NULL AND receipt_json IS NOT NULL)
        OR (state != 'ready' AND published_epoch IS NULL AND receipt_json IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS requests_session ON requests(session_id, published_epoch);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY,
    public_source_id TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL REFERENCES requests(request_id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    timestamp_provided INTEGER NOT NULL CHECK (timestamp_provided IN (0, 1)),
    source_timestamp_ms INTEGER,
    raw_content TEXT NOT NULL CHECK (length(raw_content) > 0),
    reader_token_est INTEGER NOT NULL DEFAULT 0 CHECK (reader_token_est >= 0),
    CHECK (
        (timestamp_provided = 0 AND source_timestamp_ms IS NULL)
        OR (timestamp_provided = 1 AND source_timestamp_ms IS NOT NULL)
    ),
    UNIQUE(request_id, ordinal)
);
CREATE INDEX IF NOT EXISTS messages_request_order ON messages(request_id, ordinal);
CREATE INDEX IF NOT EXISTS messages_source_time ON messages(source_timestamp_ms, message_id);

CREATE TABLE IF NOT EXISTS embedding_inputs (
    input_id INTEGER PRIMARY KEY,
    embedding_space_id TEXT NOT NULL,
    input_hash BLOB NOT NULL CHECK (length(input_hash) = 32),
    input_text TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'running', 'retry_wait', 'ready', 'permanent_error', 'cancelled')
    ),
    fence INTEGER NOT NULL DEFAULT 0 CHECK (fence >= 0),
    lease_owner TEXT,
    lease_until_ms INTEGER,
    retry_at_ms INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    dense_blob BLOB,
    sparse_present INTEGER NOT NULL DEFAULT 0 CHECK (sparse_present IN (0, 1)),
    last_error TEXT,
    CHECK (dense_blob IS NULL OR length(dense_blob) > 0),
    CHECK (state != 'ready' OR dense_blob IS NOT NULL),
    UNIQUE(embedding_space_id, input_hash)
);
CREATE INDEX IF NOT EXISTS embedding_inputs_recovery
    ON embedding_inputs(state, retry_at_ms, lease_until_ms);

CREATE TABLE IF NOT EXISTS windows (
    window_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests(request_id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    scale TEXT NOT NULL CHECK (scale IN ('leaf', 'context')),
    message_id INTEGER NOT NULL REFERENCES messages(message_id),
    input_id INTEGER NOT NULL REFERENCES embedding_inputs(input_id),
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char > start_char),
    proxy_token_count INTEGER NOT NULL CHECK (proxy_token_count > 0),
    lexical_en TEXT NOT NULL DEFAULT '',
    lexical_zh TEXT NOT NULL DEFAULT '',
    published_epoch INTEGER,
    UNIQUE(request_id, message_id, scale, ordinal),
    UNIQUE(window_id, input_id, scale)
);
CREATE INDEX IF NOT EXISTS windows_request ON windows(request_id, scale, ordinal);
CREATE INDEX IF NOT EXISTS windows_message_span ON windows(message_id, start_char, end_char);
CREATE INDEX IF NOT EXISTS windows_published ON windows(published_epoch, scale, window_id);

CREATE TABLE IF NOT EXISTS active_windows (
    window_id INTEGER PRIMARY KEY,
    input_id INTEGER NOT NULL,
    scale TEXT NOT NULL,
    published_epoch INTEGER NOT NULL,
    FOREIGN KEY (window_id, input_id, scale)
        REFERENCES windows(window_id, input_id, scale)
);
CREATE INDEX IF NOT EXISTS active_input ON active_windows(input_id, published_epoch, window_id);

CREATE TABLE IF NOT EXISTS active_input_counts (
    input_id INTEGER PRIMARY KEY REFERENCES embedding_inputs(input_id),
    n_windows INTEGER NOT NULL CHECK (n_windows > 0)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_en USING fts5(tokens, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_zh USING fts5(tokens, tokenize='unicode61');
