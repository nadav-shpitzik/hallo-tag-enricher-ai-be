-- Migration: Add approval workflow tables and columns
-- This creates lecture_tag_suggestions with full schema and adds audit/sync tables

-- 1. Create or extend lecture_tag_suggestions table
CREATE TABLE IF NOT EXISTS lecture_tag_suggestions (
    suggestion_id       BIGSERIAL PRIMARY KEY,
    lecture_id          BIGINT NOT NULL,
    lecture_external_id VARCHAR NOT NULL,
    tag_id              VARCHAR NOT NULL,
    tag_name_he         VARCHAR NOT NULL,
    score               NUMERIC(5,4) NOT NULL,
    rationale           TEXT,
    sources             JSONB DEFAULT '["title","description"]',
    model               TEXT NOT NULL,
    status              VARCHAR NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'synced', 'failed')),
    approved_by         VARCHAR,
    approved_at         TIMESTAMP,
    synced_at           TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (lecture_id, tag_id)
);

-- For existing installations, add missing columns
ALTER TABLE lecture_tag_suggestions
ADD COLUMN IF NOT EXISTS approved_by VARCHAR,
ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS synced_at TIMESTAMP;

-- 2. Create suggestion_events table for audit trail
CREATE TABLE IF NOT EXISTS suggestion_events (
    event_id        BIGSERIAL PRIMARY KEY,
    suggestion_id   BIGINT NOT NULL REFERENCES lecture_tag_suggestions(suggestion_id) ON DELETE CASCADE,
    action          VARCHAR NOT NULL CHECK (action IN ('approve', 'reject', 'enqueue', 'sync_ok', 'sync_fail', 'retry')),
    actor           VARCHAR,
    previous_status VARCHAR,
    new_status      VARCHAR,
    details         JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create airtable_sync_items queue table
CREATE TABLE IF NOT EXISTS airtable_sync_items (
    sync_item_id        BIGSERIAL PRIMARY KEY,
    lecture_external_id VARCHAR NOT NULL,
    tag_id              VARCHAR NOT NULL,
    suggestion_id       BIGINT NOT NULL REFERENCES lecture_tag_suggestions(suggestion_id) ON DELETE CASCADE,
    status              VARCHAR NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    attempt             INTEGER DEFAULT 0,
    last_error          TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (lecture_external_id, tag_id)
);

-- 4. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sugg_status ON lecture_tag_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_sugg_lecture ON lecture_tag_suggestions(lecture_id);
CREATE INDEX IF NOT EXISTS idx_sugg_external_id ON lecture_tag_suggestions(lecture_external_id);
CREATE INDEX IF NOT EXISTS idx_events_suggestion ON suggestion_events(suggestion_id);
CREATE INDEX IF NOT EXISTS idx_events_created ON suggestion_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sync_items_status ON airtable_sync_items(status);
CREATE INDEX IF NOT EXISTS idx_sync_items_lecture ON airtable_sync_items(lecture_external_id);

-- 5. Create materialized view for fast UI queries
CREATE MATERIALIZED VIEW IF NOT EXISTS v_pending_lectures AS
SELECT 
    l.id AS lecture_id,
    l.lecture_external_id,
    l.lecture_title,
    l.lecturer_name,
    COUNT(s.suggestion_id) AS suggestion_count,
    COUNT(CASE WHEN s.status = 'pending' THEN 1 END) AS pending_count,
    COUNT(CASE WHEN s.status = 'approved' THEN 1 END) AS approved_count,
    COUNT(CASE WHEN s.status = 'rejected' THEN 1 END) AS rejected_count,
    COUNT(CASE WHEN s.status = 'synced' THEN 1 END) AS synced_count,
    COUNT(CASE WHEN s.status = 'failed' THEN 1 END) AS failed_count,
    AVG(s.score) AS avg_score,
    MAX(s.created_at) AS latest_suggestion_at
FROM enriched_lectures l
INNER JOIN lecture_tag_suggestions s ON l.id = s.lecture_id
GROUP BY l.id, l.lecture_external_id, l.lecture_title, l.lecturer_name;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_v_pending_lectures_id ON v_pending_lectures(lecture_id);

-- 6. Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_pending_lectures_view()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY v_pending_lectures;
END;
$$ LANGUAGE plpgsql;

-- 7. Add updated_at trigger for airtable_sync_items
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_sync_items_updated_at
    BEFORE UPDATE ON airtable_sync_items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Migration complete
