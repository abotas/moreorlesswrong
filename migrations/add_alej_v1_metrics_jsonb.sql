-- Safe migration to add alej_v1_metrics JSONB column to fellowship_mvp table
-- This stores flattened metric scores from data/post_metrics/mini1000/*.json

-- First, check if column already exists to make this idempotent
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'fellowship_mvp' 
        AND column_name = 'alej_v1_metrics'
    ) THEN
        -- Add the JSONB column for metrics
        ALTER TABLE fellowship_mvp 
        ADD COLUMN alej_v1_metrics JSONB;
        
        -- Add timestamp to track when metrics were imported
        ALTER TABLE fellowship_mvp 
        ADD COLUMN alej_v1_metrics_imported_at TIMESTAMP WITH TIME ZONE;
        
        -- Create GIN index for efficient JSONB queries
        CREATE INDEX idx_alej_v1_metrics ON fellowship_mvp USING GIN (alej_v1_metrics);
        
        RAISE NOTICE 'Successfully added alej_v1_metrics column';
    ELSE
        RAISE NOTICE 'Column alej_v1_metrics already exists, skipping';
    END IF;
END $$;