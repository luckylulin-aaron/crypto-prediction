-- Initialize database for crypto trading bot
-- This script runs when the PostgreSQL container starts

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_date 
ON historical_data (symbol, date);

CREATE INDEX IF NOT EXISTS idx_historical_data_date 
ON historical_data (date);

CREATE INDEX IF NOT EXISTS idx_data_cache_symbol 
ON data_cache (symbol);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_historical_data_updated_at 
    BEFORE UPDATE ON historical_data 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres; 