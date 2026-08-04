"""Non-destructive baseline for historical cache and signal ledger."""

from alembic import op
import sqlalchemy as sa

revision = "20260804_0001"
down_revision = None
branch_labels = None
depends_on = None


def _tables():
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    tables = _tables()
    if "historical_data" not in tables:
        op.create_table(
            "historical_data",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("symbol", sa.String(20), nullable=False),
            sa.Column("date", sa.DateTime(), nullable=False),
            sa.Column("open_price", sa.Float(), nullable=False),
            sa.Column("high_price", sa.Float(), nullable=False),
            sa.Column("low_price", sa.Float(), nullable=False),
            sa.Column("close_price", sa.Float(), nullable=False),
            sa.Column("volume", sa.Float(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
        op.create_index("ix_historical_data_id", "historical_data", ["id"])
        op.create_index("ix_historical_data_symbol", "historical_data", ["symbol"])
        op.create_index("ix_historical_data_date", "historical_data", ["date"])

    if "data_cache" not in tables:
        op.create_table(
            "data_cache",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("symbol", sa.String(20), nullable=False),
            sa.Column("last_updated", sa.DateTime(), nullable=False),
            sa.Column("data_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cache_key", sa.String(100), nullable=False, unique=True),
            sa.Column("created_at", sa.DateTime()),
        )
        op.create_index("ix_data_cache_id", "data_cache", ["id"])
        op.create_index("ix_data_cache_symbol", "data_cache", ["symbol"])

    if "signal_ledger" not in tables:
        op.create_table(
            "signal_ledger",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("asset_type", sa.String(20), nullable=False),
            sa.Column("exchange", sa.String(40), nullable=False),
            sa.Column("asset", sa.String(20), nullable=False),
            sa.Column("strategy", sa.String(100), nullable=False),
            sa.Column("strategy_version", sa.String(40), nullable=False),
            sa.Column("signal_date", sa.DateTime(), nullable=False),
            sa.Column("action", sa.String(10), nullable=False),
            sa.Column("buy_percentage", sa.Float()),
            sa.Column("sell_percentage", sa.Float()),
            sa.Column("delivery_status", sa.String(20), nullable=False),
            sa.Column("delivery_attempts", sa.Integer(), nullable=False),
            sa.Column("first_seen_at", sa.DateTime(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(), nullable=False),
            sa.Column("delivered_at", sa.DateTime()),
            sa.Column("last_error", sa.Text()),
            sa.UniqueConstraint(
                "asset_type",
                "asset",
                "strategy",
                "strategy_version",
                "signal_date",
                "action",
                name="uq_signal_ledger_identity",
            ),
        )
        op.create_index("ix_signal_ledger_id", "signal_ledger", ["id"])
        op.create_index("ix_signal_ledger_asset", "signal_ledger", ["asset"])
        op.create_index("ix_signal_ledger_asset_type", "signal_ledger", ["asset_type"])
        op.create_index(
            "ix_signal_ledger_delivery_status", "signal_ledger", ["delivery_status"]
        )
        op.create_index(
            "ix_signal_ledger_signal_date", "signal_ledger", ["signal_date"]
        )

    if "signal_ledger_checkpoint" not in tables:
        op.create_table(
            "signal_ledger_checkpoint",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("asset_type", sa.String(20), nullable=False),
            sa.Column("asset", sa.String(20), nullable=False),
            sa.Column("strategy", sa.String(100), nullable=False),
            sa.Column("strategy_version", sa.String(40), nullable=False),
            sa.Column("last_evaluated_candle", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "asset_type",
                "asset",
                "strategy",
                "strategy_version",
                name="uq_signal_ledger_checkpoint",
            ),
        )
        op.create_index(
            "ix_signal_ledger_checkpoint_id", "signal_ledger_checkpoint", ["id"]
        )


def downgrade() -> None:
    # Baseline migrations never delete user market history or notification state.
    pass
