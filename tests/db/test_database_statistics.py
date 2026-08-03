from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import database


def test_statistics_are_derived_from_historical_records(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'statistics.db'}")
    database.Base.metadata.create_all(bind=engine)
    test_session = sessionmaker(bind=engine)

    with test_session() as session:
        session.add_all(
            [
                database.HistoricalData(
                    symbol="BTCUSDT__1d",
                    date=datetime(2023, 8, 4),
                    open_price=1,
                    high_price=2,
                    low_price=0.5,
                    close_price=1.5,
                    volume=10,
                ),
                database.HistoricalData(
                    symbol="BTCUSDT__1d",
                    date=datetime(2026, 8, 2),
                    open_price=2,
                    high_price=3,
                    low_price=1.5,
                    close_price=2.5,
                    volume=20,
                ),
            ]
        )
        session.commit()
        assert session.query(database.DataCache).count() == 0

    monkeypatch.setattr(database, "SessionLocal", test_session)

    assert database.db_manager.get_data_statistics() == [
        (
            "BTCUSDT__1d",
            2,
            datetime(2023, 8, 4),
            datetime(2026, 8, 2),
        )
    ]
