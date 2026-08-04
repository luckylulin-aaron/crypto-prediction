"""Delivery orchestration for durable daily signal notifications."""


class NotificationService:
    def __init__(self, signal_repository, renderer, logger):
        self._signal_repository = signal_repository
        self._renderer = renderer
        self._logger = logger

    def send_daily(
        self,
        *,
        log_file,
        recipient_list,
        from_email,
        app_password,
        best_summaries=None,
    ) -> bool:
        try:
            pending_signals = self._signal_repository.pending()
        except Exception as exc:
            self._logger.error(
                f"Failed to read pending signal ledger; using log fallback: {exc}"
            )
            pending_signals = None

        delivered = self._renderer(
            log_file,
            recipient_list,
            from_email,
            app_password,
            best_summaries=best_summaries,
            pending_signals=pending_signals,
        )

        if pending_signals:
            signal_ids = [row["id"] for row in pending_signals]
            try:
                self._signal_repository.mark_delivery(
                    signal_ids,
                    success=bool(delivered),
                    error=None if delivered else "admin email delivery failed",
                )
                self._logger.info(
                    f"Signal ledger delivery recorded: signals={len(signal_ids)}, "
                    f"success={bool(delivered)}"
                )
            except Exception as exc:
                self._logger.error(
                    f"Failed to update signal ledger delivery status: {exc}"
                )
                return False
        return bool(delivered)
