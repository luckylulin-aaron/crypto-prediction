import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from core.logger import get_logger

logger = get_logger(__name__)


def send_email(subject, body, to_emails, from_email, app_password, html_body=None):
    if not to_emails or not from_email or not app_password:
        logger.warning("Gmail credentials not set, skipping email notification.")
        return
    # Use multipart/alternative so email clients render either HTML (preferred) OR plain text (fallback),
    # instead of showing both versions back-to-back.
    msg = MIMEMultipart("alternative")
    msg["From"] = from_email
    # For privacy, only show sender in To, put all recipients in BCC
    msg["To"] = from_email
    msg["Bcc"] = ", ".join(to_emails)
    msg["Subject"] = subject
    # Plain-text fallback always included
    msg.attach(MIMEText(body or "", "plain"))
    # Optional HTML part for nicer formatting
    if html_body:
        msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_emails, msg.as_string())
        logger.info(f"Sent trading recommendations to {to_emails} (BCC)")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
