import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from core.logger import get_logger

logger = get_logger(__name__)

def send_email(subject, body, to_emails, from_email, app_password):
    if not to_emails or not from_email or not app_password:
        logger.warning("Gmail credentials not set, skipping email notification.")
        return
    msg = MIMEMultipart()
    msg['From'] = from_email
    # For privacy, only show sender in To, put all recipients in BCC
    msg['To'] = from_email
    msg['Bcc'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_emails, msg.as_string())
        logger.info(f"Sent trading recommendations to {to_emails} (BCC)")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")