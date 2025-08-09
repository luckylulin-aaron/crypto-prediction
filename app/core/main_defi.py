import os
import configparser
import sys

# Custom imports with fallback for script/module execution
try:
    from core.config import DEBUG
    from data.defi_event_client import DefiEventClient
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.config import DEBUG
    from data.defi_event_client import DefiEventClient

def main():
    # Read from secret.ini
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "secret.ini")
    config.read(config_path)
    section = "CONFIG"
    def get_secret(key, env_fallback=None):
        if config.has_option(section, key):
            return config.get(section, key).strip('"')
        return os.environ.get(env_fallback or key, "")

    to_emails = get_secret("DEFI_REPORT_TO_EMAILS").split(",")
    from_email = get_secret("DEFI_REPORT_FROM_EMAIL")
    app_password = get_secret("DEFI_REPORT_APP_PASSWORD")
    to_emails = [e for e in to_emails if e]
    if DEBUG and to_emails:
        to_emails = to_emails[:1]
    client = DefiEventClient()
    client.run_and_email(to_emails, from_email, app_password, top_n=3)

if __name__ == "__main__":
    main() 