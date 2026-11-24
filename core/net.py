"""Network / TLS helpers.

Some networks (corporate proxies, antivirus HTTPS inspection) re-sign TLS traffic with
a root certificate that lives only in the Windows certificate store. Python's
``certifi`` bundle and ``curl_cffi`` (used by yfinance) don't trust it, so every
request fails with ``CERTIFICATE_VERIFY_FAILED``.

``configure_ssl()`` builds a CA bundle = certifi + OS trust store and points
``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE`` at it. TLS verification
is NEVER disabled.
"""

from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path

from loguru import logger

_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def build_ca_bundle(dest: Path) -> Path:
    """Write certifi's bundle plus the OS trust store (Windows ROOT/CA) to ``dest``."""
    import certifi

    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    if sys.platform == "win32":
        for store in ("ROOT", "CA"):
            try:
                for cert, encoding, _trust in ssl.enum_certificates(store):  # type: ignore[attr-defined]
                    if encoding == "x509_asn":
                        parts.append(ssl.DER_cert_to_PEM_cert(cert))
            except OSError as exc:
                logger.warning(f"Could not read Windows cert store {store}: {exc}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts), encoding="utf-8")
    return dest


def configure_ssl(runtime_dir: Path, *, use_os_store: bool = True, ca_bundle: str | None = None) -> str | None:
    """Configure process-wide CA bundle env vars.

    Args:
        runtime_dir: directory for the generated bundle.
        use_os_store: merge the OS trust store into the bundle (Windows only).
        ca_bundle: explicit PEM path (overrides everything, e.g. from ``.env CA_BUNDLE_PATH``).

    Returns:
        The bundle path in use, or ``None`` if the defaults were left untouched.
    """
    path: str | None = None
    if ca_bundle:
        path = ca_bundle
    elif use_os_store and sys.platform == "win32":
        try:
            path = str(build_ca_bundle(runtime_dir / "ca_bundle.pem"))
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.warning(f"CA bundle build failed, using defaults: {exc}")
            path = None
    if path:
        for var in _ENV_VARS:
            os.environ[var] = path
        logger.debug(f"TLS CA bundle: {path}")
    return path
