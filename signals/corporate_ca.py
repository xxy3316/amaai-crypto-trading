"""Build a CA bundle that works behind corporate TLS interception.

WHY THIS EXISTS
---------------
This machine sits behind a TLS-inspecting proxy, so every HTTPS request from
Python fails with:

    SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
              self signed certificate in certificate chain

The proxy re-signs traffic with a private root CA that Windows trusts but that
`certifi` (which `requests` uses) does not. The fix is to hand `requests` a
bundle containing BOTH certifi's public roots and the Windows trust store.

Never "fix" this by passing verify=False. That disables certificate checking
entirely, and a paper whose data pipeline silently accepted any certificate is
a paper with an unverifiable data provenance claim.

USAGE
-----
    python -m signals.corporate_ca

Then export the printed path, e.g. in PowerShell:

    $env:REQUESTS_CA_BUNDLE = "$env:TEMP\\corp-ca-bundle.pem"

or persist it for future shells:

    setx REQUESTS_CA_BUNDLE "%TEMP%\\corp-ca-bundle.pem"

`requests` reads REQUESTS_CA_BUNDLE automatically, so every data source in this
package works afterwards with no code change.
"""

from __future__ import annotations

import base64
import os
import ssl
import sys
from pathlib import Path


def _certifi_pem() -> str:
    try:
        import certifi
    except ImportError:
        return ""
    return Path(certifi.where()).read_text(encoding="utf-8")


def _windows_roots() -> list:
    """DER-encoded certificates from the Windows trust stores.

    ssl.enum_certificates is stdlib and Windows-only; it reads the same stores
    the OS trusts, which is where the proxy's root CA was installed.
    """
    if not hasattr(ssl, "enum_certificates"):
        return []
    out = []
    for store in ("ROOT", "CA"):
        try:
            for der, _enc, _trust in ssl.enum_certificates(store):
                out.append(der)
        except Exception as e:  # noqa: BLE001
            print(f"  warning: could not read Windows store {store}: {e}",
                  file=sys.stderr)
    return out


def _der_to_pem(der: bytes) -> str:
    body = base64.b64encode(der).decode("ascii")
    lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN CERTIFICATE-----\n{lines}\n-----END CERTIFICATE-----\n"


def default_bundle_path() -> Path:
    """Stable location for the generated bundle, so it is built at most once."""
    root = os.getenv("EXOGENOUS_CACHE_ROOT", "data/exogenous")
    return Path(root) / "_certs" / "corp-ca-bundle.pem"


def ensure_bundle(destination: Path | None = None) -> Path | None:
    """Return a usable CA bundle path, building it only if not already present.

    Purely local: reads certifi and the Windows trust store, makes no network
    request. Returns None on a platform with no Windows cert store to merge,
    where the caller should simply keep the default verification.

    This is the auto-heal path used when a data fetch hits the corporate proxy.
    It is applied per-request (`session.verify`) rather than by setting
    REQUESTS_CA_BUNDLE globally, so components that already work -- ccxt price
    downloads, the Azure OpenAI client -- are left completely untouched.
    """
    destination = Path(destination) if destination else default_bundle_path()
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    if not hasattr(ssl, "enum_certificates"):
        return None
    try:
        return build_bundle(destination, quiet=True)
    except Exception as e:  # noqa: BLE001
        print(f"could not build a corporate CA bundle: {e}", file=sys.stderr)
        return None


def build_bundle(destination: Path | None = None, quiet: bool = False) -> Path:
    """Write a combined certifi + Windows-trust-store PEM and return its path."""
    destination = Path(destination) if destination else Path(
        os.getenv("TEMP", ".")) / "corp-ca-bundle.pem"

    parts = []
    base = _certifi_pem()
    if base:
        parts.append(base if base.endswith("\n") else base + "\n")

    ders = _windows_roots()
    seen = set()
    added = 0
    for der in ders:
        if der in seen:
            continue
        seen.add(der)
        parts.append(_der_to_pem(der))
        added += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(parts), encoding="ascii")

    if not quiet:
        print(f"certifi roots     : {'included' if base else 'certifi not installed'}")
        print(f"windows certs     : {added}")
        print(f"bundle            : {destination}  ({destination.stat().st_size} bytes)")
    return destination


def verify(bundle: Path, url: str = "https://data.binance.vision/") -> bool:
    try:
        import requests
    except ImportError:
        print("requests not installed; skipping verification", file=sys.stderr)
        return False
    try:
        resp = requests.get(url, timeout=30, verify=str(bundle))
        print(f"verification GET  : HTTP {resp.status_code} from {url}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"verification FAIL : {type(e).__name__}: {str(e)[:160]}",
              file=sys.stderr)
        return False


def main() -> int:
    bundle = build_bundle()
    ok = verify(bundle)
    print()
    if ok:
        print("Set this for the current PowerShell session:")
        print(f'  $env:REQUESTS_CA_BUNDLE = "{bundle}"')
        print("\nOr persist it for all future shells:")
        print(f'  setx REQUESTS_CA_BUNDLE "{bundle}"')
        return 0
    print("The bundle was written but verification still failed. The proxy root "
          "CA may not be in the Windows store either; ask IT for the root "
          "certificate PEM and append it to the bundle above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
