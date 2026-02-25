"""Claude SDK authentication for trading engine.

Auth priority:
1. ANTHROPIC_API_KEY env var (server/cloud — no expiry, recommended)
2. API profile (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)
3. CLAUDE_CODE_OAUTH_TOKEN env var (local dev)
4. Credentials file (~/.claude/.credentials.json) with auto-refresh
5. System credential store (macOS Keychain, Windows creds)

NOTE: OAuth token refresh from datacenter IPs is blocked by Cloudflare
(403 error 1010). Use ANTHROPIC_API_KEY for server/cloud deployments.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger("trading-engine.claude_auth")

# Claude Code OAuth client ID (public, same for all users)
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/api/oauth/token"

AUTH_TOKEN_ENV_VARS = [
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
]

_SDK_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "NO_PROXY",
    "DISABLE_TELEMETRY",
    "DISABLE_COST_WARNINGS",
    "API_TIMEOUT_MS",
    "CLAUDE_CODE_GIT_BASH_PATH",
    "CLAUDE_CLI_PATH",
    "CLAUDE_CONFIG_DIR",
]


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _is_linux() -> bool:
    return platform.system() == "Linux"


# ------------------------------------------------------------------
# Credentials file handling + auto-refresh
# ------------------------------------------------------------------

def _credentials_file_paths() -> list[str]:
    """Candidate credentials file paths for the current platform."""
    paths = []
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        paths.append(os.path.join(config_dir, ".credentials.json"))
    if _is_linux() or _is_macos():
        paths += [
            os.path.expanduser("~/.claude/.credentials.json"),
            os.path.expanduser("~/.claude/credentials.json"),
        ]
    elif _is_windows():
        paths += [
            os.path.expandvars(r"%USERPROFILE%\.claude\.credentials.json"),
            os.path.expandvars(r"%USERPROFILE%\.claude\credentials.json"),
        ]
    return paths


def _read_credentials_file() -> tuple[Optional[dict], Optional[str]]:
    """Read credentials file. Returns (data, file_path) or (None, None)."""
    for cred_path in _credentials_file_paths():
        if os.path.exists(cred_path):
            try:
                with open(cred_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("claudeAiOauth", {}).get("accessToken"):
                    return data, cred_path
            except Exception:
                continue
    return None, None


def _refresh_access_token(refresh_token: str) -> Optional[dict]:
    """Refresh an expired access token using the refresh token.

    Calls https://console.anthropic.com/api/oauth/token with grant_type=refresh_token.

    Returns:
        {"accessToken": "sk-ant-oat01-...", "refreshToken": "sk-ant-ort01-...", "expiresAt": int}
        or None on failure.
    """
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _OAUTH_CLIENT_ID,
        }).encode("utf-8")

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "claude-code/1.0",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        # Response may have different key formats — normalize
        access_token = result.get("access_token") or result.get("accessToken")
        new_refresh = result.get("refresh_token") or result.get("refreshToken") or refresh_token
        expires_in = result.get("expires_in", 3600)
        expires_at = int(time.time() * 1000) + expires_in * 1000

        if access_token:
            logger.info("[auth] Token refreshed successfully (expires_in=%ds)", expires_in)
            return {
                "accessToken": access_token,
                "refreshToken": new_refresh,
                "expiresAt": expires_at,
            }
        logger.warning("[auth] Token refresh response missing access_token: %s", list(result.keys()))
        return None

    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        logger.error("[auth] Token refresh HTTP %d: %s", e.code, body)
        return None
    except Exception as e:
        logger.error("[auth] Token refresh failed: %s", e)
        return None


def _get_fresh_token_from_credentials() -> Optional[str]:
    """Get a fresh access token from credentials file, refreshing if expired."""
    data, cred_path = _read_credentials_file()
    if not data or not cred_path:
        return None

    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    refresh_token = oauth.get("refreshToken", "")
    expires_at = oauth.get("expiresAt", 0)

    # Check if token is expired (with 5 min buffer)
    now_ms = int(time.time() * 1000)
    is_expired = expires_at > 0 and now_ms > (expires_at - 300_000)

    if not is_expired and access_token.startswith("sk-ant-oat01-"):
        return access_token

    # Token expired — try refresh
    if not refresh_token:
        logger.warning("[auth] Token expired but no refresh token available")
        return access_token if access_token.startswith("sk-ant-oat01-") else None

    logger.info("[auth] Access token expired, refreshing...")
    new_oauth = _refresh_access_token(refresh_token)
    if not new_oauth:
        return None

    # Update credentials file
    data["claudeAiOauth"].update(new_oauth)
    try:
        with open(cred_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("[auth] Credentials file updated: %s", cred_path)
    except Exception as e:
        logger.warning("[auth] Failed to write credentials file: %s", e)

    return new_oauth["accessToken"]


# ------------------------------------------------------------------
# Platform-specific credential store readers (Keychain, etc.)
# ------------------------------------------------------------------

def _get_token_from_macos_keychain() -> Optional[str]:
    """Read OAuth token from macOS Keychain."""
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password",
             "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        creds_json = result.stdout.strip()
        if not creds_json:
            return None
        data = json.loads(creds_json)
        token = data.get("claudeAiOauth", {}).get("accessToken")
        if token and token.startswith("sk-ant-oat01-"):
            return token
        return None
    except Exception:
        return None


def _get_token_from_windows_creds() -> Optional[str]:
    """Read OAuth token from Windows credential files."""
    cred_paths = []
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        cred_paths.append(os.path.join(config_dir, ".credentials.json"))
        cred_paths.append(os.path.join(config_dir, "credentials.json"))
    cred_paths += [
        os.path.expandvars(r"%USERPROFILE%\.claude\.credentials.json"),
        os.path.expandvars(r"%USERPROFILE%\.claude\credentials.json"),
        os.path.expandvars(r"%LOCALAPPDATA%\Claude\credentials.json"),
        os.path.expandvars(r"%APPDATA%\Claude\credentials.json"),
    ]
    for cred_path in cred_paths:
        if os.path.exists(cred_path):
            try:
                with open(cred_path, encoding="utf-8") as f:
                    data = json.load(f)
                token = data.get("claudeAiOauth", {}).get("accessToken")
                if token and token.startswith("sk-ant-oat01-"):
                    return token
            except Exception:
                continue
    return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_auth_token() -> Optional[str]:
    """Resolve Claude Code OAuth token.

    Priority:
    1. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_AUTH_TOKEN env var
    2. Credentials file (with auto-refresh if expired)
    3. System credential store (Keychain, Windows creds)
    """
    for var in AUTH_TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token and not token.startswith("enc:"):
            return token

    # Credentials file with auto-refresh
    file_token = _get_fresh_token_from_credentials()
    if file_token:
        return file_token

    # System credential store
    if _is_macos():
        return _get_token_from_macos_keychain()
    elif _is_windows():
        return _get_token_from_windows_creds()

    return None


def is_claude_available() -> bool:
    """Check if Claude can be invoked (API key or OAuth token resolvable)."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True
    return get_auth_token() is not None


def get_sdk_env_vars() -> dict[str, str]:
    """Collect environment variables to pass to SDK subprocess."""
    env: dict[str, str] = {}
    for var in _SDK_ENV_VARS:
        value = os.environ.get(var)
        if value:
            env[var] = value

    if _is_windows() and "CLAUDE_CODE_GIT_BASH_PATH" not in env:
        for candidate in [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]:
            if os.path.isfile(candidate):
                env["CLAUDE_CODE_GIT_BASH_PATH"] = candidate
                break
        else:
            bash = shutil.which("bash")
            if bash:
                env["CLAUDE_CODE_GIT_BASH_PATH"] = bash

    return env


def configure_sdk_authentication(config_dir: Optional[str] = None) -> None:
    """Configure SDK authentication.

    Priority:
    1. ANTHROPIC_API_KEY — direct API (recommended for server/headless)
    2. API profile mode (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)
    3. OAuth token (CLAUDE_CODE_OAUTH_TOKEN / credentials file with auto-refresh)

    Raises:
        ValueError: If no auth method is available.
    """
    # 1. API key mode (simplest, no token refresh needed)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        logger.info("[auth] Using ANTHROPIC_API_KEY (direct API mode)")
        return

    # 2. API profile mode (enterprise proxy)
    api_profile_mode = bool(os.environ.get("ANTHROPIC_BASE_URL", "").strip())
    if api_profile_mode:
        if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise ValueError(
                "API profile mode active (ANTHROPIC_BASE_URL is set) "
                "but ANTHROPIC_AUTH_TOKEN is not set"
            )
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        logger.info("[auth] Using API profile authentication")
        return

    # 3. OAuth token (auto-refreshes expired credentials file tokens)
    token = get_auth_token()
    if not token:
        raise ValueError(
            "No auth configured. Set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN.\n"
            "For API key: https://console.anthropic.com/settings/keys\n"
            "For OAuth: claude login"
        )
    if token.startswith("enc:"):
        raise ValueError(
            "Token is encrypted (enc:...). Decryption not supported.\n"
            "Re-authenticate with: claude login"
        )

    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token
    logger.info("[auth] OAuth token configured for SDK")
