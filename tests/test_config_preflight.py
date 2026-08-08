"""Preflight key validation.

A new reviewer who copies .env.example and forgets to swap the placeholders should
get an actionable message, not a bare SDK 401 traceback.
"""
import pytest

from app.config import MissingKeysError, Settings, require_live_keys


def test_placeholder_keys_are_rejected():
    settings = Settings(
        together_api_key="dummy-together-key",
        pinecone_api_key="dummy-pinecone-key",
    )
    with pytest.raises(MissingKeysError) as exc:
        require_live_keys(settings)

    msg = str(exc.value)
    assert "TOGETHER_API_KEY" in msg
    assert "PINECONE_API_KEY" in msg
    assert "placeholder" in msg
    assert "--dry-run" in msg, "should point at the no-keys-needed path"


def test_empty_keys_are_rejected():
    with pytest.raises(MissingKeysError) as exc:
        require_live_keys(Settings(together_api_key="", pinecone_api_key=""))
    assert "is not set" in str(exc.value)


def test_only_one_missing_key_is_named():
    with pytest.raises(MissingKeysError) as exc:
        require_live_keys(Settings(together_api_key="real-looking", pinecone_api_key=""))
    msg = str(exc.value)
    assert "PINECONE_API_KEY" in msg
    assert "TOGETHER_API_KEY" not in msg


def test_real_looking_keys_pass():
    require_live_keys(
        Settings(together_api_key="tgp_v1_xxx", pinecone_api_key="pcsk_xxx")
    )
