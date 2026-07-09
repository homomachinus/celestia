"""HuggingFace Hub push utilities.

Handles authentication and uploading a merged model + tokenizer.
"""

from __future__ import annotations

import logging
import os
from getpass import getpass

from huggingface_hub import HfApi, login

log = logging.getLogger(__name__)


def hf_login(token: str | None = None) -> None:
    """Authenticate with HuggingFace Hub.

    If *token* is ``None``, prompts interactively.
    Token must have **write** access to push models.
    """
    if token:
        login(token=token, add_to_git_credential=True)
    else:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=True)
        else:
            log.info("No HF_TOKEN in env; prompting for token ...")
            token = getpass("HuggingFace Hub token (write access): ")
            login(token=token, add_to_git_credential=True)
    log.info("HuggingFace login successful.")


def push_to_hub(
    model_dir: str,
    repo_id: str,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload a merged model + tokenizer to HuggingFace Hub.

    Parameters
    ----------
    model_dir:
        Local directory with ``config.json``, ``model.safetensors``,
        ``tokenizer.json``, etc.
    repo_id:
        e.g. ``"nuxt/celestia-plato-3b"``
    private:
        Whether the repo should be private.
    token:
        HF token (if not already logged in).

    Returns
    -------
    The full Hub URL.
    """
    if token:
        hf_login(token)

    api = HfApi()
    url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
    )
    log.info("Pushing model from %s to %s ...", model_dir, url)

    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message="Upload merged model + tokenizer",
    )

    hub_url = f"https://huggingface.co/{repo_id}"
    log.info("Push complete: %s", hub_url)
    return hub_url
