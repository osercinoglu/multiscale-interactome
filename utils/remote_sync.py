"""Remote SSH/SFTP synchronization for diffusion profile computation."""

from __future__ import annotations

import getpass
import logging
import os
import stat
import time
from typing import TYPE_CHECKING

import paramiko

if TYPE_CHECKING:
    from paramiko import SFTPClient, SSHClient


class UploadError(Exception):
    """Raised when file upload fails after all retries."""

    pass


class DownloadError(Exception):
    """Raised when file download fails after all retries."""

    pass


class RemoteSync:
    """Remote file synchronization via SSH/SFTP.

    Provides methods for uploading files, checking remote file existence,
    and listing remote directories. Supports both key-based and password
    authentication with automatic fallback.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds, doubles each retry

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        remote_path: str,
        key_path: str | None = None,
        password: str | None = None,
        logger: logging.Logger | None = None,
        timeout: int = 30,
    ):
        """Initialize RemoteSync.

        Args:
            host: Remote SSH hostname
            port: Remote SSH port
            username: SSH username
            remote_path: Base path on remote server for uploads
            key_path: Path to SSH private key (optional)
            password: SSH password (optional, will prompt if needed)
            logger: Logger instance (optional)
            timeout: SSH/SFTP operation timeout in seconds (default: 30).
                     Set to 0 for no timeout. Applies to connection and
                     all SFTP channel operations.
        """
        self.host = host
        self.port = port
        self.username = username
        self.remote_path = remote_path
        self.key_path = key_path
        self.password = password
        self.logger = logger or logging.getLogger("diffusion_profiles")
        self.timeout = timeout if timeout > 0 else None
        self._client: SSHClient | None = None
        self._sftp: SFTPClient | None = None

    def connect(self) -> None:
        """Establish SSH connection.

        Tries key-based authentication first, falls back to password.
        Prompts for password if not provided and key auth fails.
        """
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Try key-based authentication first
        try:
            self.logger.debug(f"Attempting key-based auth to {self.host}:{self.port}")
            connect_kwargs = {
                "hostname": self.host,
                "port": self.port,
                "username": self.username,
                "timeout": self.timeout,
            }
            if self.key_path:
                connect_kwargs["key_filename"] = self.key_path
                self._client.connect(**connect_kwargs)
            else:
                # Try default SSH agent / keys
                self._client.connect(**connect_kwargs, look_for_keys=True)
            self._sftp = self._client.open_sftp()
            if self.timeout is not None:
                self._sftp.get_channel().settimeout(self.timeout)
            self.logger.info(f"Connected to {self.host}:{self.port} (key auth)")
            return
        except paramiko.AuthenticationException:
            self.logger.debug("Key-based auth failed, trying password")
        except Exception as e:
            self.logger.debug(f"Key-based auth failed: {e}, trying password")

        # Fall back to password authentication
        if self.password is None:
            self.password = getpass.getpass(f"Password for {self.username}@{self.host}: ")

        try:
            self._client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=self.timeout,
            )
            self._sftp = self._client.open_sftp()
            if self.timeout is not None:
                self._sftp.get_channel().settimeout(self.timeout)
            self.logger.info(f"Connected to {self.host}:{self.port} (password auth)")
        except Exception as e:
            self.logger.error(f"SSH connection failed: {e}")
            raise

    def close(self) -> None:
        """Close SSH/SFTP connections."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self.logger.debug("SSH connection closed")

    def _reconnect(self) -> None:
        """Close and re-establish SSH/SFTP after a channel error."""
        self.logger.debug("Reconnecting SSH/SFTP...")
        self.close()
        self.connect()

    def __enter__(self) -> "RemoteSync":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def remote_file_exists(self, remote_file: str) -> bool:
        """Check if file exists on remote server.

        Args:
            remote_file: Full remote path to check

        Returns:
            True if file exists, False otherwise
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            self._sftp.stat(remote_file)
            return True
        except FileNotFoundError:
            return False
        except IOError:
            return False

    def _mkdir_p(self, remote_dir: str) -> None:
        """Recursively create remote directories.

        Args:
            remote_dir: Remote directory path to create
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")

        if not remote_dir or remote_dir == "/":
            return

        # Normalize path
        remote_dir = remote_dir.rstrip("/")

        try:
            self._sftp.stat(remote_dir)
            return  # Directory exists
        except FileNotFoundError:
            pass

        # Recursively create parent
        parent = os.path.dirname(remote_dir)
        if parent and parent != remote_dir:
            self._mkdir_p(parent)

        # Create this directory
        try:
            self._sftp.mkdir(remote_dir)
            self.logger.debug(f"Created remote directory: {remote_dir}")
        except IOError as e:
            # Directory might have been created by another process
            try:
                info = self._sftp.stat(remote_dir)
                if not stat.S_ISDIR(info.st_mode):
                    raise
            except FileNotFoundError:
                raise e

    def upload_file(
        self,
        local_path: str,
        remote_subpath: str,
    ) -> bool:
        """Upload a single file to remote server with retry logic.

        Args:
            local_path: Local file path to upload
            remote_subpath: Relative path under remote_path for the file

        Returns:
            True on success

        Raises:
            UploadError: After MAX_RETRIES failures
        """
        remote_full = os.path.join(self.remote_path, remote_subpath)
        remote_dir = os.path.dirname(remote_full)

        delay = self.RETRY_DELAY
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if not self._sftp:
                    self._reconnect()
                # Create remote directories if needed
                self._mkdir_p(remote_dir)
                self._sftp.put(local_path, remote_full)
                self.logger.debug(f"Uploaded: {local_path} -> {remote_full}")
                return True
            except Exception as e:
                last_error = e
                if attempt == self.MAX_RETRIES:
                    self.logger.error(
                        f"Failed to upload {local_path} after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise UploadError(
                        f"Failed to upload {local_path} after {self.MAX_RETRIES} attempts: {e}"
                    )
                self.logger.warning(
                    f"Upload attempt {attempt} failed for {local_path}, "
                    f"retrying in {delay}s... ({e})"
                )
                try:
                    self._reconnect()
                except Exception as reconn_err:
                    self.logger.warning(f"Reconnect failed: {reconn_err}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        # Should not reach here, but just in case
        raise UploadError(f"Failed to upload {local_path}: {last_error}")

    def list_remote_profiles(self, run_dir: str) -> list[str]:
        """List profile files in remote run directory.

        Args:
            run_dir: Relative path under remote_path to list

        Returns:
            List of profile filenames (ending in '_p_visit_array.npy')
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")

        remote_path = os.path.join(self.remote_path, run_dir)
        try:
            files = self._sftp.listdir(remote_path)
            return [f for f in files if f.endswith("_p_visit_array.npy")]
        except FileNotFoundError:
            return []
        except IOError:
            return []

    def count_remote_profiles(self, run_dir: str) -> int:
        """Count profile files in remote run directory.

        Args:
            run_dir: Relative path under remote_path

        Returns:
            Number of profile files found
        """
        return len(self.list_remote_profiles(run_dir))

    def remote_run_exists(self, run_dir: str) -> bool:
        """Check if a run directory exists on remote.

        Args:
            run_dir: Relative path under remote_path

        Returns:
            True if directory exists
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")

        remote_path = os.path.join(self.remote_path, run_dir)
        try:
            info = self._sftp.stat(remote_path)
            return stat.S_ISDIR(info.st_mode)
        except FileNotFoundError:
            return False
        except IOError:
            return False


    def download_file(
        self,
        remote_subpath: str,
        local_path: str,
    ) -> bool:
        """Download a single file from remote server with retry logic.

        Args:
            remote_subpath: Relative path under remote_path on remote
            local_path: Local file path to save to

        Returns:
            True on success

        Raises:
            DownloadError: After MAX_RETRIES failures
        """
        remote_full = os.path.join(self.remote_path, remote_subpath)
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        delay = self.RETRY_DELAY
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if not self._sftp:
                    self._reconnect()
                self._sftp.get(remote_full, local_path)
                # Verify download completeness
                remote_size = self._sftp.stat(remote_full).st_size
                local_size = os.path.getsize(local_path)
                if local_size != remote_size:
                    os.remove(local_path)
                    raise IOError(
                        f"Size mismatch: local {local_size} != remote {remote_size}"
                    )
                self.logger.debug(f"Downloaded: {remote_full} -> {local_path}")
                return True
            except Exception as e:
                # Remove partial/corrupt file left by interrupted transfer
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass
                if attempt == self.MAX_RETRIES:
                    self.logger.error(
                        f"Failed to download {remote_full} after {self.MAX_RETRIES} attempts: {e}"
                    )
                    raise DownloadError(
                        f"Failed to download {remote_full} after {self.MAX_RETRIES} attempts: {e}"
                    )
                self.logger.warning(
                    f"Download attempt {attempt} failed for {remote_full}, "
                    f"retrying in {delay}s... ({e})"
                )
                try:
                    self._reconnect()
                except Exception as reconn_err:
                    self.logger.warning(f"Reconnect failed: {reconn_err}")
                time.sleep(delay)
                delay *= 2

        raise DownloadError(f"Failed to download {remote_full}")

    def _is_run_dir(self, dir_path: str) -> bool:
        """Check if a remote directory looks like a diffusion-profile run.

        A directory is a run if it contains ``node2idx.pkl`` (fast stat check)
        **or** any ``.npy`` file (slower listdir fallback).
        """
        # Fast path: metadata present
        try:
            self._sftp.stat(os.path.join(dir_path, "node2idx.pkl"))
            return True
        except (FileNotFoundError, IOError):
            pass
        # Slow path: check for .npy profile files
        try:
            return any(f.endswith(".npy") for f in self._sftp.listdir(dir_path))
        except (FileNotFoundError, IOError):
            return False

    def list_remote_dir(self, subpath: str = "") -> list[str]:
        """List filenames in a remote directory.

        Args:
            subpath: Relative path under remote_path to list.

        Returns:
            List of filenames (not full paths).
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")
        full = os.path.join(self.remote_path, subpath) if subpath else self.remote_path
        try:
            return self._sftp.listdir(full)
        except (FileNotFoundError, IOError):
            return []

    def list_remote_subdirs(
        self, subpath: str = "", recursive: bool = False,
    ) -> list[str]:
        """List subdirectories that look like diffusion-profile runs.

        A directory is considered a run if it contains ``node2idx.pkl`` or
        any ``.npy`` file.

        Args:
            subpath: Relative path under remote_path to scan
            recursive: If True, directories that are not runs are treated
                       as category folders and scanned one level deeper.
                       Results include the category prefix (e.g. "docking/run_a").

        Returns:
            Sorted list of run identifiers (relative to remote_path/subpath)
        """
        if not self._sftp:
            raise RuntimeError("Not connected. Call connect() first.")

        base = os.path.join(self.remote_path, subpath) if subpath else self.remote_path
        try:
            entries = self._sftp.listdir(base)
        except (FileNotFoundError, IOError):
            return []

        self.logger.debug(f"Scanning {len(entries)} entries in {base}")
        valid_runs = []
        for i, entry in enumerate(entries, 1):
            entry_path = os.path.join(base, entry)
            try:
                info = self._sftp.stat(entry_path)
                if not stat.S_ISDIR(info.st_mode):
                    continue
            except (FileNotFoundError, IOError):
                continue

            if self._is_run_dir(entry_path):
                self.logger.debug(f"  [{i}/{len(entries)}] {entry} -> run")
                valid_runs.append(entry)
            elif recursive:
                # Not a run — treat as a category folder
                self.logger.info(f"  Scanning category folder: {entry}")
                try:
                    sub_entries = self._sftp.listdir(entry_path)
                except (FileNotFoundError, IOError):
                    continue
                for j, sub_entry in enumerate(sub_entries, 1):
                    sub_path = os.path.join(entry_path, sub_entry)
                    try:
                        sub_info = self._sftp.stat(sub_path)
                        if not stat.S_ISDIR(sub_info.st_mode):
                            continue
                    except (FileNotFoundError, IOError):
                        continue
                    if self._is_run_dir(sub_path):
                        self.logger.debug(
                            f"    [{j}/{len(sub_entries)}] {entry}/{sub_entry} -> run"
                        )
                        valid_runs.append(f"{entry}/{sub_entry}")

        return sorted(valid_runs)


def make_upload_callback(
    remote_sync: RemoteSync,
    run_dir: str,
    local_dir: str,
    delete_after_upload: bool = False,
    logger: logging.Logger | None = None,
):
    """Create a callback function for uploading profiles as they complete.

    Args:
        remote_sync: Connected RemoteSync instance
        run_dir: Relative path for this run on remote
        local_dir: Local directory where profiles are saved
        delete_after_upload: If True, delete local file after successful upload
        logger: Logger instance

    Returns:
        Callback function that takes a node_id and uploads the corresponding profile
    """
    log = logger or logging.getLogger("diffusion_profiles")

    def _clean_file_name(file_name: str) -> str:
        """Clean filename for saving (matches diffusion_profiles.py logic)."""
        return "".join(
            c for c in file_name if c.isalpha() or c.isdigit() or c == " " or c == "_"
        ).rstrip()

    def callback(node_id: str) -> None:
        """Upload a completed profile file.

        Args:
            node_id: The node ID whose profile was just computed

        Raises:
            UploadError: If upload fails after retries (aborts computation)
        """
        clean_name = _clean_file_name(node_id)
        local_file = os.path.join(local_dir, f"{clean_name}_p_visit_array.npy")
        remote_subpath = os.path.join(run_dir, f"{clean_name}_p_visit_array.npy")

        # Upload file (raises UploadError after retries, aborting computation)
        remote_sync.upload_file(local_file, remote_subpath)

        # Delete local file after successful upload if requested
        if delete_after_upload and os.path.exists(local_file):
            try:
                os.remove(local_file)
                log.debug(f"Deleted local file after upload: {local_file}")
            except OSError as e:
                log.warning(f"Failed to delete local file {local_file}: {e}")

    return callback
