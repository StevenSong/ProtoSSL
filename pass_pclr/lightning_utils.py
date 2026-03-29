import errno
import os
import re
import shutil
import tempfile
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from wandb.util import generate_id

from .defines import STAGE_T


class StrictWandbLogger(WandbLogger):
    def __init__(
        self,
        *,
        project: str,
        name: str,
        save_dir: str,
        pipeline_stage: STAGE_T,
        resume_from_checkpoint: str | None = None,
    ):
        run_dir = os.path.join(save_dir, name, pipeline_stage)
        if resume_from_checkpoint is not None:
            if not resume_from_checkpoint.startswith(run_dir):
                raise ValueError(
                    f"Logger resume_from_checkpoint={resume_from_checkpoint}, but does not start with specified run_dir={run_dir}"
                )
            version = (
                resume_from_checkpoint.replace(run_dir, "")
                .strip(os.path.sep)
                .split(os.path.sep)[0]
            )
        else:
            version = next_version(run_dir)
        save_dir = os.path.join(run_dir, version)
        self.best_link = os.path.join(save_dir, "best.ckpt")
        self.best_link_warned_once = False  # only used to prevent cluttering stdout for non-symlink checkpointing

        super().__init__(
            project=project,
            name=f"{name}-{pipeline_stage}",
            version=version,
            save_dir=save_dir,
        )
        if resume_from_checkpoint is None:
            if os.path.exists(self.save_dir):  # type: ignore
                raise FileExistsError(
                    "\033[91mREAD THIS ERROR MSG: \033[0m"
                    f"Experiment already exists at {self.save_dir}."
                    " This logger uses some custom logic to put all logs,"
                    " checkpoints, and configs related to an experiment"
                    " under one directory. Please delete or rename to retry."
                )
            makedirs_wrapper(self.save_dir)

    def after_save_checkpoint(self, checkpoint_callback):
        best_model_path = checkpoint_callback.best_model_path
        link_dir = os.path.dirname(self.best_link)
        if supports_symlinks(link_dir):
            best_model_path = best_model_path.replace(link_dir, "")
            best_model_path = best_model_path.lstrip(os.path.sep)
            if os.path.exists(self.best_link):
                os.remove(self.best_link)
            os.symlink(
                best_model_path,  # rel path e.g. bar/model.ckpt instead of /foo/bar/model.ckpt
                self.best_link,  # full path e.g. /foo/best.ckpt -> bar/model.ckpt
            )
        else:
            if not self.best_link_warned_once:
                print("======StrictWandbLogger.after_save_checkpoint======")
                print(
                    f"WARNING: filesystem of {link_dir} does not support making symlinks, copying checkpoint of best model to {self.best_link}"
                )
                print("===================================================")
                self.best_link_warned_once = True
            if os.path.exists(self.best_link):
                os.remove(self.best_link)
            # use best link here as file path, not link e.g. /foo/best.ckpt
            # copyfile is safest method e.g. for Fat32 drives
            shutil.copyfile(best_model_path, self.best_link)


@rank_zero_only
def makedirs_wrapper(save_dir):
    os.makedirs(save_dir)
    link_dir = os.path.dirname(save_dir)  # /foo/bar (if save_dir is e.g. /foo/bar/v1)
    latest_link = os.path.join(link_dir, "latest")  # /foo/bar/latest
    if supports_symlinks(link_dir):
        if os.path.exists(latest_link):
            os.remove(latest_link)
        rel_dir = save_dir
        rel_dir = rel_dir.replace(os.path.dirname(rel_dir), "")
        rel_dir = rel_dir.lstrip(os.path.sep)
        os.symlink(rel_dir, latest_link)
    else:
        print("=================makedirs_wrapper==================")
        print(
            f"WARNING: filesystem of {link_dir} does not support making symlinks, will copy run to `latest` directory after run finishes"
        )
        print("===================================================")


def next_version(path, prefix="v"):
    path = Path(path)
    if not path.exists():
        return f"{prefix}1-{generate_id()}"

    versions = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)-(.+)$")

    for p in path.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                versions.append(int(m.group(1)))

    # the "UUID" suffix is needed in case runs are deleted and versions are reused
    # as wandb does not allow name reusage even after "deletion" (not really deleted?)
    return f"{prefix}{max(versions) + 1 if versions else 1}-{generate_id()}"


def supports_symlinks(path="."):
    with tempfile.TemporaryDirectory(dir=path) as tmpdir:
        src = os.path.join(tmpdir, "src")
        link = os.path.join(tmpdir, "link")
        open(src, "w").close()  # create a dummy file
        try:
            os.symlink(src, link)
            return True
        except OSError as e:
            if e.errno in (errno.EPERM, errno.EIO, errno.ENOTSUP):
                return False
            raise  # re-raise unexpected errors


def copytree_no_meta(src, dst):
    # shutil allows copying using copyfile, however still tries to set metadata
    # on copied directories, which is not supported for all drive formats e.g. FAT32
    # so we make our own copytree using an explicit stack to avoid stack overflow
    # on deeply nested directory trees
    os.makedirs(dst, exist_ok=True)
    stack = [(src, dst)]
    while stack:
        current_src, current_dst = stack.pop()
        for item in os.scandir(current_src):
            s = item.path
            d = os.path.join(current_dst, item.name)
            if item.is_dir():
                os.makedirs(d, exist_ok=True)
                stack.append((s, d))
            else:
                shutil.copyfile(s, d)


def check_final_link(save_dir):
    link_dir = os.path.dirname(save_dir)  # /foo/bar (if save_dir is e.g. /foo/bar/v1)
    latest_link = os.path.join(link_dir, "latest")  # /foo/bar/latest
    # do not depend on checking existence of latest link as it may be stale from prior run
    # if this function is called, we are intent on copying over the current run to latest
    # (makedirs_wrapper above made this link already if it could have, should be mutually exclusive with below)
    if not supports_symlinks(link_dir):
        # if we couldn't create symlink, copy contents over to `latest` dir
        print("=================check_final_link==================")
        print(
            f"WARNING: filesystem of {link_dir} does not support making symlinks, copying run to {latest_link}"
        )
        print("===================================================")
        if os.path.exists(latest_link):
            shutil.rmtree(latest_link)
        copytree_no_meta(save_dir, latest_link)
