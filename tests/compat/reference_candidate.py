"""Reference-backed candidate used only to validate the harness itself."""

from torch import *  # noqa: F403
import torch as _torch


def __getattr__(name):
    return getattr(_torch, name)
