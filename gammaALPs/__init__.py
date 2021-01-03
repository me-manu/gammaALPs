from __future__ import absolute_import, division, print_function

__all__ = ['bfields', 'nel', 'base', 'utils', 'core']
__version__ = "unknown"

try:
    from .version import get_git_version
    __version__ = get_git_version()
except Exception as message:
    print(message)

__author__ = "Manuel Meyer // me-manu.github.io"
