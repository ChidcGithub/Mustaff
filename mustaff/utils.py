"""
通用小工具
"""

import os
from typing import Optional, Tuple


def split_artist_title(filepath: str) -> Tuple[str, Optional[str]]:
    """从文件名解析 (标题, 艺术家)

    支持 "Artist - Title.mp3" 这类常见命名；
    无法解析时返回 (文件名去扩展名, None)。
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in base:
        artist, title = base.split(" - ", 1)
        artist, title = artist.strip(), title.strip()
        if artist and title:
            return title, artist
    return base, None
