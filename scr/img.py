"""图片资源路径集中管理。

说明：集中管理项目中使用到的图片路径，避免散落的硬编码字符串，方便统一修改与查找。
分组：通用 IMG（跨模式通用），模式专用（如 CWIMG）。
"""


class IMG:
	BASE = "img"

	MENU = f"{BASE}/menu.png"
	SKIP1 = f"{BASE}/skip1.png"
	SKIP2 = f"{BASE}/skip2.png"
	READ = f"{BASE}/read.png"
	NO_VOICE = f"{BASE}/no_voice.png"
	LOCK = f"{BASE}/lock.png"
	ENTER = f"{BASE}/enter.png"

