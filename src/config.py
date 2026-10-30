"""설정 파일(config/settings.yaml) 읽기 도우미."""

from pathlib import Path

import yaml

# 프로젝트 최상위 폴더 (src/의 한 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """YAML 설정 파일을 읽어 딕셔너리로 돌려준다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
