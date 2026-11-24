"""Configuración común de pytest: permite importar los módulos de code/ por su nombre."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
