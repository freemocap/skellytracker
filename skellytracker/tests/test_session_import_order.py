"""Public session imports must work before detector registry initialization."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', [
    'skellytracker.core.sessions.mediapipe_session',
    'skellytracker.core.sessions.shared_sessions',
    'skellytracker.core.tracker.tracker_factory',
    'skellytracker.core.tracker',
])
def test_public_session_and_factory_imports_are_independent(module):
    result = subprocess.run([
        sys.executable, '-B', '-c',
        f'import importlib, sys; importlib.import_module({module!r}); '
        'from skellytracker.core.sessions.mediapipe_session import MediaPipeSession; '
        'from skellytracker.core.sessions.shared_sessions import SharedSessionPool; '
        'from skellytracker.core.tracker import build_tracker; '
        'assert "mediapipe" not in sys.modules; assert "onnxruntime" not in sys.modules',
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
