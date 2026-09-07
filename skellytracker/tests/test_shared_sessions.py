import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import Mock, patch

from skellytracker.core.config.session_config import SessionConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.sessions.session import Session
from skellytracker.core.sessions.shared_sessions import SessionRequest, SharedSessionPool
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.detection_stage import DetectionStage


class TestSessionConfig(SessionConfig):
    backend: str = "test"
    batch_size: int = 1


@dataclass
class TestSession(Session):
    kind: ClassVar[str] = "test"
    closes: int = 0

    @classmethod
    def create(cls, config: SessionConfig) -> "TestSession":
        return cls()

    def close(self) -> None:
        self.closes += 1


class SharedSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = SharedSessionPool()
        self.addCleanup(self.pool.close)
        self.request = SessionRequest(session_type=TestSession, config=TestSessionConfig())

    def test_reuse_does_not_share_trackers_or_close_another_clients_session(self) -> None:
        first = self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        second = self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        session = first._entries[0].session
        self.assertIs(session, second._entries[0].session)
        self.assertIsNot(first.tracker, second.tracker)
        first.close()
        self.assertEqual(session.closes, 0)
        second.close()
        second.close()
        self.assertEqual(session.closes, 1)

    def test_batch_configuration_separates_sessions_and_is_snapshotted(self) -> None:
        first = self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        self.request.config.batch_size = 2
        second = self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        self.assertIsNot(first._entries[0].session, second._entries[0].session)
        self.assertEqual(first._entries[0].request.config.batch_size, 1)

    def test_detector_creation_failure_releases_only_its_references(self) -> None:
        first = self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        session = first._entries[0].session
        with patch.object(Tracker, "create_with_shared_sessions", side_effect=RuntimeError("detector failed")):
            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))
        self.assertEqual(session.closes, 0)
        first.close()
        self.assertEqual(session.closes, 1)

    def test_cross_thread_access_fails_before_allocating(self) -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.pool.create_tracker, config=TrackerConfig(stages=[]), requests=(self.request,))
            with self.assertRaisesRegex(RuntimeError, "owning inference thread"):
                future.result()

    def test_closed_pool_rejects_new_work(self) -> None:
        self.pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.pool.create_tracker(config=TrackerConfig(stages=[]), requests=(self.request,))

    def test_detector_instances_are_independent_and_close_before_sessions(self) -> None:
        stages = [Mock(spec=DetectionStage), Mock(spec=DetectionStage)]
        config = Mock(spec=TrackerConfig)
        config.stages = [Mock()]
        with patch("skellytracker.core.tracker.tracker.DetectionStage.create", side_effect=stages):
            first = self.pool.create_tracker(config=config, requests=(self.request,))
            second = self.pool.create_tracker(config=config, requests=(self.request,))
        first.close()
        stages[0].close.assert_called_once()
        stages[1].close.assert_not_called()
        self.assertEqual(second._entries[0].session.closes, 0)


if __name__ == "__main__":
    unittest.main()
