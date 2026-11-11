"""Software-only reservation and immutable exposure-ledger checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.fresh_exposure_guard import (LEASE_NAME, ledger_transaction,
                                         verify_prior_commitments)


class FreshExposureReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'private-ledger'

    def event(self, name, content):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def test_nested_competing_reservation_fails_without_releasing_owner(self):
        with ledger_transaction(self.root) as held:
            self.assertEqual(held, self.root)
            self.assertTrue((self.root / LEASE_NAME).is_dir())
            with self.assertRaisesRegex(ValueError, 'busy or stale'):
                with ledger_transaction(self.root):
                    self.fail('A second transaction entered while the first held the lease')
            self.assertTrue((self.root / LEASE_NAME).is_dir())
        self.assertFalse((self.root / LEASE_NAME).exists())
        # Normal release permits a subsequent, serial reservation.
        with ledger_transaction(self.root):
            self.assertTrue((self.root / LEASE_NAME).is_dir())

    def test_exception_releases_owned_empty_lease_and_preserves_event(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic verification failure'):
            with ledger_transaction(self.root):
                self.event('retained.json', b'{"attempt":"software-fixture"}')
                raise RuntimeError('synthetic verification failure')
        self.assertFalse((self.root / LEASE_NAME).exists())
        self.assertTrue((self.root / 'retained.json').is_file())

    def test_stale_lease_is_never_automatically_removed(self):
        stale = self.root / LEASE_NAME
        stale.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'busy or stale'):
            with ledger_transaction(self.root):
                self.fail('A stale lease must require investigation')
        self.assertTrue(stale.is_dir())

    def test_cleanup_refuses_nonempty_or_replaced_lease(self):
        lease = self.root / LEASE_NAME
        with self.assertRaisesRegex(ValueError, 'safely released'):
            with ledger_transaction(self.root):
                (lease / 'unexpected').write_text('preserve')
        self.assertEqual((lease / 'unexpected').read_text(), 'preserve')
        (lease / 'unexpected').unlink(); lease.rmdir()
        # Keep the original inode alive so an immediate mkdir cannot reuse it.
        moved = self.root / 'moved-owner'
        with self.assertRaisesRegex(ValueError, 'ownership changed'):
            with ledger_transaction(self.root):
                lease.rename(moved); lease.mkdir()
        self.assertTrue(lease.is_dir())
        self.assertTrue(moved.is_dir())

    def test_prior_commitments_reject_changed_or_deleted_events_allow_additions(self):
        prior = self.event('previous.json', b'{"attempt":1}')
        self.event('later.json', b'{"attempt":2}')
        verify_prior_commitments(self.root, {'previous.json': prior})
        (self.root / 'previous.json').write_bytes(b'{"attempt":9}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_prior_commitments(self.root, {'previous.json': prior})
        (self.root / 'previous.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            verify_prior_commitments(self.root, {'previous.json': prior})

    def test_commitment_validation_is_read_only_and_rejects_paths_or_symlinks(self):
        verify_prior_commitments(self.root, {})
        self.assertFalse(self.root.exists())
        digest = self.event('original.json', b'{"attempt":1}')
        (self.root / 'alias.json').symlink_to(self.root / 'original.json')
        for name in ['../original.json', '/original.json', 'sub/original.json', 'sub\\original.json', 'alias.json']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_prior_commitments(self.root, {name: digest})
        for invalid in [None, [], {'original.json': 'not-a-sha256'}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                verify_prior_commitments(self.root, invalid)


if __name__ == '__main__':
    unittest.main()
