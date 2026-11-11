"""Fail-closed serialization and prior-event integrity for private exposures.

The caller must re-verify metadata and numerical overlap, then reserve the full
cohort, inside ``ledger_transaction`` and before entering any predictor. A lease
is never recovered by age: an abandoned lease requires explicit investigation.
These guards cannot establish physical independence or authenticate provenance.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import stat


LEASE_NAME = '.reservation.lock'


@contextmanager
def ledger_transaction(root):
    """Hold one atomic directory lease; busy or stale leases fail closed.

    Yield the canonical ledger directory. The lease contains no files and is
    removed only when its device/inode still match the directory we created.
    No predictor should run after an acquisition or integrity error.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lease = root / LEASE_NAME
    try:
        lease.mkdir()
    except FileExistsError as error:
        raise ValueError('Performance exposure reservation is busy or stale; no automatic lease recovery') from error
    owned = lease.lstat()
    try:
        yield root
    finally:
        try:
            current = lease.lstat()
            if (not stat.S_ISDIR(current.st_mode) or
                    (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino)):
                raise ValueError('Performance exposure lease ownership changed; refusing cleanup')
            # rmdir refuses a nonempty lease and never recursively removes files.
            lease.rmdir()
        except OSError as error:
            raise ValueError('Performance exposure lease could not be safely released') from error


def verify_prior_commitments(root, commitments):
    """Require every previously locked event's exact bytes; allow added events.

    Keys are JSON event basenames, not paths. Empty commitments allow an absent
    ledger at the first attempt. Verification itself does not create any files.
    """
    root = Path(root).resolve()
    if type(commitments) is not dict:
        raise ValueError('Prior performance exposure commitments must be a mapping')
    for name, expected in commitments.items():
        if (type(name) is not str or not name or '/' in name or '\\' in name or
                ':' in name or Path(name).name != name or not name.endswith('.json')):
            raise ValueError('Prior performance exposure key must be a JSON event basename')
        if (type(expected) is not str or len(expected) != 64 or
                any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('Prior performance exposure SHA256 is invalid')
        path = root / name
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                raise ValueError('Previously committed performance exposure is missing or not a regular ledger event')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError('Previously committed performance exposure cannot be read') from error
        if actual != expected:
            raise ValueError('Previously committed performance exposure changed')
