"""`$ROBOJEV_HOME`, and the one path a box's own secret takes into a policy.

The thing being tested in the second half is a *secret*, so the assertions are about where a
value may and may not appear. The value here is an obvious fake, and no test prints one.
"""
from __future__ import annotations

import pathlib

import pytest

from robojev import home as home_mod
from robojev import jev_api

#: Not shaped like anybody's real key, on purpose.
FAKE_KEY = "test-key-not-real"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv(home_mod.LEGACY_HOME_ENV, raising=False)
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


# --------------------------------------------------------------------------------- the root

def test_the_default_root_is_under_the_users_home(monkeypatch):
    monkeypatch.delenv(home_mod.HOME_ENV, raising=False)
    monkeypatch.delenv(home_mod.LEGACY_HOME_ENV, raising=False)
    assert home_mod.home() == pathlib.Path("~/.robojev").expanduser()


def test_the_variable_wins_and_is_read_per_call(tmp_path, monkeypatch):
    monkeypatch.delenv(home_mod.LEGACY_HOME_ENV, raising=False)
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "one"))
    assert home_mod.home() == tmp_path / "one"
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "two"))
    assert home_mod.home() == tmp_path / "two"


def test_the_older_variable_is_a_fallback_and_only_that(tmp_path, monkeypatch):
    """An operator whose harvests are already under the old root keeps them -- but the new name
    wins wherever both are set, or an upgrade would silently keep writing to the old place."""
    monkeypatch.delenv(home_mod.HOME_ENV, raising=False)
    monkeypatch.setenv(home_mod.LEGACY_HOME_ENV, str(tmp_path / "legacy"))
    assert home_mod.home() == tmp_path / "legacy"
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "current"))
    assert home_mod.home() == tmp_path / "current"


def test_reading_the_root_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "nope"))
    assert home_mod.home() == tmp_path / "nope"
    assert not (tmp_path / "nope").exists()


def test_the_four_subdirectories_are_under_it(home):
    assert home_mod.data_dir("robojev", "libero_spatial") == home / "data/robojev/libero_spatial"
    assert home_mod.checkpoints_dir() == home / "checkpoints"
    assert home_mod.src_dir() == home / "src"
    assert home_mod.env_dir("robojev@abc") == home / "envs/robojev@abc"


# ---------------------------------------------------------------------------------- the key

def write_env(text: str) -> pathlib.Path:
    path = pathlib.Path(jev_api.env_file())
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_the_key_file_lives_beside_the_runtime_root(home):
    assert jev_api.env_file() == str(home / "env")


def test_the_key_file_is_read_under_the_new_variable_only(tmp_path, monkeypatch):
    """The secret is looked for under one name. `home()` honours an older variable for the sake
    of harvests somebody already has; a key is not something to go hunting for under a second
    name, and the older root may belong to another tool entirely."""
    monkeypatch.delenv(home_mod.HOME_ENV, raising=False)
    monkeypatch.setenv(home_mod.LEGACY_HOME_ENV, str(tmp_path / "legacy"))
    assert str(tmp_path / "legacy") not in jev_api.env_file()


def test_no_file_and_no_variable_is_a_failure_that_names_both(home):
    with pytest.raises(jev_api.JevApiError) as exc:
        jev_api.api_key({})
    message = str(exc.value)
    assert jev_api.API_KEY_ENV in message and str(home / "env") in message


def test_the_environment_wins_over_the_file(home):
    write_env(f"{jev_api.API_KEY_ENV}=from-the-file-not-real\n")
    assert jev_api.api_key({jev_api.API_KEY_ENV: FAKE_KEY}) == FAKE_KEY


def test_the_file_is_name_equals_value_one_per_line(home):
    write_env(f"# a comment\nOTHER=2\n{jev_api.API_KEY_ENV}={FAKE_KEY}\n")
    assert jev_api.api_key({}) == FAKE_KEY


def test_quotes_around_the_value_are_not_part_of_it(home):
    write_env(f'{jev_api.API_KEY_ENV}="{FAKE_KEY}"\n')
    assert jev_api.api_key({}) == FAKE_KEY


def test_an_unreadable_file_is_the_same_as_no_key_never_an_oserror(home):
    (home / "env").mkdir()              # a directory where a file should be
    with pytest.raises(jev_api.JevApiError):
        jev_api.api_key({})
