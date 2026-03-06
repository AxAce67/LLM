from data_preprocessor.quality_filter import contains_pii, detect_language


def test_contains_pii_does_not_flag_date():
    text = "リリース日は 2026-03-05 です。"
    assert contains_pii(text) is False


def test_contains_pii_does_not_flag_version_number():
    text = "current version is 1.2.3.4 and build is stable."
    assert contains_pii(text) is False


def test_contains_pii_flags_email():
    text = "contact me: test.user@example.com"
    assert contains_pii(text) is True


def test_contains_pii_flags_ip_with_context():
    text = "server ip address is 192.168.1.10"
    assert contains_pii(text) is True


def test_detect_language_simple():
    assert detect_language("これは日本語の文章です。") == "ja"
    assert detect_language("This is an English sentence.") == "en"
