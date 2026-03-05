import data_collector.web_crawler as web_crawler


class _DummyDB:
    def is_url_crawled(self, _url):
        return False

    def insert_crawled_data(self, **_kwargs):
        return True


def test_start_crawler_stops_before_submitting_tasks(monkeypatch):
    called = {"crawl_url": 0}

    def _fake_crawl_url(*args, **kwargs):
        called["crawl_url"] += 1
        return []

    monkeypatch.setattr(web_crawler, "DBManager", _DummyDB)
    monkeypatch.setattr(web_crawler, "crawl_url", _fake_crawl_url)

    web_crawler.start_crawler(
        seed_urls=["https://example.com"],
        max_workers=2,
        max_pages=10,
        should_stop_cb=lambda: True,
    )

    assert called["crawl_url"] == 0
