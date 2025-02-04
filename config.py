class Config:
    DB_PATH = "db"
    TEST_DB_PATH = "test_db"
    DATA_PATH = "data"
    UNUSED_DATA_PATH = "unused_data"
    MODEL_NAME = "llama3.1"
    REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/56.0.2924.76 Safari/537.36',
        "Upgrade-Insecure-Requests": "1", "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate"
    }
